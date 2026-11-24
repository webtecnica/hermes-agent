"""#159684: a superseded rate-limit stamp must not re-match the respawn blocker.

When a worker run ends rate-limited, the requeue path stamps
``tasks.last_failure_error`` with quota-flavored prose (deliberately *not*
counted as a failure). Nothing removed that stamp afterwards, so the moment the
retry reached ``kanban_request_review`` the card sat in ``review`` with a
``last_failure_error`` that still matched ``_RESPAWN_BLOCKER_RE``:
``check_respawn_guard(lane="review")`` answered ``"blocker_auth"`` forever and
the reviewer was never spawned — a handoff that already cleared the quota wall
was parked as if it had hit it. The dispatcher also wrote one
``respawn_guarded`` event per tick with no dedupe, one row per tick on a board
that never changes.

Contracts pinned here:

* ``request_review`` clears ``last_failure_error`` in the review transition and
  leaves ``consecutive_failures`` alone (only ``complete_task`` resets it).
* The review-lane guard no longer sees the superseded stamp once the newest run
  is ``review_requested``.
* Two dispatch ticks against the same unchanged guard condition emit ONE
  ``respawn_guarded`` event, like ``skipped_nonspawnable`` already does.
* A real quota wall with no later run still guards — the fix must not empty the
  guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# The exact prose the rate-limit requeue stamps (see _record_rate_limit_exit).
RATE_LIMIT_STAMP = (
    "pid 4242 exited rate-limited (quota wall) — requeued without counting a failure"
)
# A real quota/auth wall: no run of this card has run since, so it must keep guarding.
QUOTA_STAMP = "403 forbidden: invalid api key for profile dev"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _row(conn, tid):
    return conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, block_kind, "
        "block_recurrences FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
        (tid,),
    ).fetchall()
    out = [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
        for r in rows
    ]
    if kind is not None:
        out = [e for e in out if e[0] == kind]
    return out


def _stamp_failure(conn, tid, error: str, *, consecutive_failures: int) -> None:
    """Persist a failure stamp the way the crash/rate-limit paths do."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_failure_error = ?, consecutive_failures = ? WHERE id = ?",
            (error, consecutive_failures, tid),
        )


def _handoff_card(conn, *, stamp: str, consecutive_failures: int = 2):
    """A running card carrying ``stamp`` that completes its review handoff."""
    tid = kb.create_task(conn, title="handoff after a rate limit", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    _stamp_failure(conn, tid, stamp, consecutive_failures=consecutive_failures)
    ok = kb.request_review(
        conn, tid, summary="implementation complete",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is True
    return tid


# ---------------------------------------------------------------------------
# (a) request_review drops the superseded stamp, keeps the failure counter
# ---------------------------------------------------------------------------


def test_request_review_clears_a_superseded_rate_limit_stamp(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = _handoff_card(conn, stamp=RATE_LIMIT_STAMP, consecutive_failures=2)

        row = _row(conn, tid)
        assert row["status"] == "review"
        # The handoff run supersedes the requeue's prose: it is no longer the
        # card's latest condition.
        assert row["last_failure_error"] is None
        # ... but a review transition is not evidence the pathology cleared.
        assert row["consecutive_failures"] == 2
        # Still not a block: recurrence accounting untouched.
        assert row["block_kind"] is None
        assert (row["block_recurrences"] or 0) == 0


# ---------------------------------------------------------------------------
# (b) the review lane spawns the reviewer instead of parking the card
# ---------------------------------------------------------------------------


def test_review_lane_guard_ignores_the_stamp_once_the_handoff_run_is_newest(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    with kbc.connect() as conn:
        tid = _handoff_card(conn, stamp=RATE_LIMIT_STAMP, consecutive_failures=2)

        latest = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC, id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert latest is not None and latest["outcome"] == "review_requested"

        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
        # And the dispatch tick actually spends its slot on the reviewer.
        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) is None


# ---------------------------------------------------------------------------
# (c) respawn_guarded is written once per unchanged guard condition
# ---------------------------------------------------------------------------


def test_repeated_ticks_emit_one_respawn_guarded_event(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ticks, same card, same guard reason -> ONE diagnostic event.

    ``skipped_nonspawnable`` already dedupes against the card's last event; the
    respawned-guard row ran unguarded and appended one row per tick forever.
    """
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    spawns: list[str] = []

    def spawn(task, workspace, board=None):  # pragma: no cover - must not run
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wall", assignee="worker")
        _stamp_failure(conn, tid, QUOTA_STAMP, consecutive_failures=1)

        first = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert dict(first.respawn_guarded).get(tid) == "blocker_auth"
        second = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert dict(second.respawn_guarded).get(tid) == "blocker_auth"

        assert spawns == [], "a guarded card must never be spawned"
        assert len(_events(conn, tid, "respawn_guarded")) == 1


# ---------------------------------------------------------------------------
# (d) control: a real quota wall that nothing superseded still guards
# ---------------------------------------------------------------------------


def test_unsuperseded_quota_wall_still_guards_both_lanes(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        ready_tid = kb.create_task(conn, title="real quota wall", assignee="worker")
        _stamp_failure(conn, ready_tid, QUOTA_STAMP, consecutive_failures=1)
        assert kbd.check_respawn_guard(conn, ready_tid) == "blocker_auth"

        review_tid = kb.create_task(conn, title="quota wall awaiting review", assignee="reviewer")
        _stamp_failure(conn, review_tid, QUOTA_STAMP, consecutive_failures=1)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_tid,))
        assert kbd.check_respawn_guard(conn, review_tid, lane="review") == "blocker_auth"
