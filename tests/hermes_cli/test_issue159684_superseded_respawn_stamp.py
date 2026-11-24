"""#159684: a superseded rate-limit stamp must not re-match the respawn blocker.

The rate-limit requeue path stamps ``tasks.last_failure_error`` with
quota-flavored prose and deliberately does not count it as a failure. Nothing
ever removed that prose, so when the retry reached ``kanban_request_review``
the card sat in ``review`` carrying a ``last_failure_error`` that still matched
``_RESPAWN_BLOCKER_RE``: ``check_respawn_guard(lane="review")`` answered
``"blocker_auth"`` on every tick and the reviewer was never spawned, even
though the handoff run had just cleared the quota wall. The dispatcher also
appended a ``respawn_guarded`` event per tick, forever, for the same unchanged
condition.

Contracts pinned here:

* ``request_review`` clears ``last_failure_error`` in the same UPDATE as the
  review transition and leaves ``consecutive_failures`` alone (only
  ``complete_task`` resets that counter).
* Once the newest run is ``review_requested`` the review lane guard no longer
  answers ``blocker_auth`` for the superseded stamp.
* Two dispatch ticks against the same unchanged guard condition write ONE
  ``respawn_guarded`` event, like ``skipped_nonspawnable`` already does.
* A real quota wall with no later run still guards: the fix must not empty the
  guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

# The prose the rate-limit requeue stamps (see _record_rate_limit_exit).
RATE_LIMIT_STAMP = (
    "pid 4242 exited rate-limited (quota wall) — requeued without counting a failure"
)
# A genuine quota/auth wall: no run has happened since, so it must keep guarding.
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
    """Persist a failure stamp the way the crash / rate-limit paths do."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_failure_error = ?, consecutive_failures = ? WHERE id = ?",
            (error, consecutive_failures, tid),
        )


def _handoff_after_stamp(conn, *, stamp: str, consecutive_failures: int = 2) -> str:
    """A claimed card carrying ``stamp`` that completes the review handoff."""
    tid = kb.create_task(conn, title="handoff after a rate limit", assignee="worker")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    _stamp_failure(conn, tid, stamp, consecutive_failures=consecutive_failures)
    ok = kb.request_review(
        conn, tid,
        summary="implementation complete",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is True
    return tid


# ---------------------------------------------------------------------------
# (a) request_review drops the superseded stamp and keeps the failure counter
# ---------------------------------------------------------------------------


def test_request_review_clears_a_superseded_rate_limit_stamp(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        tid = _handoff_after_stamp(conn, stamp=RATE_LIMIT_STAMP, consecutive_failures=2)

        row = _row(conn, tid)
        assert row["status"] == "review"
        # The handoff run supersedes the requeue prose: it is no longer the
        # card's latest condition.
        assert row["last_failure_error"] is None
        # But reaching review is not proof the pathology cleared.
        assert row["consecutive_failures"] == 2
        # Still not a block: recurrence accounting untouched.
        assert row["block_kind"] is None
        assert (row["block_recurrences"] or 0) == 0


# ---------------------------------------------------------------------------
# (b) the review lane spawns the reviewer instead of parking the card
# ---------------------------------------------------------------------------


def test_review_lane_guard_ignores_the_stamp_once_the_handoff_run_is_newest(
    kanban_home: Path,
) -> None:
    with kbc.connect() as conn:
        tid = _handoff_after_stamp(conn, stamp=RATE_LIMIT_STAMP, consecutive_failures=2)

        # Premise: the prose would still trip the regex if it survived.
        assert kbd._RESPAWN_BLOCKER_RE.search(RATE_LIMIT_STAMP) is not None

        latest = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC, id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert latest is not None and latest["outcome"] == "review_requested"

        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


# ---------------------------------------------------------------------------
# (c) respawn_guarded is written once per unchanged guard condition
# ---------------------------------------------------------------------------


def test_repeated_ticks_emit_one_respawn_guarded_event(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two ticks, same card, same guard reason -> ONE diagnostic event."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    spawned: list[str] = []

    def spawn(task, workspace, board=None):  # pragma: no cover - must not run
        spawned.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota wall", assignee="worker")
        _stamp_failure(conn, tid, QUOTA_STAMP, consecutive_failures=1)

        first = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert dict(first.respawn_guarded).get(tid) == "blocker_auth"
        second = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert dict(second.respawn_guarded).get(tid) == "blocker_auth"

        assert spawned == [], "a guarded card must never be spawned"
        # Dedupe suppresses the duplicate row, not the guard itself.
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
