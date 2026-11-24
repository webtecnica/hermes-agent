"""A navigation listing reads committed rows; accounting-aware listings still drain queued usage.

``list_sessions_rich`` drained pending token/cost deltas on every call because its rows carry the
totals columns, and projecting a compression tip drained a second time (#159573). The legacy
navigation ``session.list`` response drops those columns, so it must not wait for a usage writer
stuck behind another connection's write lock — while a receiver that DOES read the totals keeps the
read-your-writes drain.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

import tui_gateway.server as srv
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import TokenCounts, UsageDelta
from persistence.sqlite import _holder
from persistence.sqlite.conversation._store.sessions import list_sessions_rich
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder

#: Generous navigation deadline: the regression waits the holder's full 5 s flush budget per drain
#: (twice when a compression tip is projected), so 2 s separates "returned" from "still waiting"
#: with room for a slow machine.
NAVIGATION_DEADLINE_S = 2.0
#: How long the holder's own flush budget plus both drains can take before a failing call returns.
DRAIN_GRACE_S = 40.0

QUEUED_INPUT_TOKENS = 19


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A temporary SQLite store, wired as the legacy gateway's launch handle (the ``session.list`` feed)."""
    database = open_persistence(profile_scope(home=tmp_path))
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _seed(store, *, with_tip: bool) -> set[str]:
    """Two conversations; with a tip, ``root`` is a compression root whose live continuation is ``tip``."""
    store.sessions.create(NewSession(session_id="other", source="cli"))
    store.sessions.set_title("other", "old")
    store.sessions.create(NewSession(session_id="root", source="cli"))
    if with_tip:
        store.sessions.create(NewSession(session_id="tip", source="cli", parent_session_id="root"))
        raw_holder(store)._write_sql(
            "UPDATE sessions SET ended_at = started_at, end_reason = 'compression' WHERE id = 'root'")
        return {"other", "tip"}
    return {"other", "root"}


def _hold_write_lock(path, started: threading.Event, release: threading.Event) -> None:
    """A sibling connection's write transaction, uncommitted until ``release``: the writer under test blocks on it."""
    conn = sqlite3.connect(str(path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE sessions SET title = 'renamed' WHERE id = 'other'")
        started.set()
        assert release.wait(DRAIN_GRACE_S)
        conn.execute("COMMIT")
    finally:
        conn.close()


@contextmanager
def _unapplied_usage_delta(store, monkeypatch):
    """A usage delta queued while the holder's writer is claimed but cannot commit.

    Yields once the writer has claimed the batch and entered the applier (the holder's own event, not a
    sleep), so "pending" is a state, not a race. ``flush_token_counts(timeout=0) is False`` pins it.
    """
    holder = raw_holder(store)
    entered, started, release = threading.Event(), threading.Event(), threading.Event()
    apply_delta = _holder._token_delta_applier

    def signalling_applier(target, session_id, kwargs):
        entered.set()
        apply_delta(target, session_id, kwargs)

    monkeypatch.setattr(_holder, "_token_delta_applier", signalling_applier)
    sibling = threading.Thread(target=_hold_write_lock, args=(holder.db_path, started, release), daemon=True)
    sibling.start()
    try:
        assert started.wait(10.0), "the sibling connection never took the write lock"
        store.usage.queue_usage("other", UsageDelta(tokens=TokenCounts(input_tokens=QUEUED_INPUT_TOKENS), source="cli"))
        assert entered.wait(10.0), "the holder's writer never claimed the queued delta"
        assert holder.flush_token_counts(timeout=0) is False
        yield holder
    finally:
        release.set()
        sibling.join(DRAIN_GRACE_S)
        assert not sibling.is_alive()


def _start(fn, *args, **kwargs):
    """``(thread, box)`` running ``fn``; the box carries its value or its exception."""
    box: dict[str, object] = {}

    def run():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # re-raised by the assertions below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, box


def _under_deadline(deadline_s, fn, *args, **kwargs):
    """Run ``fn`` on a sync thread and require it to return inside ``deadline_s``."""
    worker, box = _start(fn, *args, **kwargs)
    started = time.monotonic()
    worker.join(deadline_s)
    if worker.is_alive():
        # Let the holder's drain budget expire so the thread exits before its store closes; the overrun
        # is the measurement below.
        worker.join(DRAIN_GRACE_S)
        elapsed = time.monotonic() - started
        pytest.fail(
            f"the call was still waiting for the usage write after {elapsed:.2f}s "
            f"(deadline {deadline_s:g}s) — a navigation read drained queued usage it does not return")
    elapsed = time.monotonic() - started
    assert "error" not in box, box.get("error")
    return box["value"], elapsed


def _navigation_rows():
    """One legacy ``session.list`` navigation call, as the Dashboard's RPC dispatcher makes it."""
    response = srv._methods["session.list"](17, {"limit": 5})
    assert "result" in response, response
    return response["result"]["sessions"]


@pytest.mark.parametrize("with_tip", [False, True], ids=["plain-page", "compression-tip"])
def test_legacy_session_list_does_not_wait_for_usage_it_never_returns(store, monkeypatch, with_tip):
    """The navigation response drops the totals, so it must not drain the usage write twice."""
    expected_ids = _seed(store, with_tip=with_tip)
    baseline = _navigation_rows()
    assert {row["id"] for row in baseline} == expected_ids
    assert not {"input_tokens", "output_tokens", "estimated_cost_usd", "actual_cost_usd"} & set(baseline[0])
    if with_tip:
        assert next(row for row in baseline if row["id"] == "tip")["_lineage_root_id"] == "root"

    with _unapplied_usage_delta(store, monkeypatch):
        rows, elapsed = _under_deadline(NAVIGATION_DEADLINE_S, _navigation_rows)
        # Order, visibility and continuation identity are the committed page: the pending delta is invisible.
        assert rows == baseline, f"navigation rows changed while usage was pending (returned in {elapsed:.2f}s)"

    # The delta was pending, not lost: the next accounting-aware read drains and sees it.
    assert store.sessions.get("other").input_tokens == QUEUED_INPUT_TOKENS


def test_rich_list_opt_out_returns_committed_rows_without_draining(store, monkeypatch):
    """``needs_usage_totals=False`` on the store read: same committed rows, no wait, totals still stale."""
    _seed(store, with_tip=False)
    with _unapplied_usage_delta(store, monkeypatch) as holder:
        rows, elapsed = _under_deadline(
            NAVIGATION_DEADLINE_S, list_sessions_rich, holder, limit=10, needs_usage_totals=False)
        assert {row["id"] for row in rows} == {"other", "root"}, f"rows changed (returned in {elapsed:.2f}s)"
        assert next(row for row in rows if row["id"] == "other")["input_tokens"] == 0


def test_rich_list_default_still_drains_usage_totals(store, monkeypatch):
    """The default keeps the read-your-writes drain: a caller reading the totals waits for the delta."""
    _seed(store, with_tip=False)
    holder = raw_holder(store)
    with _unapplied_usage_delta(store, monkeypatch):
        worker, box = _start(list_sessions_rich, holder, limit=10)
        worker.join(NAVIGATION_DEADLINE_S)
        assert worker.is_alive(), "the default listing returned while a queued usage delta was unapplied"
    worker.join(DRAIN_GRACE_S)
    assert not worker.is_alive()
    assert "error" not in box, box.get("error")
    assert next(row for row in box["value"] if row["id"] == "other")["input_tokens"] == QUEUED_INPUT_TOKENS
