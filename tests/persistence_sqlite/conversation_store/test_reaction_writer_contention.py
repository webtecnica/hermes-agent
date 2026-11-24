"""An empty per-turn reaction scan must not hold the session DB writer (#159549).

The collector runs on every user turn. On the reported upstream it entered ``_execute_write``
before knowing whether there was anything to claim, so an empty history scan held the store's
single write mutex and starved writes from unrelated conversations. These tests pin the fixed
contract: no candidate row -> no writer, and a real reaction is still marked seen exactly once.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from threading import Event

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    set_message_reaction,
    take_unseen_reactions,
)
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        reacting = create_session(db, "reacting-session", "test")
        other = create_session(db, "other-session", "test")
        target = append_message(db, reacting, "assistant", "the answered message")
        for _ in range(20):
            append_message(db, reacting, "assistant", "an older answer")
        yield db, reacting, other, target
    finally:
        db.close()


def _observing_read_ctx(db, pause_scan):
    """Wrap ``_read_ctx`` so the trace hook also sees pooled readers, not just the writer."""
    real_read_ctx = db._read_ctx

    @contextmanager
    def observing_read_ctx(**kwargs):
        with real_read_ctx(**kwargs) as conn:
            if conn is not None and conn is not db._conn:
                conn.set_trace_callback(pause_scan)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)
            else:
                yield conn

    return observing_read_ctx


def test_empty_reaction_scan_does_not_block_another_session_commit(store, monkeypatch):
    db, reacting, other, _ = store
    scanning, release = Event(), Event()

    def pause_scan(sql):
        # Match the reaction scan wherever it runs: the writer on the unfixed path, a pooled
        # reader after the fix. Nothing else in this flow reads reactions.
        if "$.reactions" in sql:
            scanning.set()
            release.wait(8)

    db._conn.set_trace_callback(pause_scan)
    monkeypatch.setattr(db, "_read_ctx", _observing_read_ctx(db, pause_scan))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            collector = pool.submit(take_unseen_reactions, db, reacting)
            try:
                assert scanning.wait(4), "did not reach the reaction scan"
                writer = pool.submit(append_message, db, other, "user", "a concurrent write")
                # The unrelated session must commit while the empty scan is still held.
                row_id = writer.result(timeout=3)
                with closing(sqlite3.connect(f"file:{db.db_path.as_posix()}?mode=ro", uri=True)) as observer:
                    assert observer.execute(
                        "SELECT content FROM messages WHERE id = ?", (row_id,)
                    ).fetchone() == ("a concurrent write",)
                assert not collector.done(), "the empty scan must still be held at the barrier"
            finally:
                release.set()
            assert collector.result(timeout=4) == []
    finally:
        db._conn.set_trace_callback(None)


def test_empty_reaction_scan_never_calls_execute_write(store, monkeypatch):
    db, reacting, _, _ = store
    writes = []
    write = db._execute_write

    def record_write(fn, **kwargs):
        writes.append(fn)
        return write(fn, **kwargs)

    monkeypatch.setattr(db, "_execute_write", record_write)
    assert take_unseen_reactions(db, reacting) == []
    assert writes == [], "an empty reaction scan must not enter the writer"


def test_real_reaction_is_marked_seen_exactly_once(store):
    db, reacting, _, target = store
    assert set_message_reaction(db, reacting, target, "👍")

    pending = take_unseen_reactions(db, reacting)
    assert [(item["row_id"], item["role"], item["emoji"], item["text"]) for item in pending] == [
        (target, "assistant", "👍", "the answered message"),
    ]
    assert take_unseen_reactions(db, reacting) == []
