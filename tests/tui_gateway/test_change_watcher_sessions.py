from __future__ import annotations

import sqlite3

from tui_gateway import server


def _seed_store(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            last_activity_at REAL,
            message_count INTEGER,
            archived INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE gateway_heartbeats (
            backend_id TEXT PRIMARY KEY,
            last_heartbeat REAL
        );
        INSERT INTO sessions(id, title, last_activity_at, message_count)
        VALUES ('session-1', 'Original', 100, 1);
        INSERT INTO gateway_heartbeats(backend_id, last_heartbeat)
        VALUES ('backend-1', 100);
        """
    )
    conn.commit()
    conn.close()


def _write_title(path, title):
    conn = sqlite3.connect(path)
    conn.execute("UPDATE sessions SET title = ? WHERE id = 'session-1'", (title,))
    conn.commit()
    conn.close()


class _Stamp:
    """A settable ``_newest_mtime_ns``: a fixed, already-settled stamp the test moves by hand.

    Pinning the mtime keeps these tests off the filesystem clock: a commit landing in the same coarse
    tick as the seeding write must not make a test pass or fail by accident (that window has its own
    test below, with an explicit racy switch).
    """

    def __init__(self, mtime=10**18):
        self.mtime = mtime

    def __call__(self, _paths):
        return self.mtime

    def bump(self, ns=1_000_000_000):
        self.mtime += ns
        return self.mtime


def _watched(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})


def test_sessions_signature_ignores_gateway_heartbeat_only_writes(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    _watched(monkeypatch, tmp_path)
    stamp = _Stamp()
    monkeypatch.setattr(server, "_newest_mtime_ns", stamp)

    before = server._sessions_sig()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE gateway_heartbeats SET last_heartbeat = 200")
    conn.commit()
    conn.close()
    stamp.bump()

    assert server._sessions_sig() == before


def test_sessions_signature_changes_for_session_metadata_updates(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    _watched(monkeypatch, tmp_path)
    stamp = _Stamp()
    monkeypatch.setattr(server, "_newest_mtime_ns", stamp)

    before = server._sessions_sig()
    _write_title(db_path, "Renamed")
    stamp.bump()

    assert server._sessions_sig() != before


def test_sessions_signature_is_stable_across_passes_without_a_write(tmp_path, monkeypatch):
    """The digest order may differ from the old ``ORDER BY id`` one (it is now the rowid scan order),
    but nothing else: two passes over an unchanged store must not look like a change, or the watcher
    would broadcast ``sessions.changed`` on every tick."""
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    _watched(monkeypatch, tmp_path)
    stamp = _Stamp()
    monkeypatch.setattr(server, "_newest_mtime_ns", stamp)

    first = server._sessions_sig()
    assert server._sessions_sig() == first
    stamp.bump()  # a stat-only pass after the mtime moved but the rows did not
    assert server._sessions_sig() == first


def test_signature_query_scans_sessions_by_rowid_without_the_id_index(tmp_path):
    """``sessions.id`` is a TEXT PRIMARY KEY, so ``ORDER BY id`` made every pass plan as
    ``SCAN sessions USING INDEX sqlite_autoindex_sessions_1`` — the autoindex plus one table seek
    per row, 115 MB of ``rchar`` per pass on a 31.7k-session install (#154960). ``sessions`` is a
    rowid table: the digest query must be the plain table scan in b-tree order — no index, no
    sorter, no per-row seek.

    Built on the real ``SCHEMA_SQL``: the production plan shape (autoindex for ``ORDER BY id``)
    only shows up on the real table shape, and this test is worth nothing on a toy one."""
    from hermes_state_common import SCHEMA_SQL

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('session-1', 'cron', 100)")
    conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('session-2', 'cli', 101)")
    conn.commit()
    available = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    sql = server._session_rows_query(conn, ("id", "title"), available)
    plan = [row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
    old_plan = [row[-1] for row in conn.execute(
        "EXPLAIN QUERY PLAN SELECT id, title FROM sessions ORDER BY id")]
    conn.close()

    assert sql.endswith(" ORDER BY rowid"), sql
    assert any("SCAN" in step and "sessions" in step for step in plan), plan
    assert not any("USING INDEX" in step or "autoindex" in step for step in plan), plan
    assert not any("TEMP B-TREE" in step for step in plan), plan
    assert any("autoindex_sessions_1" in step for step in old_plan), old_plan


def test_signature_query_keeps_id_order_for_a_without_rowid_store():
    """A ``WITHOUT ROWID`` sessions table (or a declared ``rowid`` column shadowing the implicit key)
    has no stable rowid order to scan, so the query keeps the old ``id`` order instead of failing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT) WITHOUT ROWID")
    available = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}

    assert server._session_rows_query(conn, ("id", "title"), available).endswith(" ORDER BY id")
    conn.close()


def test_sessions_signature_sees_a_commit_inside_the_same_mtime_tick(tmp_path, monkeypatch):
    """A commit landing in the tick of the previous read leaves the DB/WAL mtime unchanged, so the
    watcher cannot see it on the next 0.5 s pass without re-scanning the table on every tick — the
    cost #154960 is about. It answers from cache while the mtime is racily clean and confirms with one
    read once it settles (``utils.RACY_MTIME_WINDOW_NS`` = 2 s).

    The wait is intentional and bounded by that window, which sits under the 2 s
    ``sessions.changed`` broadcast floor: the observable guarantee is "a same-tick commit shows up
    within ~2 s", not "on the very next tick"."""
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    _watched(monkeypatch, tmp_path)
    stamp = _Stamp()
    monkeypatch.setattr(server, "_newest_mtime_ns", stamp)
    racy = [True]
    monkeypatch.setattr(server, "_mtime_racy", lambda _mtime: racy[0])

    before = server._sessions_sig()
    _write_title(db_path, "Renamed")  # the mtime does not move: same tick

    assert server._sessions_sig() == before, "inside the racy window the cached digest answers"
    racy[0] = False  # ~2 s later the tick has settled
    assert server._sessions_sig() != before, "the confirmation read sees the same-tick commit"
    assert server._sessions_sig() != before, "and the confirmed digest is not re-read away"


def test_racy_passes_answer_from_cache_and_confirm_once_reads_the_rows_once(tmp_path, monkeypatch):
    """The re-scan per tick is the cost: while the mtime is racily clean and unchanged, consecutive
    passes must return the same digest *without* reading the rows; the settle does exactly one
    confirmation read."""
    db_path = tmp_path / "state.db"
    _seed_store(db_path)
    _watched(monkeypatch, tmp_path)
    monkeypatch.setattr(server, "_newest_mtime_ns", _Stamp())
    racy = [True]
    monkeypatch.setattr(server, "_mtime_racy", lambda _mtime: racy[0])
    reads = []
    real_read = server._session_rows_sig

    def counting_read(conn, mtime):
        reads.append(mtime)
        return real_read(conn, mtime)

    monkeypatch.setattr(server, "_session_rows_sig", counting_read)

    first = server._sessions_sig()
    for _ in range(4):  # 4 further 0.5 s ticks inside the 2 s window
        assert server._sessions_sig() == first
    assert len(reads) == 1, "a pass inside the racy window must not re-scan the sessions table"

    racy[0] = False  # the mtime settled
    assert server._sessions_sig() == first
    assert len(reads) == 2, "exactly one confirmation read when the mtime settles"
    assert server._sessions_sig() == first
    assert len(reads) == 2, "a confirmed digest is cached like any other"
