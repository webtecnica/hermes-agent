"""A 0-byte ``kanban.db`` stub must not resurrect an archived/deleted board.

``connect()``'s resurrect guard (#43243) bailed out on *any* existing path
(``if path.exists(): return``), so a zero-byte ``kanban.db`` left behind beside
an archived tombstone (or in the directory of a hard-deleted board) made the
archived/missing-metadata refusals below it unreachable: sqlite happily opens an
empty file, the init path writes the full schema into it and the dead board is
silently usable again (#159636).

The stub is still not a board — it has no schema, no metadata and no rows — so
the guard now treats a 0-byte file as a missing one and lets the refusals run.
A live board whose DB is legitimately 0 bytes (fresh home, crashed first init)
still self-initializes: it has no tombstone and no archived metadata.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# Ensure the worktree (not a stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
import sys

if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Throwaway ``HERMES_HOME``: no prior kanban state, no board env pins."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_TASK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    # Drop any memoized root from a previous test's HERMES_HOME.
    try:
        import hermes_constants

        hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    yield home
    kb._INITIALIZED_PATHS.clear()


def _assert_full_schema(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"tasks", "task_runs", "task_events", "task_links"} <= tables


def _zero_byte_stub(board: str) -> Path:
    """Recreate the historical stub shape: an empty ``kanban.db`` only."""
    stub = kb.board_dir(board) / "kanban.db"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.touch()
    assert stub.stat().st_size == 0
    return stub


def test_archived_board_with_zero_byte_stub_is_refused(fresh_home):
    """Archived tombstone + 0-byte stub: refuse, do not write the schema."""
    kb.create_board("gone")
    kb.remove_board("gone", archive=True)
    assert kb.read_board_metadata("gone")["archived"] is True
    stub = _zero_byte_stub("gone")

    with pytest.raises(ValueError, match="is archived; refusing to recreate its database"):
        kbc.connect(board="gone")

    # The stub is untouched: no schema, no sidecars, no init lock.
    assert stub.stat().st_size == 0
    assert not stub.with_name(stub.name + ".init.lock").exists()
    assert not stub.with_name(stub.name + "-wal").exists()
    assert not stub.with_name(stub.name + "-shm").exists()


def test_hard_deleted_board_with_zero_byte_stub_is_refused(fresh_home):
    """Hard-deleted board (no board.json) + 0-byte stub: refused as missing."""
    kb.create_board("nuked")
    kb.remove_board("nuked", archive=False)
    assert not kb.board_metadata_path("nuked").exists()
    stub = _zero_byte_stub("nuked")

    with pytest.raises(ValueError, match="does not exist"):
        kbc.connect(board="nuked")

    assert stub.stat().st_size == 0
    assert not stub.with_name(stub.name + ".init.lock").exists()


def test_live_board_with_zero_byte_db_still_initializes(fresh_home):
    """A live board's empty DB is a crashed first init, not a resurrection."""
    kb.create_board("live")
    db = kb.board_dir("live") / "kanban.db"
    for sidecar in (db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        sidecar.unlink(missing_ok=True)
    db.write_bytes(b"")
    kb._INITIALIZED_PATHS.clear()
    assert db.stat().st_size == 0

    with kbc.connect_closing(board="live") as conn:
        _assert_full_schema(conn)
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) "
            "VALUES ('t_1', 'first', 'todo', 0)"
        )

    assert db.stat().st_size > 0
    with kbc.connect_closing(board="live") as conn:
        assert [row[0] for row in conn.execute("SELECT id FROM tasks")] == ["t_1"]


def test_default_board_still_self_creates(fresh_home):
    """board=None keeps creating the default DB in a fresh home."""
    with kbc.connect_closing() as conn:
        _assert_full_schema(conn)

    assert kb.kanban_db_path().exists()
    assert kb.kanban_db_path().stat().st_size > 0


def test_archived_board_with_real_database_still_opens(fresh_home, tmp_path):
    """A non-empty DB is a plain open even beside an archived tombstone."""
    kb.create_board("keepdb")
    src = kb.board_dir("keepdb") / "kanban.db"
    snapshot = tmp_path / "snapshot.db"
    with sqlite3.connect(str(src)) as source, sqlite3.connect(str(snapshot)) as dest:
        source.backup(dest)
    kb.remove_board("keepdb", archive=True)
    assert kb.read_board_metadata("keepdb")["archived"] is True

    restored = kb.board_dir("keepdb") / "kanban.db"
    restored.write_bytes(snapshot.read_bytes())
    kb._INITIALIZED_PATHS.clear()
    assert restored.stat().st_size > 0

    with kbc.connect_closing(board="keepdb") as conn:
        _assert_full_schema(conn)


def test_connect_closing_refuses_archived_board_with_zero_byte_stub(fresh_home):
    """connect_closing() shares the guard with connect()."""
    kb.create_board("gone")
    kb.remove_board("gone", archive=True)
    stub = _zero_byte_stub("gone")

    with pytest.raises(ValueError, match="is archived; refusing to recreate its database"):
        with kbc.connect_closing(board="gone"):
            pytest.fail("connect_closing() opened a resurrected archived board")

    assert stub.stat().st_size == 0
