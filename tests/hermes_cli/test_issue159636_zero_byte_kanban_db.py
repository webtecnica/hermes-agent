"""Regression tests for #159636: a 0-byte ``kanban.db`` is not a database.

``_refuse_dead_board_resurrection`` used to return early on ``path.exists()``,
so a 0-byte stub created by an accidental read-write ``sqlite3`` open (a loop
over ``boards/*/kanban.db``, or ``?immutable=1`` which still opens with the
create flag) counted as an existing DB.  For an archived board (``board.json``
with ``"archived": true``) or a hard-deleted board (no ``board.json``) the
schema was then written into the stub and the board was silently resurrected:
cards created there never reach the dispatcher.

The guard must treat a 0-byte file like a missing one so the archived and
missing-metadata refusals still run, while a live board's own 0-byte file (the
legitimate first-init case) keeps initializing normally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture(autouse=True)
def isolated_kanban_home(tmp_path, monkeypatch):
    """Point every kanban resolver at an empty throwaway root.

    Nothing here may touch the real ``~/.hermes``: ``HERMES_KANBAN_HOME`` wins
    over ``HERMES_HOME`` for the board root, and the per-path init cache is
    reset on both sides of the test.
    """
    root = tmp_path / "hermes_home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants

        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    yield root
    kb._INITIALIZED_PATHS.clear()


def _write_zero_byte_stub(slug: str) -> Path:
    """Create the board directory if needed and leave a 0-byte kanban.db."""
    stub = kb.board_dir(slug) / "kanban.db"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_bytes(b"")
    assert stub.stat().st_size == 0
    return stub


def test_archived_board_with_zero_byte_stub_is_refused(isolated_kanban_home):
    """An archived board must be refused even when a 0-byte stub sits beside
    its tombstone, and the stub must stay untouched (no schema, no init lock).
    """
    slug = "archived-board"
    kb.create_board(slug)
    kb.remove_board(slug, archive=True)
    assert kb.read_board_metadata(slug)["archived"] is True

    stub = _write_zero_byte_stub(slug)
    init_lock = stub.with_name(stub.name + ".init.lock")
    # create_board() above initialized the DB whose file the archive moved.
    kb._INITIALIZED_PATHS.clear()

    with pytest.raises(ValueError, match="is archived"):
        kbc.connect(board=slug)

    assert stub.stat().st_size == 0, "the refused stub must not receive the schema"
    assert not init_lock.exists(), "no init lock may be taken for a refused board"

    # The refusal must leave the path usable by explicit creation.
    meta = kb.create_board(slug)
    assert meta["archived"] is False
    with kbc.connect_closing(board=slug) as conn:
        assert kbc._schema_is_present(conn)


def test_deleted_board_with_zero_byte_stub_is_refused(isolated_kanban_home):
    """A hard-deleted board (no board.json) must be refused even when a 0-byte
    stub re-creates its directory, and the stub must stay untouched.
    """
    slug = "deleted-board"
    kb.create_board(slug)
    kb.remove_board(slug, archive=False)
    assert not kb.board_metadata_path(slug).exists()

    stub = _write_zero_byte_stub(slug)
    kb._INITIALIZED_PATHS.clear()

    with pytest.raises(ValueError, match="does not exist.*boards create"):
        kbc.connect(board=slug)

    assert stub.stat().st_size == 0, "the refused stub must not receive the schema"

    # Explicit creation still works at the same slug afterwards.
    kb.create_board(slug)
    with kbc.connect_closing(board=slug) as conn:
        assert kbc._schema_is_present(conn)


def test_live_board_with_zero_byte_stub_still_initializes(isolated_kanban_home):
    """A 0-byte file on a live board is the legitimate first-init case: the
    guard must not refuse it and connect() must write the schema.
    """
    slug = "live-board"
    kb.create_board(slug)
    stub = kb.kanban_db_path(board=slug)
    stub.write_bytes(b"")
    assert stub.stat().st_size == 0
    kb._INITIALIZED_PATHS.clear()

    with kbc.connect_closing(board=slug) as conn:
        assert kbc._schema_is_present(conn)
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

    assert stub.stat().st_size > 0


def test_live_board_with_populated_db_is_opened_not_recreated(isolated_kanban_home):
    """A live board with a real DB must keep opening as a plain open: same
    inode, same size, same mtime — nothing re-created behind the caller.
    """
    slug = "populated-board"
    kb.create_board(slug)
    path = kb.kanban_db_path(board=slug)
    kb._INITIALIZED_PATHS.clear()

    with kbc.connect_closing(board=slug) as first:
        assert kbc._schema_is_present(first)
        before = path.stat()
        with kbc.connect_closing(board=slug) as second:
            assert kbc._schema_is_present(second)
        after = path.stat()

    assert after.st_ino == before.st_ino, "connect() must not replace the file"
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
