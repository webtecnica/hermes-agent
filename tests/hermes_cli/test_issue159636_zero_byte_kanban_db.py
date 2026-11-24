"""Regression tests for #159636: a 0-byte ``kanban.db`` is not a database.

``_refuse_dead_board_resurrection()`` used to bail out on any existing path
(``if path.exists(): return``), so a zero-byte ``kanban.db`` left beside an
archived tombstone — or inside a hard-deleted board's leftover directory —
made the archived/missing-board refusals unreachable. ``connect()`` then wrote
the full schema into the stub, silently resurrecting the dead board.

Every DB here lives under ``tmp_path`` (via ``HERMES_KANBAN_HOME``); the tests
never touch the real home.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not a stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated kanban home: no ambient board/DB pins, empty init cache."""
    home = tmp_path / "kanban_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
        "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(var, raising=False)
    # Module-level init cache must not leak between tests.
    with kbc._INIT_LOCK:
        kbc._INITIALIZED_PATHS.clear()
    yield home
    with kbc._INIT_LOCK:
        kbc._INITIALIZED_PATHS.clear()


def _board_dir(home: Path, slug: str) -> Path:
    return home / "kanban" / "boards" / slug


def _seed_board_dir(home: Path, slug: str, *, archived: bool | None) -> Path:
    """Create ``boards/<slug>/`` with a 0-byte ``kanban.db`` stub.

    ``archived=None`` leaves no ``board.json`` (the hard-deleted shape);
    otherwise a metadata file is written with the requested ``archived`` flag.
    """
    d = _board_dir(home, slug)
    d.mkdir(parents=True)
    if archived is not None:
        (d / "board.json").write_text(
            json.dumps({"slug": slug, "archived": archived}) + "\n",
            encoding="utf-8",
        )
    (d / "kanban.db").write_bytes(b"")
    return d


def _assert_untouched_stub(d: Path) -> None:
    db = d / "kanban.db"
    assert db.exists()
    assert db.stat().st_size == 0, (
        f"a refusal must not write the schema into the 0-byte stub; "
        f"{db} grew to {db.stat().st_size} bytes"
    )
    assert not (d / "kanban.db.init.lock").exists(), (
        "the init lock must not be taken for a refused resurrection"
    )


class TestZeroByteStubIsNotAnExistingDb:
    def test_archived_board_with_zero_byte_stub_is_refused(self, kanban_home):
        d = _seed_board_dir(kanban_home, "archived-stub", archived=True)

        with pytest.raises(ValueError, match="is archived"):
            kbc.connect(board="archived-stub").close()

        _assert_untouched_stub(d)

    def test_missing_board_with_zero_byte_stub_is_refused(self, kanban_home):
        d = _seed_board_dir(kanban_home, "deleted-stub", archived=None)

        with pytest.raises(ValueError, match="does not exist|boards create"):
            kbc.connect(board="deleted-stub").close()

        _assert_untouched_stub(d)


class TestLiveBoardsStillInitialize:
    def test_live_board_with_zero_byte_db_self_initializes(self, kanban_home):
        d = _seed_board_dir(kanban_home, "live-empty", archived=False)

        with kbc.connect_closing(board="live-empty") as conn:
            assert kbc._schema_is_present(conn) is True

        assert (d / "kanban.db").stat().st_size > 0

    def test_live_board_with_populated_db_is_opened_in_place(self, kanban_home):
        kb.create_board("live-populated")
        db = kb.kanban_db_path("live-populated")
        before = db.stat()

        with kbc.connect_closing(board="live-populated") as conn:
            assert kbc._schema_is_present(conn) is True
            conn.execute("SELECT 1").fetchone()

        after = db.stat()
        assert after.st_ino == before.st_ino, "connect() must not recreate a populated DB"
        assert after.st_mtime_ns == before.st_mtime_ns, "connect() must not rewrite a populated DB"
        assert after.st_size == before.st_size
