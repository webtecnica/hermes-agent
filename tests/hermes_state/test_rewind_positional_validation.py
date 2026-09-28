"""Regression tests for #126765:
rewind_user_turn validates warm history by user turn identity / position,
preventing stale warm prefix installation, and enforces adopt_row_ids prefix equality.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from hermes_state import SessionDB
from hermes_state_rewind import _HISTORY_CHANGED


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    handle = SessionDB(db_path=tmp_path / "state.db")
    yield handle
    handle.close()


def test_stale_warm_prefix_different_first_turn_raises(db):
    """When warm history has the same user-turn count and identical target turn
    but a different prefix turn, rewind_user_turn must refuse and leave durable state unchanged."""
    sid = "test-rewind-stale-prefix"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "original_q1")
    db.append_message(sid, "assistant", "original_a1")
    db.append_message(sid, "user", "target_q2")
    db.append_message(sid, "assistant", "target_a2")

    # Warm history has count 2 and matching target_q2, but stale/different first turn
    warm = [
        {"role": "user", "content": "stale_q1"},
        {"role": "assistant", "content": "stale_a1"},
        {"role": "user", "content": "target_q2"},
        {"role": "assistant", "content": "target_a2"},
    ]

    with pytest.raises(RuntimeError, match=_HISTORY_CHANGED):
        db.rewind_user_turn(sid, user_ordinal=1, warm_history=warm)

    # Durable messages in DB must be completely untouched
    durable = db.get_messages_as_conversation(sid)
    assert len(durable) == 4
    assert durable[0]["content"] == "original_q1"
    assert durable[2]["content"] == "target_q2"


def test_adopt_row_ids_mismatched_prefix_raises(db):
    """When adopt_row_ids=True and warm prefix has different assistant content,
    rewind_user_turn must raise RuntimeError and leave DB untouched."""
    sid = "test-rewind-adopt-mismatch"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "durable_a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "durable_a2")

    warm = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "divergent_a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "durable_a2"},
    ]

    with pytest.raises(RuntimeError, match=_HISTORY_CHANGED):
        db.rewind_user_turn(sid, user_ordinal=1, warm_history=warm, adopt_row_ids=True)

    durable = db.get_messages_as_conversation(sid)
    assert len(durable) == 4
    assert durable[1]["content"] == "durable_a1"


def test_matching_warm_history_rewinds_successfully(db):
    """Matching warm history passes validation and successfully rewinds."""
    sid = "test-rewind-matching-warm"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "q1")
    db.append_message(sid, "assistant", "a1")
    db.append_message(sid, "user", "q2")
    db.append_message(sid, "assistant", "a2")

    warm = db.get_messages_as_conversation(sid)
    outcome = db.rewind_user_turn(sid, user_ordinal=1, warm_history=warm)

    assert outcome.turns_undone == 1
    assert outcome.live_view["content"] == "q2"
    durable = db.get_messages_as_conversation(sid)
    assert len(durable) == 2
    assert durable[0]["content"] == "q1"
