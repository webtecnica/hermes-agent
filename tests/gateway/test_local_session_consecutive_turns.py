"""Tests for consecutive turns on local sessions (#143537).

A live session's source records the triggering message_id of the previous turn,
which must not be treated as a receipt identity mismatch.
"""
import logging
import pytest

from gateway.session_contract import Principal
from gateway.session_local import create_local_session
from gateway.session_local_recovery import restore_local_session
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_local_recovery_pre_profile_origin import _served_authority


@pytest.mark.asyncio
async def test_local_session_answers_consecutive_turns_with_message_id_stamped(tmp_path, monkeypatch, caplog):
    """#143537: A local session answers turn 1, stamping message_id on live.source.
    Subsequent calls to restore_local_session must not fail with storage_unavailable
    due to message_id differing from the initial receipt source."""
    served, _db, secondary = await _served_authority(tmp_path, monkeypatch, 'inst1')
    actor = Principal('owner', str(secondary), frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(
        served, actor,
        {'request_id': 'consecutive_turns', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []},
    )

    # First turn: fresh live session
    live = served.sessions[ref.session_id]
    assert live.source.message_id is None
    res1 = restore_local_session(served, ref.session_id)
    assert res1.session_id == ref.session_id

    # Turn 1 completes: inbound turn stamped message_id on live.source
    live.source.message_id = 'msg-turn-1-abc123'
    assert 'message_id' in live.source.to_dict()

    # Turn 2: restore_local_session must succeed despite live.source having message_id
    res2 = restore_local_session(served, ref.session_id)
    assert res2.session_id == ref.session_id

    # Turn 2 completes: updated message_id
    live.source.message_id = 'msg-turn-2-def456'
    res3 = restore_local_session(served, ref.session_id)
    assert res3.session_id == ref.session_id


@pytest.mark.asyncio
async def test_local_session_refuses_when_live_source_identity_differs(tmp_path, monkeypatch, caplog):
    """#143537: Real source differences (e.g. user_id or platform tampered) still fail
    and log the mismatch warning."""
    served, _db, secondary = await _served_authority(tmp_path, monkeypatch, 'inst2')
    actor = Principal('owner', str(secondary), frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(
        served, actor,
        {'request_id': 'identity_check', 'source': 'gui', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []},
    )

    live = served.sessions[ref.session_id]
    live.source.user_id = 'tampered:999'

    with caplog.at_level(logging.WARNING, logger='gateway.session_local_recovery'):
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            restore_local_session(served, ref.session_id)

    warnings = [r.getMessage() for r in caplog.records if 'live source differs from its receipt' in r.getMessage()]
    assert len(warnings) == 1
    assert ref.session_id in warnings[0]
