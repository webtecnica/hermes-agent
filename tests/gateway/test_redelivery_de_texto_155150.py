"""A WhatsApp text redelivered under the same provider id is the same admission (#155150).

The bridge re-sends an inbound whose ACK was late — a busy follow-up's turn is still answering, so
the frame is re-delivered seconds later. The adapter builds a fresh ``MessageEvent`` per frame and
leaves ``timestamp`` at its default (``datetime.now()``), so the second delivery prepares an
envelope stamped at its own arrival. That stamp is transport, not identity: the resend must resolve
to the row the first delivery committed (one FIFO entry, no ``admission_conflict``), while the same
provider id carrying different text — under a new ``:edit:`` revision or not — is still refused.
"""
from datetime import UTC, datetime

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

CHAT = '15551234567@s.whatsapp.net'
MESSAGE_ID = 'wamid.redelivery.1'
EDIT = '1800000010'


async def _authority(tmp_path, monkeypatch, name='redelivery'):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id=name)
    runner._wire_adapter_handlers(adapter)
    return authority, adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')


def _event(source, text, *, redelivered, revision=None):
    """One bridge frame: same provider id and text, built at its own arrival instant."""
    at = datetime(2026, 11, 13, 18, 9, 9 if redelivered else 1, tzinfo=UTC)
    return MessageEvent(text=text, source=source, message_id=MESSAGE_ID, timestamp=at,
                        edit_revision=revision)


def _rows(authority, session_id):
    return list_session_admissions(authority.db, session_id=session_id, pending_only=False)


@pytest.mark.asyncio
async def test_redelivered_text_resolves_the_admission_its_first_delivery_committed(tmp_path, monkeypatch):
    authority, source = await _authority(tmp_path, monkeypatch)
    try:
        first = await authority.admit_native(_event(source, 'are you there?', redelivered=False))
        # The ACK was lost while the earlier turn replied: the bridge re-sends the same message.
        second = await authority.admit_native(_event(source, 'are you there?', redelivered=True))
        rows = _rows(authority, first.ref.session_id)
        assert second.admission_id == first.admission_id, 'the resend was not the committed admission'
        assert second.status == 'queued'
        assert [row['request_id'] for row in rows] == [MESSAGE_ID], 'the message must occupy one FIFO entry'
        # The committed envelope keeps the arrival stamp of the delivery that committed it.
        assert rows[0]['payload']['native_text_v1']['timestamp'] == '2026-11-13T18:09:01+00:00'

        # An edit is its own input; a redelivery of that edit is a replay of it, re-stamped or not.
        edit = await authority.admit_native(_event(source, 'hello', redelivered=False, revision=EDIT))
        again = await authority.admit_native(_event(source, 'hello', redelivered=True, revision=EDIT))
        assert again.admission_id == edit.admission_id
        assert [row['request_id'] for row in _rows(authority, first.ref.session_id)] == [
            MESSAGE_ID, MESSAGE_ID + ':edit:' + EDIT]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_changed_text_under_the_same_provider_id_still_conflicts(tmp_path, monkeypatch):
    authority, source = await _authority(tmp_path, monkeypatch, name='changed')
    try:
        first = await authority.admit_native(_event(source, 'are you there?', redelivered=False))
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_native(_event(source, 'are you still there?', redelivered=True))
        # A new edit revision is a different input, not a conflict ...
        revise = await authority.admit_native(
            _event(source, 'are you still there?', redelivered=True, revision='1800000020'))
        assert revise.admission_id != first.admission_id
        # ... while changed text under an already-admitted revision is still refused.
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await authority.admit_native(_event(source, 'and now?', redelivered=False, revision='1800000020'))
        assert [row['request_id'] for row in _rows(authority, first.ref.session_id)] == [
            MESSAGE_ID, MESSAGE_ID + ':edit:1800000020']
    finally:
        authority.db.close()
