"""A steer that outlives its turn is a user row: it keeps its ``reply_expected`` (#147041).

With ``display.busy_input_mode: steer`` a message that is *not* addressed to the bot is injected
with ``reply_expected=False``: when the model answers it with a bare silence marker the session
stays quiet. When the turn ends before the next tool batch can deliver that steer, the leftover
drain hands the text back and it becomes the NEXT user turn — and that turn used to be created
with no ``reply_expected`` at all, so an unknown ``None`` replaced the known ``False``:
``silence_allowed`` rejected the legitimate marker and the chat got the visible "unexpected
silence" fallback for a message that never asked for an answer.

The turn result now carries the drained batch's admission flag
(``pending_steer_reply_expected``) and the follow-up turn inherits it.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.response_filters import silence_allowed
from gateway.turn_context import TurnContext
from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run as _finalize
from tests.gateway.test_internal_event_pin_wiring import (
    KEY,
    _capture,
    _human_source,
    _make_runner,
)


class _SteerAgent(_StubAgent, InterruptControlMixin):
    """The real steer buffer (``InterruptControlMixin``) on the surface ``finalize_turn`` reads."""

    def __init__(self):
        super().__init__(raise_in=())
        self._pending_steer = None
        self._pending_steer_parts = []
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect_lock = threading.Lock()
        self._turn_user_intervened = False


class Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def test_the_turn_result_reports_the_leftover_steers_reply_expected():
    """The finalizer's leftover handoff is where the metadata was being dropped."""
    agent = _SteerAgent()
    assert agent.steer("side chatter about the deploy", reply_expected=False) is True

    result = _finalize(agent)

    assert result["pending_steer"] == "side chatter about the deploy"
    assert result["pending_steer_reply_expected"] is False


def test_a_steer_without_metadata_leaves_the_turn_result_as_before():
    """The new parameter defaults to ``None``: no flag is invented for callers that pass none."""
    agent = _SteerAgent()
    assert agent.steer("side chatter about the deploy") is True

    result = _finalize(agent)

    assert result["pending_steer"] == "side chatter about the deploy"
    assert "pending_steer_reply_expected" not in result
    # The predicate itself is untouched: only a known False authorizes silence, never an unknown.
    assert silence_allowed(None, False) is True
    assert silence_allowed(None, None) is False


@pytest.mark.asyncio
async def test_unaddressed_leftover_steer_stays_silent_in_the_followup_turn(monkeypatch, caplog):
    runner = _make_runner(monkeypatch)
    calls: list[dict] = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    source = _human_source()
    agent = _SteerAgent()

    # 1. Busy session, message not addressed to the bot: steer mode admits it with reply_expected=False.
    event = MessageEvent(
        text="side chatter about the deploy",
        source=source,
        message_id="msg-77",
        reply_expected=False,
    )
    runner._hm_busy_steer(event, agent, KEY)
    assert agent._pending_steer and "side chatter about the deploy" in agent._pending_steer

    # 2. The turn ends before the next tool batch delivers it; the finalizer hands the steer back.
    result = _finalize(agent)
    leftover = result["pending_steer"]
    assert leftover and "side chatter about the deploy" in leftover

    # 3. The gateway makes it the next user turn — which is a user row, not machinery.
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, KEY)
    assert pending_event is None and pending == leftover
    ctx = TurnContext(
        source=source, session_key=KEY, session_id="sess-wiring", run_generation=1, history=[],
    )
    merged = await runner._run_agent_queued_followup(
        ctx, adapter, pending, pending_event, result, result, None,
    )

    # 4. The model answers that turn with a bare marker: the terminal verdict is silence, no warning.
    entry = runner.session_store.get_or_create_session.return_value
    terminal = {
        **merged,
        "final_response": "[SILENT]",
        "completed": False,
        "messages": [
            {"role": "user", "content": pending},
            {"role": "assistant", "content": "[SILENT]"},
        ],
    }
    with caplog.at_level("DEBUG"):
        response, intentional, _messages = await runner._hmwa_shape_agent_response(
            terminal, source, [], entry, KEY, KEY, 1, entry.session_id, "discord", time.time(),
        )

    # The caller's own rule (``_handle_message_with_agent``): an allowed marker delivers nothing.
    assert ("" if intentional else response) == ""
    assert intentional is True
    assert not [
        record for record in caplog.records
        if record.levelname == "WARNING" and "silence marker" in record.message
    ]
    # The flag that verdict depends on reached the follow-up turn's own admission.
    assert calls[-1]["reply_expected"] is False
