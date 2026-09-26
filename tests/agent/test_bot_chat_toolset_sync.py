"""Regression test for #124211: Toolset changes reach canonical Bot Chat."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hermes_state import SessionDB


def _make_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_capability_fingerprint_changes_with_platform_toolsets(tmp_path):
    from tools.bot_mode_probe import capability_fingerprint

    cfg_before = {
        "platform_toolsets": {"cli": ["terminal", "web_search"]},
    }
    cfg_after = {
        "platform_toolsets": {"cli": ["terminal", "web_search", "computer_use"]},
    }

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg_before):
        fp1 = capability_fingerprint(tmp_path)

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg_after):
        fp2 = capability_fingerprint(tmp_path)

    assert fp1 != "unavailable"
    assert fp2 != "unavailable"
    assert fp1 != fp2, "Capability fingerprint must change when platform_toolsets change"


def test_capability_fingerprint_changes_with_disabled_toolsets(tmp_path):
    from tools.bot_mode_probe import capability_fingerprint

    cfg1 = {"agent": {"disabled_toolsets": []}}
    cfg2 = {"agent": {"disabled_toolsets": ["computer_use"]}}

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg1):
        fp1 = capability_fingerprint(tmp_path)

    with patch("hermes_cli.config.load_config_readonly", return_value=cfg2):
        fp2 = capability_fingerprint(tmp_path)

    assert fp1 != fp2, "Capability fingerprint must change when agent.disabled_toolsets change"


def test_bot_chat_capability_stale_refreshes_and_persists_tools(tmp_path):
    from agent.conversation_loop import _restore_or_build_system_prompt
    from tools.mcp_tool_agent import tool_pin_version

    session_id = "canonical-bot-session"
    db_path = tmp_path / "state.db"

    initial_tools = [_make_tool("read_file"), _make_tool("terminal")]
    updated_tools = [_make_tool("read_file"), _make_tool("terminal"), _make_tool("computer_use")]

    with SessionDB(db_path=db_path) as db:
        db.create_session(session_id, source="desktop")
        db.set_session_title(session_id, "Bot Chat")
        old_prompt = (
            "You are a helpful assistant.\n\n"
            "Capability epoch: 111111111111\n"
            "Model: test-model\n"
            "Provider: test-provider"
        )
        db.update_system_prompt(session_id, old_prompt)
        db.update_session_tool_names(
            session_id,
            {"version": tool_pin_version(), "tools": initial_tools},
        )

        agent = SimpleNamespace(
            session_id=session_id,
            _session_db=db,
            _bot_mode_protocol=True,
            _session_title_hint="Bot Chat",
            model="test-model",
            provider="test-provider",
            platform="desktop",
            quiet_mode=True,
            tools=list(initial_tools),
            valid_tool_names={t["function"]["name"] for t in initial_tools},
            _cached_system_prompt=None,
            _build_system_prompt=MagicMock(return_value="NEW_PROMPT\nCapability epoch: 222222222222\nModel: test-model\nProvider: test-provider"),
            _persist_disabled=False,
        )

        def fake_refresh(agent_arg, **kwargs):
            agent_arg.tools = list(updated_tools)
            agent_arg.valid_tool_names = {t["function"]["name"] for t in updated_tools}
            return {"computer_use"}

        with (
            patch("tools.bot_mode_probe.stored_prompt_capability_stale", return_value=True),
            patch("tools.mcp_tool_agent.refresh_agent_mcp_tools", side_effect=fake_refresh),
            patch("tools.mcp_tool_agent.reprobe_tool_availability"),
            patch("hermes_cli.config.load_config_readonly", return_value={"platform_toolsets": {"desktop": ["computer_use"]}}),
        ):
            # Turn 1: capability epoch changed on disk
            _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        # Verify agent.tools was refreshed
        assert "computer_use" in agent.valid_tool_names
        assert any(t["function"]["name"] == "computer_use" for t in agent.tools)

        # Verify session DB was updated with new tools pin and new prompt
        row = db.get_session(session_id)
        assert row["system_prompt"].startswith("NEW_PROMPT")
        saved_tools_data = json.loads(row["tool_names"])
        saved_names = [t["function"]["name"] for t in saved_tools_data["tools"]]
        assert "computer_use" in saved_names, "Persisted tool pin must include computer_use"


def test_compaction_refresh_reloads_config_platform_toolsets():
    from agent.conversation_compression import _refresh_agent_tool_definitions

    agent = SimpleNamespace(
        platform="cli",
        tools=[_make_tool("terminal")],
        valid_tool_names={"terminal"},
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value={"platform_toolsets": {"cli": ["terminal", "computer_use"]}}),
        patch("hermes_cli.tools_config._get_platform_tools", return_value={"terminal", "computer_use"}),
        patch("tools.mcp_tool_agent.reprobe_tool_availability"),
        patch("tools.mcp_tool_agent.refresh_agent_mcp_tools", return_value={"computer_use"}) as mock_refresh,
    ):
        result = _refresh_agent_tool_definitions(agent)
        assert result is True
        mock_refresh.assert_called_once()
        _, kwargs = mock_refresh.call_args
        assert kwargs.get("enabled_override") == ["computer_use", "terminal"]
