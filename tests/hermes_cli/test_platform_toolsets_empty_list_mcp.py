"""#147025: ``platform_toolsets.<platform>: []`` must load no MCP server.

An explicitly saved EMPTY list is the user saying "no tools" — the zero-tool state #82010 already
established for native toolsets, and the one cron reads for a per-job ``[]``. The MCP half of the
resolver did not follow it: ``_merge_mcp_servers`` only recognised the ``no_mcp`` sentinel, so an
empty list fell through to "every enabled server" and an e-mail session configured with
``platform_toolsets.email: []`` loaded all 87 tools of its two configured servers while
``terminal`` — a native toolset — was correctly refused.

Only an ABSENT key means "everything"; an explicit ``no_mcp`` keeps meaning "no MCP server".
"""

import json
from types import SimpleNamespace

from hermes_cli.tools_config import _get_platform_tools


def _config(platform_toolsets=None, servers=("alpha", "beta")):
    config = {"mcp_servers": {name: {"url": f"https://example.com/{name}"} for name in servers}}
    if platform_toolsets is not None:
        config["platform_toolsets"] = platform_toolsets
    return config


def test_an_explicit_empty_list_loads_no_mcp_server():
    """RED on main: the two enabled servers came back as ``{'alpha', 'beta'}``."""
    enabled = _get_platform_tools(_config({"email": []}), "email")

    assert not {"alpha", "beta", "mcp-alpha", "mcp-beta"} & enabled, enabled
    assert enabled == set(), enabled


def test_an_explicit_empty_list_stays_zero_toolsets():
    """The native half already failed closed; the MCP merge must not reopen the selection."""
    enabled = _get_platform_tools(_config({"email": []}, servers=()), "email")

    assert enabled == set(), enabled


def test_an_absent_key_still_means_every_enabled_server():
    """Control: absent (``None``) is the only shape that means "everything" — unchanged."""
    enabled = _get_platform_tools(_config(), "email")

    assert {"alpha", "beta"} <= enabled, enabled
    assert enabled - {"alpha", "beta"}, "the platform default toolsets must still resolve"


def test_the_no_mcp_sentinel_still_drops_every_server():
    """Control: the documented token keeps working, with and without other toolsets."""
    enabled = _get_platform_tools(_config({"email": ["terminal", "no_mcp"]}), "email")

    assert not {"alpha", "beta"} & enabled, enabled
    assert "terminal" in enabled


def test_naming_one_server_is_still_the_allowlist():
    """Control: a named server still narrows the selection to itself."""
    enabled = _get_platform_tools(_config({"email": ["terminal", "alpha"]}), "email")

    assert "alpha" in enabled, enabled
    assert "beta" not in enabled, enabled


def _connections(*names):
    return json.dumps({
        "targets": [
            {"name": name, "kind": "mcp", "state": "connected", "tools": ["probe"]} for name in names
        ]
    })


def test_a_connected_server_is_not_scoped_into_an_empty_selection():
    """``_scope_in_connected_mcp_servers`` treated an empty selection as "every toolset" and added
    the just-connected server back (``['alpha']`` on main)."""
    from agent.inline_tool_executors import _scope_in_connected_mcp_servers

    agent = SimpleNamespace(enabled_toolsets=[])
    _scope_in_connected_mcp_servers(agent, _connections("alpha"))

    assert agent.enabled_toolsets == [], agent.enabled_toolsets


def test_a_connected_server_is_still_scoped_into_a_live_selection():
    """Control: a selection that already holds toolsets still gains the server mid-turn."""
    from agent.inline_tool_executors import _scope_in_connected_mcp_servers

    agent = SimpleNamespace(enabled_toolsets=["terminal"])
    _scope_in_connected_mcp_servers(agent, _connections("alpha"))

    assert agent.enabled_toolsets == ["terminal", "alpha"], agent.enabled_toolsets


def test_a_late_grant_is_registered_but_does_not_widen_an_empty_selection(monkeypatch):
    """The same rule for a grant adopted between turns: the server is registered, the zero
    selection stays zero (on main the agent got ``['alpha']``)."""
    from hermes_constants import hermes_home_key
    from tools.connectors import mcp
    from tools.connectors.gateway.config import operation_session_key

    class _Approved:
        def poll(self):
            return {"status": "approved"}

    key = (hermes_home_key(), operation_session_key("s1"))
    monkeypatch.setitem(mcp._LATE_ATTEMPTS, key, {"alpha": _Approved()})
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", lambda: {"alpha": {"url": "u"}})
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.register_mcp_servers", lambda servers: list(servers)
    )
    agent = SimpleNamespace(session_id="s1", enabled_toolsets=[])

    assert mcp.adopt_late_connections(agent) == ["alpha"]
    assert agent.enabled_toolsets == [], agent.enabled_toolsets
