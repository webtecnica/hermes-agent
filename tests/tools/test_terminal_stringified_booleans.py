"""Stringified boolean args on the terminal tool are coerced, invalid strings are refused.

Issue #157759 (Bug 1): models emit `True`/`False` (a Python repr) or `"false"` where the
schema declares a JSON boolean. The handler used to read those with `args.get(...)`, so a
non-empty string was truthy: `background="false"` started a background process, and
`notify="True"` was rejected outright because the notify branch required a real bool.

Only the two JSON spellings are accepted (`true`/`false`, case- and whitespace-insensitive);
`1`/`yes`/`on` are refused with a message that names the accepted spellings, so no invalid
string turns into a silent boolean. Real booleans, the notify pattern list and `heartbeat`
behaviour are unchanged.

The handler is exercised directly (`_handle_terminal`), the same entry point the registry
calls, because `registry.dispatch` hands the args to the handler untouched: the
schema-guided layer in `tools.arg_coercion` only runs on some client paths and cannot
repair `notify` (an `anyOf` with no top-level `type`).
"""

import json
from unittest.mock import patch

import pytest


def _env_config(cwd):
    return {
        "env_type": "local", "timeout": 180, "cwd": str(cwd), "host_cwd": None, "modal_mode": "auto",
        "docker_image": "", "modal_image": "", "daytona_image": "",
    }


def _capture_terminal_tool(monkeypatch):
    """Patch the spawn function the handler calls and return the kwargs it received."""
    from tools import terminal_tool as tt

    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "ok", "exit_code": 0})

    monkeypatch.setattr(tt, "terminal_tool", fake_terminal_tool)
    return captured


def _run_real(args, tmp_path, monkeypatch):
    """Run the real handler against the local backend (approval guards patched out)."""
    from tools import terminal_tool as tt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    with patch.object(tt, "_get_env_config", return_value=_env_config(tmp_path)), \
         patch.object(tt, "_start_cleanup_thread"), \
         patch.object(tt, "_check_all_guards", return_value={"approved": True}):
        return json.loads(tt._handle_terminal(args))


# ---------------------------------------------------------------------------
# The dangerous direction: a string "false" must NOT run in the background.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["false", "False", "  FALSE  "])
def test_string_false_background_runs_in_the_foreground(tmp_path, monkeypatch, raw):
    marker = tmp_path / "ran"
    result = _run_real({"command": f"echo x >> {marker}", "background": raw}, tmp_path, monkeypatch)

    assert result.get("error") is None, result
    assert "session_id" not in result
    assert result["exit_code"] == 0 and result["output"] == ""
    assert marker.read_text(encoding="utf-8-sig").count("x") == 1


@pytest.mark.parametrize("raw", ["true", "True", " true "])
def test_string_true_background_and_notify_spawn_one_notified_process(tmp_path, monkeypatch, raw):
    marker = tmp_path / "ran"
    result = _run_real(
        {"command": f"echo bg >> {marker}", "background": raw, "notify": "True"},
        tmp_path, monkeypatch,
    )

    assert result.get("error") is None, result
    assert result["session_id"].startswith("proc_")
    assert result["notify_on_complete"] is True


# ---------------------------------------------------------------------------
# notify: string booleans take the boolean branch; the list branch is untouched.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("true", True), ("True", True), (" TRUE ", True),
    ("false", False), ("False", False), ("false\n", False),
])
def test_notify_string_booleans_map_to_notify_on_complete(monkeypatch, raw, expected):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "background": "true", "notify": raw}))

    assert not result.get("error"), result
    assert captured["notify_on_complete"] is expected
    assert captured["watch_patterns"] is None


def test_notify_pattern_list_is_unchanged(monkeypatch):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "background": True, "notify": ["BUILD_DONE"]}))

    assert not result.get("error"), result
    assert captured["watch_patterns"] == ["BUILD_DONE"]
    assert captured["notify_on_complete"] is False


def test_string_false_notify_on_a_foreground_call_is_not_notification_intent(monkeypatch):
    """Before coercion `"False"` was truthy, so a foreground call was refused as if it
    asked for a notification it never wanted."""
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "notify": "False"}))

    assert not result.get("error"), result
    assert captured["background"] is False
    assert captured["notify_on_complete"] is False


# ---------------------------------------------------------------------------
# background / pty / persist_on_release: JSON spellings only.
# ---------------------------------------------------------------------------
def test_pty_and_persist_on_release_string_booleans_are_coerced(monkeypatch):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({
        "command": "pwd", "background": "true", "pty": "true", "persist_on_release": "true",
    }))
    assert not result.get("error"), result
    assert captured["background"] is True
    assert captured["pty"] is True
    assert captured["persist_on_release"] is True

    captured.clear()
    result = json.loads(_handle_terminal({
        "command": "pwd", "background": "true", "pty": "False", "persist_on_release": "false",
    }))
    assert not result.get("error"), result
    assert captured["pty"] is False
    assert captured["persist_on_release"] is False


@pytest.mark.parametrize("argument, raw", [
    ("background", "1"),
    ("background", "yes"),
    ("background", "on"),
    ("background", "talvez"),
    ("pty", "1"),
    ("pty", "maybe"),
    ("persist_on_release", "yes"),
    ("persist_on_release", "0"),
])
def test_non_json_strings_are_refused_not_coerced(monkeypatch, argument, raw):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    args = {"command": "pwd", "background": "true", argument: raw}
    if argument == "background":
        args = {"command": "pwd", "background": raw}

    result = json.loads(_handle_terminal(args))

    assert result.get("error"), result
    assert argument in result["error"]
    assert "true" in result["error"] and "false" in result["error"]
    assert captured == {}


def test_notify_rejection_names_the_accepted_spellings(monkeypatch):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "background": True, "notify": "talvez"}))

    assert result.get("error"), result
    assert "notify must be" in result["error"]
    assert "true/false" in result["error"]
    assert "list of strings" in result["error"]
    assert captured == {}


# ---------------------------------------------------------------------------
# Regressions: real booleans, legacy notify_on_complete, heartbeat.
# ---------------------------------------------------------------------------
def test_real_booleans_still_reach_the_tool_unchanged(monkeypatch):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({
        "command": "pwd", "background": True, "pty": True,
        "persist_on_release": True, "notify": True, "heartbeat": 120,
    }))

    assert not result.get("error"), result
    assert captured["background"] is True
    assert captured["pty"] is True
    assert captured["persist_on_release"] is True
    assert captured["notify_on_complete"] is True
    assert captured["watch_patterns"] is None
    assert captured["heartbeat"] == 120


def test_legacy_notify_on_complete_string_is_coerced_too(monkeypatch):
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "background": "true", "notify_on_complete": "False"}))

    assert not result.get("error"), result
    assert captured["notify_on_complete"] is False


def test_numeric_string_heartbeat_is_still_refused(monkeypatch):
    """Measured before the fix: heartbeat never accepted a numeric string, so this change
    leaves it alone (only boolean arguments are coerced)."""
    from tools.terminal_tool import _handle_terminal

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(_handle_terminal({"command": "pwd", "background": True, "heartbeat": "60"}))

    assert result.get("error"), result
    assert "heartbeat must be a whole number" in result["error"]
    assert captured == {}


# ---------------------------------------------------------------------------
# Deadline floor and advertised hint.
# ---------------------------------------------------------------------------
def test_deadline_floor_reads_a_string_background_as_the_handler_does():
    from tools.terminal_tool import _terminal_deadline_floor

    assert _terminal_deadline_floor({"background": "false", "timeout": 30}) == 30.0
    assert _terminal_deadline_floor({"background": "true", "timeout": 30}) is None
    assert _terminal_deadline_floor({"background": False, "timeout": 30}) == 30.0


def test_timeout_hint_names_the_advertised_notify_parameter():
    from tools.terminal_tool import TERMINAL_SCHEMA

    hint = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
    assert "notify=true" in hint
    assert "notify_on_complete" not in hint


def test_promoted_timeout_note_names_the_advertised_notify_parameter():
    """The over-cap result note is model-facing too; it used to advertise the internal kwarg."""
    from tools.terminal_tool import _PROMOTED_NOTE

    assert "notify=true" in _PROMOTED_NOTE
    assert "notify_on_complete" not in _PROMOTED_NOTE


# ---------------------------------------------------------------------------
# The registry route, where `registry.dispatch` hands the args to the handler as-is.
# ---------------------------------------------------------------------------
def test_registry_dispatch_coerces_string_booleans(monkeypatch):
    from tools.registry import registry

    captured = _capture_terminal_tool(monkeypatch)
    result = json.loads(registry.dispatch("terminal", {"command": "pwd", "background": "False", "notify": "False"}))

    assert not result.get("error"), result
    assert captured["background"] is False
    assert captured["notify_on_complete"] is False
