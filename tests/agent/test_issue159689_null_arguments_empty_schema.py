"""#159689: a tool that declares no input must accept ``null`` arguments.

An OpenAI-compatible gateway sends ``None`` (and some send the JSON text ``"null"``) for a
tool the model calls without arguments. The executor rejected every non-object payload
before ``pre_tool_call``, so such a tool never ran and no plugin could repair it.

The narrow rule under test: a *complete* ``null`` means "no arguments" and parses as ``{}``
only for a tool whose resolved schema declares no input (empty/absent ``properties`` and
no ``required``). Everything #105189 protects keeps erroring: truncated JSON, non-object
payloads, unknown tool names, and tools that declare required input.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import tool_executor

# A tool whose schema admits no arguments at all.
NO_ARGUMENT_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}
# Same declaration minus the extraProperties marker: the shape `_make_tool_defs` in
# tests/agent/test_run_agent.py builds for every mocked tool.
EMPTY_PROPERTIES_SCHEMA = {"type": "object", "properties": {}}

WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"q": {"type": "string"}},
    "required": ["q"],
}
REQUIRED_ONLY_SCHEMA = {"type": "object", "required": ["q"]}

# The two spellings of a complete null payload.
NULL_PAYLOADS = [
    pytest.param(None, id="python-none"),
    pytest.param("null", id="json-null-text"),
]


def _definition(name, parameters):
    """A full OpenAI-shaped tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": parameters,
        },
    }


def _mcp_definition(name, input_schema):
    """An MCP-shaped definition: the schema sits at the top level as ``inputSchema``."""
    return {"name": name, "description": f"{name} tool", "inputSchema": input_schema}


def _tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _parse(definitions, name, arguments):
    """Drive the production ``_parse_tool_call`` against a fixed tool catalog."""
    agent = SimpleNamespace(enabled_toolsets=None, disabled_toolsets=None)
    with patch("model_tools.get_tool_definitions", return_value=definitions):
        return tool_executor._parse_tool_call(agent, _tool_call(name, arguments))


def _assert_rejected(parsed):
    """The historical structured rejection, byte-for-byte."""
    assert parsed.args == {}
    assert parsed.parse_error is not None
    payload = json.loads(parsed.parse_error)
    assert payload == {
        "error": "Invalid tool arguments",
        "message": "Tool arguments must be a valid JSON object; tool was not executed.",
    }


@pytest.mark.parametrize("payload", NULL_PAYLOADS)
@pytest.mark.parametrize("schema", [NO_ARGUMENT_SCHEMA, EMPTY_PROPERTIES_SCHEMA])
def test_null_arguments_parse_as_empty_object_for_no_argument_tool(schema, payload):
    parsed = _parse([_definition("noop", schema)], "noop", payload)

    assert parsed.args == {}
    assert parsed.parse_error is None


@pytest.mark.parametrize("payload", NULL_PAYLOADS)
def test_null_arguments_parse_as_empty_object_for_mcp_shaped_definition(payload):
    """``inputSchema`` is the other shape a definition can surface in."""
    parsed = _parse([_mcp_definition("noop", NO_ARGUMENT_SCHEMA)], "noop", payload)

    assert parsed.args == {}
    assert parsed.parse_error is None


@pytest.mark.parametrize("payload", NULL_PAYLOADS)
@pytest.mark.parametrize("schema", [WEB_SEARCH_SCHEMA, REQUIRED_ONLY_SCHEMA])
def test_null_arguments_still_rejected_for_tool_declaring_required_input(schema, payload):
    parsed = _parse([_definition("web_search", schema)], "web_search", payload)

    _assert_rejected(parsed)


@pytest.mark.parametrize("payload", NULL_PAYLOADS)
def test_null_arguments_still_rejected_for_unknown_tool(payload):
    """An unresolvable name must never be guessed into a no-argument tool."""
    parsed = _parse([_definition("noop", NO_ARGUMENT_SCHEMA)], "not_a_registered_tool", payload)

    _assert_rejected(parsed)


@pytest.mark.parametrize("payload", ['{"q":', "", "nul", "  ", '{"q": 1'])
def test_truncated_or_invalid_json_still_rejected_for_no_argument_tool(payload):
    """A cut string must never become ``{}`` (#105189): only a complete null is "no arguments"."""
    parsed = _parse([_definition("noop", NO_ARGUMENT_SCHEMA)], "noop", payload)

    _assert_rejected(parsed)


@pytest.mark.parametrize("payload", ["[]", "[1, 2]", '"text"', "42", "true"])
def test_non_object_payloads_still_rejected_for_no_argument_tool(payload):
    parsed = _parse([_definition("noop", NO_ARGUMENT_SCHEMA)], "noop", payload)

    _assert_rejected(parsed)


def _agent_with(*definitions):
    """Minimal AIAgent whose enabled catalog is ``definitions`` (patched for the test body)."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=list(definitions)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("hermes_cli.plugins.discover_plugins", lambda: None),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        yield agent


@pytest.fixture()
def no_argument_agent():
    yield from _agent_with(
        _definition("noop", NO_ARGUMENT_SCHEMA),
        _definition("web_search", WEB_SEARCH_SCHEMA),
    )


@pytest.mark.parametrize("payload", NULL_PAYLOADS)
def test_no_argument_tool_is_dispatched_when_model_sends_null(no_argument_agent, payload):
    """End to end through the concurrent executor: the tool actually runs with ``{}``."""
    noop_call = _tool_call("noop", payload, call_id="c1")
    search_call = _tool_call("web_search", '{"q":"ok"}', call_id="c2")
    message = SimpleNamespace(content="", tool_calls=[noop_call, search_call])
    messages = []
    seen = []

    def fake_handle(name, args, task_id, **kwargs):
        seen.append((kwargs["tool_call_id"], name, args))
        return json.dumps({"result": "ran"})

    with patch("model_tools.handle_function_call", side_effect=fake_handle):
        no_argument_agent._execute_tool_calls_concurrent(message, messages, "task-1")

    assert seen == [("c1", "noop", {}), ("c2", "web_search", {"q": "ok"})]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]
    assert all("tool was not executed" not in m["content"] for m in messages)


def test_required_input_tool_still_refused_end_to_end(no_argument_agent):
    """The #105189 contract, at the executor: a required-input tool is never run on null."""
    bad_call = _tool_call("web_search", None, call_id="c1")
    good_call = _tool_call("noop", None, call_id="c2")
    message = SimpleNamespace(content="", tool_calls=[bad_call, good_call])
    messages = []
    seen = []

    def fake_handle(name, args, task_id, **kwargs):
        seen.append((kwargs["tool_call_id"], name, args))
        return json.dumps({"result": "ran"})

    with patch("model_tools.handle_function_call", side_effect=fake_handle):
        no_argument_agent._execute_tool_calls_concurrent(message, messages, "task-1")

    assert seen == [("c2", "noop", {})]
    assert "tool was not executed" in messages[0]["content"].lower()
