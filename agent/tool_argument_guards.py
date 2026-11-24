"""Accept a complete ``null`` argument payload for tools that declare no input (#159689).

An OpenAI-compatible gateway hands over ``None`` — and some clients send the JSON text
``"null"`` — for a tool the model calls without arguments. The executor rejected every
non-object payload before ``pre_tool_call``, so a parameterless tool never ran and no plugin
could repair it.

The rule is deliberately narrow, because #105189 depends on a cut string never becoming
``{}``: only a *complete* ``null`` counts as "no arguments", and only for a tool whose
resolved definition declares no input (no ``required`` and no ``properties``). Truncated or
invalid JSON, non-object payloads, unknown tool names and tools that declare required input
keep the unchanged structured error.
"""

from typing import Any, Optional


def is_complete_null(raw_arguments: Any) -> bool:
    """True for the two spellings of a complete ``null``: ``None`` and the text ``"null"``.

    A truncated string (``""``, ``"nul"``, ``'{"q":'``) is never a complete null, so it keeps
    the rejection: the model cut its JSON short, and guessing ``{}`` would drop arguments a
    retry could recover.
    """
    if raw_arguments is None:
        return True
    return isinstance(raw_arguments, str) and raw_arguments.strip() == "null"


def declares_no_input(agent, name: str) -> bool:
    """Whether the canonical tool name ``name`` resolves to a definition declaring no input.

    Reads the same catalog the executor already scopes with (``function.parameters`` for an
    OpenAI-shaped definition, ``inputSchema`` for an MCP-shaped one) and answers True only for
    an object schema with no ``required`` and no ``properties``. Fail-closed: an unknown name,
    an unreadable schema, or a failed lookup answers False, so the caller keeps the historical
    rejection instead of guessing which tool the model meant.
    """
    try:
        import model_tools

        definitions = model_tools.get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        ) or []
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, KeyError):
        # No catalog, no exemption: the tool keeps erroring exactly as before this rule.
        return False
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        function = definition.get("function")
        if not isinstance(function, dict):
            function = definition
        if function.get("name") != name:
            continue
        schema = function.get("parameters")
        if not isinstance(schema, dict):
            schema = function.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            return False
        return not schema.get("required") and not schema.get("properties")
    return False


def parse_arguments_for_call(
    agent, name: str, raw_arguments: Any
) -> tuple[dict, Optional[str]]:
    """Parse ``name``'s model-emitted arguments, letting a complete ``null`` mean ``{}``.

    The literal test runs first so the (memoized) definition lookup only happens for the
    payload that can change the verdict. Everything else goes through the executor's parser
    untouched, which keeps its "never repairs or coerces" contract for truncated JSON,
    non-object values, and null for a tool that declares required input.
    """
    if is_complete_null(raw_arguments) and declares_no_input(agent, name):
        return {}, None
    from agent.tool_executor import _parse_tool_arguments

    return _parse_tool_arguments(raw_arguments)
