#!/usr/bin/env python3
"""Boundary fence stripping for ``execute_code`` (#157155).

A markdown/HTML fence that wraps the ``code`` argument is a formatting artifact, not Python: the
runner compiles the argument verbatim, so one stray ```` ``` ```` or ``</code>`` line used to turn
an otherwise valid cell into a SyntaxError. These tests pin the strip at both edges (once), its
absence for clean input, and the fact that an interior fence is still the caller's problem.
"""

import json
import os
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"

from tools.code_execution_tool import (
    FENCE_STRIP_NOTICE,
    SANDBOX_ALLOWED_TOOLS,
    _strip_code_fence,
    execute_code,
)


@pytest.fixture(autouse=True)
def _local_terminal(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    from tools.code_kernel import shutdown_all_kernels

    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code, task_id="fence-test"):
    with patch("model_tools.handle_function_call", return_value='{"status": "ok"}'):
        return json.loads(execute_code(code, task_id=task_id,
                                       enabled_tools=list(SANDBOX_ALLOWED_TOOLS)))


def test_issue_repro_closing_code_tag_runs():
    """The issue's two lines: `print("alive")` plus a trailing `</code>` line.

    Before the fix the cell died with `SyntaxError: invalid syntax` on the tag line and printed
    nothing; now the print runs and the result carries the fence notice.
    """
    result = _run('print("alive")\n</code>')

    assert result["status"] == "success", result.get("error", "")
    assert "alive" in result["output"]
    assert result["fence_stripped"] is True
    assert FENCE_STRIP_NOTICE in result["warning"]
    assert "SyntaxError" not in result["output"]


def test_markdown_fence_with_info_string_stripped_on_both_edges():
    """```` ```python ... ``` ```` — one line removed at each edge, one notice total."""
    result = _run('```python\nprint("fenced")\n```')

    assert result["status"] == "success", result.get("error", "")
    assert "fenced" in result["output"]
    assert result["fence_stripped"] is True
    assert result["warning"].count("fence line stripped before compile") == 1


def test_opening_code_tag_stripped():
    """A leading `<code>` line is a fence too (models emit it before the body, tag unclosed)."""
    result = _run('<code>\nprint("tagged")')

    assert result["status"] == "success", result.get("error", "")
    assert "tagged" in result["output"]
    assert result["fence_stripped"] is True


def test_clean_code_is_untouched():
    """No edge fence: the argument is byte-identical and the result carries no fence notice."""
    source = 'x = 1\nprint(x + 1)\n'
    stripped, changed = _strip_code_fence(source)

    assert changed is False
    assert stripped == source  # byte-identical, not merely equivalent

    result = _run(source)
    assert result["status"] == "success", result.get("error", "")
    assert result["output"].strip().endswith("2")
    assert "fence_stripped" not in result
    assert "fence line stripped before compile" not in json.dumps(result)


def test_middle_fence_is_not_stripped():
    """Only edges are normalised: a fence line inside the body still fails to compile."""
    result = _run('print("first")\n</code>\nprint("second")')

    assert result["status"] == "error"
    assert "SyntaxError" in result["output"]  # nothing executed: the cell never compiled
    assert "fence_stripped" not in result
    assert "fence line stripped before compile" not in json.dumps(result)


@pytest.mark.parametrize("source", [
    '```python\nprint(1)\n```',
    '```\nprint(1)\n```',
    '~~~python\nprint(1)\n~~~',
    '<pre>\nprint(1)\n</pre>',
    '```python\nprint(1)',
    'print(1)\n</code>',
])
def test_accepted_boundary_fences(source):
    stripped, changed = _strip_code_fence(source)

    assert changed is True
    assert stripped == "print(1)"


@pytest.mark.parametrize("source", [
    'print(1)',                      # single line, no fence
    'x = "```"\nprint(x)',           # fence characters inside a string literal
    'print(1)\n```\nprint(2)',       # fence in the middle
    'print(1)\n# ``` trailing text', # not a fence-only line
])
def test_non_fence_sources_are_untouched(source):
    assert _strip_code_fence(source) == (source, False)


def test_remote_dispatch_receives_stripped_code():
    """The remote/kernel-hosted branch gets the same normalised source as the local kernel."""
    seen = {}

    def fake_remote(code, task_id, enabled_tools, **kwargs):
        seen["code"] = code
        return json.dumps({"status": "success", "output": "remote", "tool_calls_made": 0})

    with patch("tools.code_execution_tool._get_or_create_env", return_value=(object(), "ssh")), \
         patch("tools.terminal_tool._get_env_config", return_value={"env_type": "ssh"}), \
         patch("tools.code_execution_tool._execute_remote", side_effect=fake_remote):
        raw = execute_code('```python\nprint("remote")\n```', task_id="fence-remote",
                           enabled_tools=list(SANDBOX_ALLOWED_TOOLS))

    result = json.loads(raw)
    assert seen["code"] == 'print("remote")'
    assert result["fence_stripped"] is True
    assert FENCE_STRIP_NOTICE in result["warning"]
