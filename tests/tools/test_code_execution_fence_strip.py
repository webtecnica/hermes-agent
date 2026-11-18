#!/usr/bin/env python3
"""Edge-fence normalization for execute_code (#157155).

Models often wrap the ``code`` argument in a markdown/HTML fence. The fence lines used to reach
``compile`` and raise ``SyntaxError``; the host now strips at most one delimiter line from each
edge before dispatching to any backend. These tests cover the accepted delimiters, the
byte-identical pass-through when no fence is present, the mid-body non-strip, and the notice
added to the JSON result.
"""

import json
import os
import unittest
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"

from tools.code_execution_tool import (  # noqa: E402
    SANDBOX_ALLOWED_TOOLS,
    _FENCE_STRIPPED_NOTICE,
    _strip_edge_fence_lines,
    execute_code,
)


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    from tools.code_kernel import shutdown_all_kernels

    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code):
    with patch("model_tools.handle_function_call", side_effect=lambda *a, **k: json.dumps({"ok": True})):
        raw = execute_code(code=code, task_id="test-fence", enabled_tools=list(SANDBOX_ALLOWED_TOOLS))
    return raw, json.loads(raw)


class TestStripEdgeFenceLines(unittest.TestCase):
    """Pure-helper checks for the accepted edge forms and the non-strip rules."""

    def test_closing_html_tag_only(self):
        stripped, changed = _strip_edge_fence_lines('print("alive")\n</code>\n')
        self.assertTrue(changed)
        self.assertEqual(stripped, 'print("alive")\n')

    def test_backtick_fence_with_info_string_on_both_ends(self):
        source = 'print("fenced")\n'
        stripped, changed = _strip_edge_fence_lines("```python\n" + source + "```\n")
        self.assertTrue(changed)
        self.assertEqual(stripped, source)

    def test_opening_html_code_tag(self):
        stripped, changed = _strip_edge_fence_lines("<code>\n" + 'print("tagged")\n' + "</code>")
        self.assertTrue(changed)
        self.assertEqual(stripped, 'print("tagged")\n')

    def test_other_accepted_delimiters(self):
        body = "print(1)\n"
        for opener, closer in (("~~~", "~~~"), ("<pre>", "</pre>"), ("````", "````")):
            with self.subTest(opener=opener):
                stripped, changed = _strip_edge_fence_lines(f"{opener}\n{body}{closer}\n")
                self.assertTrue(changed)
                self.assertEqual(stripped, body)

    def test_at_most_one_fence_line_per_edge(self):
        body = "print(1)\n"
        stripped, changed = _strip_edge_fence_lines("```\n```\n" + body + "```\n```\n")
        self.assertTrue(changed)
        # One opener and one closer are trimmed; the leftovers are the caller's own code.
        self.assertEqual(stripped, "```\n" + body + "```\n")

    def test_unfenced_source_is_returned_byte_identically(self):
        code = 'x = 1\n\nprint(x)   \n'
        stripped, changed = _strip_edge_fence_lines(code)
        self.assertFalse(changed)
        self.assertIs(stripped, code)

    def test_fence_in_the_middle_is_not_stripped(self):
        code = 'x = 1\n```\nprint(x)\n'
        stripped, changed = _strip_edge_fence_lines(code)
        self.assertFalse(changed)
        self.assertIs(stripped, code)

    def test_empty_source(self):
        self.assertEqual(_strip_edge_fence_lines(""), ("", False))


class TestExecuteCodeFenceStrip(unittest.TestCase):
    """End-to-end: the fenced argument now compiles and the result says so."""

    def test_issue_reproduction_print_then_closing_tag(self):
        """#157155: `print("alive")` followed by a lone `</code>` line used to raise SyntaxError."""
        raw, result = _run('print("alive")\n</code>\n')
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertIn("alive", result["output"])
        self.assertNotIn("SyntaxError", raw)
        self.assertEqual(result["notice"], _FENCE_STRIPPED_NOTICE)

    def test_backtick_fence_with_info_string_both_ends(self):
        raw, result = _run("```python\nprint('fenced')\n```\n")
        self.assertEqual(result["status"], "success")
        self.assertIn("fenced", result["output"])
        self.assertEqual(result["notice"], _FENCE_STRIPPED_NOTICE)
        self.assertEqual(raw.count(_FENCE_STRIPPED_NOTICE), 1)

    def test_opening_code_tag(self):
        result = _run("<code>\nprint('tagged')\n")[1]
        self.assertEqual(result["status"], "success")
        self.assertIn("tagged", result["output"])
        self.assertEqual(result["notice"], _FENCE_STRIPPED_NOTICE)

    def test_unfenced_code_reaches_backend_unchanged_and_gets_no_notice(self):
        code = 'print("plain")\n'
        captured = {}

        def fake_kernel(source, **kwargs):
            captured["code"] = source
            return json.dumps({"status": "success", "output": "", "tool_calls_made": 0})

        with patch("tools.code_kernel.execute_in_session_kernel", side_effect=fake_kernel):
            raw = execute_code(code=code, task_id="test-fence", enabled_tools=list(SANDBOX_ALLOWED_TOOLS))
        result = json.loads(raw)
        self.assertEqual(captured["code"], code)
        self.assertNotIn("notice", result)
        self.assertNotIn(_FENCE_STRIPPED_NOTICE, raw)

    def test_fence_in_the_middle_still_fails(self):
        """Only edges are normalized: an interior fence line keeps raising SyntaxError."""
        raw, result = _run('x = 1\n```\nprint(x)\n')
        self.assertEqual(result["status"], "error")
        self.assertIn("SyntaxError", result.get("error", ""))
        self.assertNotIn("notice", result)


class TestRemoteDispatchGetsStrippedSource(unittest.TestCase):
    """The remote backend (kernel and per-call ship) inherits the host-side normalization."""

    def test_remote_backend_receives_stripped_source_and_notice(self):
        captured = {}

        def fake_remote(source, *args, **kwargs):
            captured["code"] = source
            return json.dumps({"status": "success", "output": "alive\n"})

        with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}), \
             patch("tools.terminal_tool._docker_has_host_access", return_value=False), \
             patch("tools.approval.check_execute_code_guard",
                   return_value={"approved": True, "user_approved": False}), \
             patch("tools.code_execution_tool._execute_remote", side_effect=fake_remote):
            raw = execute_code('print("alive")\n</code>', task_id="test-fence-remote", enabled_tools=[])
        result = json.loads(raw)
        self.assertEqual(captured["code"], 'print("alive")\n')
        self.assertEqual(result["notice"], _FENCE_STRIPPED_NOTICE)


if __name__ == "__main__":
    unittest.main()
