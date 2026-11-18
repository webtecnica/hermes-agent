#!/usr/bin/env python3
"""Edge fence stripping in ``execute_code`` (#157155).

A code argument that arrived wrapped in a markdown/HTML block used to be shipped as-is:
the closing fence line reached ``exec(compile(...))`` inside the kernel runner and raised
``SyntaxError: invalid syntax``. ``execute_code`` now drops at most one fence line at each
end of the argument, on the host and before any backend dispatch, and reports
``fence line stripped before compile`` in the result. A fence in the middle of the code is
a genuine syntax error and is deliberately left alone.

Run with:  scripts/run_tests.sh tests/tools/test_code_execution_fence_strip.py
"""

import json
import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"

from tools.code_execution_tool import (  # noqa: E402
    FENCE_STRIPPED_NOTICE,
    _annotate_fence_strip,
    _is_edge_fence_line,
    _strip_edge_fence,
    execute_code,
)
from tools.code_kernel import shutdown_all_kernels  # noqa: E402

NOTICE_FRAGMENT = "fence line stripped before compile"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_kernel.py — guarantee the local backend regardless of ordering."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


@contextmanager
def _kernel_config(**overrides):
    config = {"mode": "strict", "timeout": 30}
    config.update(overrides)
    with patch("tools.code_execution_tool._load_config", return_value=config):
        yield


def _run(code, **kwargs):
    return json.loads(execute_code(code, task_id="fence-test", **kwargs))


class TestEdgeFenceDetection:
    """The accepted border lines, and everything that must not be touched."""

    @pytest.mark.parametrize("line", [
        "```", "````", "```python", "``` python", "~~~", "~~~bash",
        "<code>", "</code>", "<pre>", "</pre>", "  ```  ", "  </code> ",
    ])
    def test_recognized_fence_lines(self, line):
        assert _is_edge_fence_line(line) is True

    @pytest.mark.parametrize("line", [
        "", "   ", "print('```')", "x = '<code>'", "``", "code = 1",
        "```not a fence", "</code> # trailing comment", "<codigo>",
    ])
    def test_non_fence_lines(self, line):
        assert _is_edge_fence_line(line) is False

    def test_closing_tag_only_is_stripped_at_the_end(self):
        code = 'print("alive")\n</code>'
        stripped, changed = _strip_edge_fence(code)
        assert changed is True
        assert stripped == 'print("alive")'

    def test_both_ends_with_info_string(self):
        code = '```python\nprint("alive")\n```'
        stripped, changed = _strip_edge_fence(code)
        assert changed is True
        assert stripped == 'print("alive")'

    def test_opening_code_tag_is_stripped(self):
        code = '<code>\nprint("alive")'
        stripped, changed = _strip_edge_fence(code)
        assert changed is True
        assert stripped == 'print("alive")'

    def test_code_without_fence_is_byte_identical(self):
        code = 'x = 1\nprint(x)\n'  # trailing newline and all
        stripped, changed = _strip_edge_fence(code)
        assert changed is False
        assert stripped == code

    def test_mid_code_fence_is_untouched(self):
        code = 'print("a")\n</code>\nprint("b")'
        stripped, changed = _strip_edge_fence(code)
        assert changed is False
        assert stripped == code

    def test_only_one_line_per_end(self):
        code = '```\n```\nprint("alive")\n```\n```'
        stripped, changed = _strip_edge_fence(code)
        assert changed is True
        assert stripped == '```\nprint("alive")\n```'

    def test_lone_fence_line_is_untouched(self):
        # Never normalize the argument down to nothing: a bare fence is still a syntax error.
        stripped, changed = _strip_edge_fence("```")
        assert changed is False
        assert stripped == "```"

    def test_annotate_noop_without_strip(self):
        payload = json.dumps({"status": "success", "output": "ok\n"})
        assert _annotate_fence_strip(payload, False) == payload

    def test_annotate_appends_to_existing_warning(self):
        payload = json.dumps({"status": "success", "output": "", "warning": "earlier note"})
        decoded = json.loads(_annotate_fence_strip(payload, True))
        assert decoded["warning"].startswith("earlier note")
        assert NOTICE_FRAGMENT in decoded["warning"]

    def test_annotate_ignores_non_json(self):
        assert _annotate_fence_strip("not json", True) == "not json"


class TestFenceStripEndToEnd:
    """The issue's reproduction, run through the real local session kernel."""

    def test_issue_repro_trailing_code_tag(self):
        with _kernel_config():
            result = _run('print("alive")\n</code>')
        assert result["status"] == "success", result
        assert "alive" in result["output"]
        assert NOTICE_FRAGMENT in result.get("warning", "")

    def test_both_ends_with_info_string(self):
        with _kernel_config():
            result = _run('```python\nprint("alive")\n```')
        assert result["status"] == "success", result
        assert "alive" in result["output"]
        assert NOTICE_FRAGMENT in result.get("warning", "")

    def test_opening_code_tag(self):
        with _kernel_config():
            result = _run('<code>\nprint("alive")')
        assert result["status"] == "success", result
        assert "alive" in result["output"]
        assert NOTICE_FRAGMENT in result.get("warning", "")

    def test_unfenced_code_is_unchanged_and_silent(self):
        with _kernel_config():
            result = _run('print("alive")')
        assert result["status"] == "success", result
        assert result["output"].strip().endswith("alive")
        assert "warning" not in result

    def test_mid_code_fence_still_fails(self):
        with _kernel_config():
            result = _run('print("a")\n</code>\nprint("b")')
        assert result["status"] == "error", result
        assert "SyntaxError" in result.get("error", "")
        assert "alive" not in result.get("output", "")

    def test_strip_happens_on_the_host_for_every_backend(self):
        """The backend receives the stripped source and the reply carries the notice — the
        remote backends inherit both because the strip/annotate sit in ``execute_code``."""
        seen = {}

        def fake_remote(code, *args, **kwargs):
            seen["code"] = code
            return json.dumps({"status": "success", "output": "alive\n"})

        with patch("tools.terminal_tool._get_env_config", return_value={"env_type": "docker"}), \
                patch("tools.terminal_tool._docker_has_host_access", return_value=False), \
                patch("tools.process_registry._is_supervised_gateway_process", return_value=False), \
                patch("tools.approval.check_execute_code_guard", return_value={"approved": True}), \
                patch("tools.code_execution_tool._execute_remote", side_effect=fake_remote):
            result = json.loads(execute_code('print("alive")\n</code>', task_id="fence-remote"))
        assert seen["code"] == 'print("alive")'
        assert result["status"] == "success", result
        assert NOTICE_FRAGMENT in result.get("warning", "")
