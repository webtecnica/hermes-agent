"""Result identity for the loop guardrails (#146846).

The guardrails decide "this call returned the same result again" from a hash of the result. The
harness annotates its OWN repeats: from the 3rd identical search ``tools/file_tools.py`` attaches
``_warning`` with the run count, from the 4th it answers ``BLOCKED`` plus ``already_searched=N``,
and a 0-match search lists its probe paths in ripgrep's traversal order. Hashing those bytes made
every replay look new, so the guardrail never counted the one thing it exists to stop.

These tests pin the contract: the harness's repeat bookkeeping is not result identity, and nothing
else is folded (a genuinely different result still resets the count).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools import file_tools
from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    _result_hash,
    classify_tool_failure,
)
from tools.file_operations_common import SearchMatch, SearchResult
from tools.file_operations_search import SearchMixin
from tools.file_tools_read_tracking import _read_tracker

_ARGS = {"pattern": "needle", "path": "/repo/pkg"}


def _search_result() -> SearchResult:
    return SearchResult(
        total_count=1,
        matches=[SearchMatch(path="/repo/pkg/a.py", line_number=3, content="needle = 1")],
    )


def _zero_match_payload(paths: list[str], *, total: int = 3, probe: str = "-i") -> str:
    """A 0-match search payload exactly as the tool composes it (warning holds the probe paths)."""
    templates = dict(SearchMixin._ZERO_MATCH_PROBES)
    warning = templates[probe].format(total=total, n=len(paths), paths=", ".join(paths))
    return json.dumps({"total_count": 0, "warning": warning}, ensure_ascii=False)


@pytest.fixture()
def controller_factory():
    def build(**kwargs) -> ToolCallGuardrailController:
        return ToolCallGuardrailController(ToolCallGuardrailConfig(**kwargs))

    return build


def test_repeated_search_with_the_tools_own_warning_and_blocked_counts_trips_the_guardrail(
    tmp_path, controller_factory
):
    """The tool's repeat counters must not make each replay look like fresh output."""
    tid = "guardrail-identity-146846"
    _read_tracker.pop(tid, None)
    ops = MagicMock()
    ops.search.return_value = _search_result()
    controller = controller_factory(
        hard_stop_enabled=True, no_progress_warn_after=2, no_progress_block_after=4)

    payloads, counts, actions = [], [], []
    with patch("tools.file_tools._get_file_ops", return_value=ops):
        for _ in range(5):
            result = file_tools.search_tool(
                "needle", target="content", path=str(tmp_path), task_id=tid)
            payloads.append(json.loads(result))
            actions.append(controller.before_call("search_files", _ARGS).action)
            decision = controller.after_call("search_files", _ARGS, result)
            counts.append(decision.count)

    # The payloads really do carry the tool's own count: `_warning` from the 3rd call,
    # BLOCKED + already_searched from the 4th.
    assert "_warning" in payloads[2] and "3 times" in payloads[2]["_warning"]
    assert payloads[3]["already_searched"] == 4 and payloads[3]["guardrail_refusal"] is True
    assert payloads[4]["already_searched"] == 5

    # ...and the guardrail still sees one repeated result, not five different ones.
    assert counts == [1, 2, 3, 4, 5]
    assert actions[:4] == ["allow"] * 4
    assert actions[4] == "block", "the 5th identical call must be refused before it runs"


def test_identical_call_streak_sees_through_the_tools_repeat_counters(
    tmp_path, controller_factory
):
    """The identical-call streak (observe_call) shares the identity, so it must count too."""
    tid = "guardrail-streak-146846"
    _read_tracker.pop(tid, None)
    ops = MagicMock()
    ops.search.return_value = _search_result()
    controller = controller_factory(hard_stop_enabled=True, no_progress_block_after=3)

    observations = []
    with patch("tools.file_tools._get_file_ops", return_value=ops):
        for _ in range(3):
            result = file_tools.search_tool(
                "needle", target="content", path=str(tmp_path), task_id=tid)
            observations.append(
                controller.observe_call("search_files", _ARGS, result, tool_call_id="call-1"))

    assert observations[0].notice is None and observations[1].notice is None
    assert observations[2].kind == "identical_call_streak"
    halt = controller.halt_decision
    assert halt is not None and halt.code == "identical_call_streak_halt"


def test_zero_match_path_order_is_the_same_result(controller_factory):
    """A parallel traversal reports the same 0-match paths in another order: same finding."""
    paths = ["/repo/pkg/a.py", "/repo/pkg/b.py", "/repo/pkg/c.py"]
    first = _zero_match_payload(paths)
    reordered = _zero_match_payload(list(reversed(paths)))
    assert first != reordered  # the payloads differ, as ripgrep would emit them

    assert _result_hash(first) == _result_hash(reordered)

    controller = controller_factory(no_progress_warn_after=2)
    assert controller.after_call("search_files", _ARGS, first).count == 1
    decision = controller.after_call("search_files", _ARGS, reordered)
    assert decision.count == 2
    assert decision.action == "warn" and decision.code == "idempotent_no_progress_warning"


def test_normalization_keeps_genuinely_different_results_apart():
    """Fail-closed: folding only removes the harness's bookkeeping, never a real change."""
    paths = ["/repo/pkg/a.py", "/repo/pkg/b.py"]
    different_set = ["/repo/pkg/a.py", "/repo/pkg/d.py"]
    # A count that IS the finding ("3 match(es)") is not a repeat counter.
    different_total = _zero_match_payload(paths, total=8)
    # A different probe template means the tool found something else entirely.
    different_probe = _zero_match_payload(paths, probe="--hidden --no-ignore")

    search_hit = json.dumps({"total_count": 1, "matches": [
        {"path": "/repo/pkg/a.py", "line": 3, "content": "needle = 1"}]})
    other_hit = json.dumps({"total_count": 1, "matches": [
        {"path": "/repo/pkg/a.py", "line": 4, "content": "needle = 2"}]})
    swapped_hits = json.dumps({"total_count": 1, "matches": [
        {"path": "/repo/pkg/b.py", "line": 9, "content": "needle = 3"},
        {"path": "/repo/pkg/a.py", "line": 3, "content": "needle = 1"}]})
    hits_in_other_order = json.dumps({"total_count": 1, "matches": [
        {"path": "/repo/pkg/a.py", "line": 3, "content": "needle = 1"},
        {"path": "/repo/pkg/b.py", "line": 9, "content": "needle = 3"}]})

    base = _zero_match_payload(paths)
    hashes = {
        _result_hash(value)
        for value in (base, _zero_match_payload(different_set), different_total,
                      different_probe, search_hit, other_hit, swapped_hits)
    }
    assert len(hashes) == 7

    # Payload arrays keep their order: for matches that order is part of what the model reads.
    assert _result_hash(swapped_hits) != _result_hash(hits_in_other_order)

    # A refusal is folded to its meaning, not to "any refusal": a different target is different.
    refusal_a = json.dumps({
        "error": "BLOCKED: You have run this exact search 4 times in a row. The results have NOT "
                 "changed. You already have this information. STOP re-searching and proceed.",
        "pattern": "needle", "already_searched": 4, "guardrail_refusal": True})
    refusal_b = json.dumps({
        "error": "BLOCKED: You have run this exact search 5 times in a row. The results have NOT "
                 "changed. You already have this information. STOP re-searching and proceed.",
        "pattern": "needle", "already_searched": 5, "guardrail_refusal": True})
    refusal_other_pattern = json.dumps({
        "error": "BLOCKED: You have run this exact search 4 times in a row. The results have NOT "
                 "changed. You already have this information. STOP re-searching and proceed.",
        "pattern": "other", "already_searched": 4, "guardrail_refusal": True})
    assert _result_hash(refusal_a) == _result_hash(refusal_b)
    assert _result_hash(refusal_a) != _result_hash(refusal_other_pattern)


def test_a_real_error_is_not_the_harness_refusal_that_follows_it(controller_factory):
    """Another kind of error stays another kind of error: it is a failure, not a repeat."""
    real_error = json.dumps({"error": "Permission denied: /repo/pkg/a.py"})
    refusal = json.dumps({
        "error": "BLOCKED: You have read this exact file region 4 times in a row. The content has "
                 "NOT changed. You already have this information. STOP re-reading and proceed.",
        "path": "/repo/pkg/a.py", "already_read": 4, "guardrail_refusal": True})

    assert _result_hash(real_error) != _result_hash(refusal)
    assert classify_tool_failure("read_file", real_error)[0] is True
    assert classify_tool_failure("read_file", refusal)[0] is False

    # The failure never becomes "no progress" repetition: the failed path owns the call.
    controller = controller_factory(no_progress_warn_after=2, no_progress_block_after=2)
    controller.after_call("read_file", {"path": "/repo/pkg/a.py"}, real_error)
    decision = controller.after_call("read_file", {"path": "/repo/pkg/a.py"}, refusal)
    assert decision.code != "idempotent_no_progress_warning"
