"""Regression test for #124170: One-turn /model switch note must not be restated across compactions."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _SUMMARY_END_MARKER,
    SUMMARY_PREFIX,
    ContextCompressor,
    strip_one_turn_user_notes,
)


def _make_compressor(**overrides) -> ContextCompressor:
    kwargs = {
        "model": "test/model",
        "threshold_percent": 0.85,
        "protect_first_n": 0,
        "protect_last_n": 2,
        "quiet_mode": True,
    }
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        instance = ContextCompressor(**kwargs)
        _ = instance.context_length
    instance.tail_token_budget = 250
    return instance


def test_strip_one_turn_user_notes_patterns():
    note1 = "[Note: model was just switched from gpt-4 to claude-3-5-sonnet via Anthropic. Adjust your self-identification accordingly.]\n\nRun the migrations"
    assert strip_one_turn_user_notes(note1) == "Run the migrations"

    note2 = "[Note: switched to claude-3-opus via Anthropic. This override applies to the next turn only. Adjust your self-identification accordingly.]\nWhat is the status?"
    assert strip_one_turn_user_notes(note2) == "What is the status?"

    note3 = "[MODEL SWITCH NOTE]\nFix the CSS"
    assert strip_one_turn_user_notes(note3) == "Fix the CSS"

    note4 = "[USER INITIATED SKILLS RELOAD: reloading. Use skills_list to see the updated catalog.]\nDo the job"
    assert strip_one_turn_user_notes(note4) == "Do the job"

    # Multimodal parts
    parts = [
        {"type": "text", "text": "[Note: model was just switched from X to Y via Anthropic.]\n\nAnalyze this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]
    stripped = strip_one_turn_user_notes(parts)
    assert stripped[0]["text"] == "Analyze this image"
    assert stripped[1]["type"] == "image_url"

    # Plain message unchanged
    assert strip_one_turn_user_notes("Please review my PR") == "Please review my PR"


def test_reappend_inflight_user_task_standalone_strips_model_switch_note():
    compressor = _make_compressor()

    note = (
        "[Note: model was just switched from gpt-5 to claude-3-7-sonnet via Anthropic. "
        "Adjust your self-identification accordingly.]\n\n"
        "Investigate the payment processing timeout"
    )
    inflight = {"role": "user", "content": note}
    compressed = [
        {"role": "system", "content": "You are a helpful agent."},
        {
            "role": "assistant",
            "content": f"{SUMMARY_PREFIX}\nSummary content\n\n{_SUMMARY_END_MARKER}",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
    ]

    result = compressor._reappend_inflight_user_task(compressed, inflight)
    # The active request must be restated without the switch note
    replayed = result[-1]
    assert replayed["role"] == "user"
    assert "Investigate the payment processing timeout" in replayed["content"]
    assert "model was just switched" not in replayed["content"]
    assert _INFLIGHT_TASK_REPLAY_HEADER in replayed["content"]


def test_reappend_inflight_user_task_merged_strips_model_switch_note():
    compressor = _make_compressor()

    note = (
        "[Note: model was just switched from gpt-5 to claude-3-7-sonnet via Anthropic. "
        "Adjust your self-identification accordingly.]\n\n"
        "Investigate the payment processing timeout"
    )
    inflight = {"role": "user", "content": note}
    compressed = [
        {
            "role": "user",
            "content": f"{SUMMARY_PREFIX}\nSummary content\n\n{_SUMMARY_END_MARKER}",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
    ]

    result = compressor._reappend_inflight_user_task(compressed, inflight)
    carrier = result[0]
    assert carrier["role"] == "user"
    assert "Investigate the payment processing timeout" in carrier["content"]
    assert "model was just switched" not in carrier["content"]
    assert _INFLIGHT_TASK_REPLAY_HEADER in carrier["content"]


def test_repeated_compaction_never_restates_switch_note():
    compressor = _make_compressor()

    note = (
        "[Note: model was just switched from gpt-5 to claude-3-7-sonnet via Anthropic. "
        "Adjust your self-identification accordingly.]\n\n"
        "Long running refactoring task"
    )
    inflight = {"role": "user", "content": note}
    compressed = [
        {
            "role": "assistant",
            "content": f"{SUMMARY_PREFIX}\nSummary 1\n\n{_SUMMARY_END_MARKER}",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
    ]

    # Cycle 1
    cycle1 = compressor._reappend_inflight_user_task(compressed, inflight)
    replayed1 = cycle1[-1]
    assert "model was just switched" not in replayed1["content"]
    assert "Long running refactoring task" in replayed1["content"]

    # Cycle 2: using replayed1 as the in-flight task
    compressed2 = [
        {
            "role": "assistant",
            "content": f"{SUMMARY_PREFIX}\nSummary 2\n\n{_SUMMARY_END_MARKER}",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
    ]
    cycle2 = compressor._reappend_inflight_user_task(compressed2, replayed1)
    replayed2 = cycle2[-1]
    assert "model was just switched" not in replayed2["content"]
    assert "Long running refactoring task" in replayed2["content"]
    assert replayed2["content"].count(_INFLIGHT_TASK_REPLAY_HEADER) == 1


def test_fallback_summary_and_snapshots_strip_switch_note():
    compressor = _make_compressor()

    note = (
        "[Note: model was just switched from gpt-5 to claude-3-7-sonnet via Anthropic. "
        "Adjust your self-identification accordingly.]\n\n"
        "Write tests for the auth service"
    )
    turns = [
        {"role": "user", "content": note},
        {"role": "assistant", "content": "Working on it..."},
    ]

    anchors = compressor._fallback_anchors(turns)
    assert len(anchors["user_asks"]) == 1
    assert "model was just switched" not in anchors["user_asks"][0]
    assert "Write tests for the auth service" in anchors["user_asks"][0]

    snapshot = compressor._latest_user_task_snapshot(turns)
    assert "model was just switched" not in snapshot
    assert "Write tests for the auth service" in snapshot

    focus = compressor._derive_auto_focus_topic(turns)
    assert "model was just switched" not in focus
    assert "Write tests for the auth service" in focus
