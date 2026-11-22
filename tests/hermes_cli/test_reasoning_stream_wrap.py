"""#158905: the live reasoning flush must not hard-wrap a partial line mid-word.

Providers such as Anthropic stream reasoning without newlines. The CLI force-flushed the
buffer every ~80 characters, and ``_cprint`` closes the line it prints, so a long thought
was cut ~80 characters in — usually in the middle of a word — regardless of the terminal
width. The flush itself is wanted (reasoning stays visible before the turn ends), so the
buffer now flushes up to its last space and keeps the tail buffered. Text that arrives
with newlines keeps rendering exactly as before.
"""
import re

import pytest


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


SENTENCE = (
    "I should inspect the streaming mixin first and then check how the reasoning "
    "buffer is flushed before the response arrives so the user keeps seeing progress"
)

_CHROME = ("─", "┌", "└", "∴", "☤", "╭", "╰", "│")


def _reasoning_rows(lines: list[str]) -> list[str]:
    """Emitted lines minus the box frame / separator / thought summary."""
    return [ln for ln in lines if ln and not ln.lstrip().startswith(_CHROME)]


@pytest.fixture
def harness(monkeypatch):
    from cli import HermesCLI
    import cli as climod
    import hermes_cli.cli_stream_mixin as mixin

    def build(modern: bool, width: int = 80):
        cli = HermesCLI.__new__(HermesCLI)
        cli._modern_layout = modern
        cli.show_reasoning, cli.reasoning_full = True, False
        cli.final_response_markdown, cli.show_timestamps = "raw", False
        cli._reset_stream_state()
        emitted: list[str] = []
        monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
        # Deterministic width: the buffer's hard cap for a space-less token.
        monkeypatch.setattr(mixin, "_terminal_columns", lambda: width)
        monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: width)
        return cli, emitted

    return build


@pytest.mark.parametrize("modern", [True, False])
def test_long_reasoning_without_newlines_breaks_at_word_boundaries(harness, modern):
    cli, emitted = harness(modern)
    # Token-sized deltas, none aligned to a word boundary: the provider sends text, not lines.
    deltas = [SENTENCE[i : i + 7] for i in range(0, len(SENTENCE), 7)]

    for delta in deltas[:-1]:
        cli._stream_reasoning_delta(delta)

    # (a) Something is already on screen before the stream ends: still real-time.
    early = _reasoning_rows(emitted)
    assert early, emitted

    cli._stream_reasoning_delta(deltas[-1])
    cli._close_reasoning_box()
    rows = _reasoning_rows(emitted)

    # (b) Every line flushed while streaming ends on a consumed space; only the final
    # partial tail printed by the box close may end mid-word.
    assert all(row.endswith(" ") for row in early), early
    assert all(row.endswith(" ") for row in rows[:-1]), rows
    assert not rows[-1].endswith(" "), rows

    # (c) Nothing was dropped: the emitted text rebuilds the sentence exactly.
    assert "".join(rows) == SENTENCE
    assert cli._reasoning_buf == ""


@pytest.mark.parametrize("modern", [True, False])
def test_tokens_without_any_space_wait_for_terminal_width(harness, modern):
    cli, emitted = harness(modern, width=60)
    cli._stream_reasoning_delta("x" * 59)
    assert _reasoning_rows(emitted) == []
    assert cli._reasoning_buf == "x" * 59

    cli._stream_reasoning_delta("x")
    assert _reasoning_rows(emitted) == ["x" * 60]
    assert cli._reasoning_buf == ""


@pytest.mark.parametrize("modern", [True, False])
def test_reasoning_with_newlines_is_unchanged(harness, modern):
    cli, emitted = harness(modern)
    long_line = " ".join(["word"] * 30)  # 149 characters: longer than the old 80-column flush
    cli._stream_reasoning_delta("First thought.\n" + long_line + "\n")
    cli._close_reasoning_box()
    # Newline-terminated lines print whole, exactly as before — no word-boundary splitting.
    assert _reasoning_rows(emitted) == ["First thought.", long_line]
    assert cli._reasoning_buf == ""
