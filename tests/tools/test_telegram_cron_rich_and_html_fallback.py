"""Regression tests for Issue #124413:

Cron/standalone Telegram deliveries with rich messages, <details> normalization,
and graded MarkdownV2 fallback on HTML parse errors.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.send_message_senders import (
    _is_telegram_html,
    _normalize_telegram_details_tags,
    _send_telegram,
    _telegram_format,
    _telegram_send_text_chunk,
)


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    import sys
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=_MessageEntity,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def test_normalize_telegram_details_tags():
    """<details><summary>Title</summary> content </details> should become **Title** and markdown text."""
    raw = "<details><summary>Quiz Answers</summary>\n1. Paris\n2. London\n</details>"
    normalized = _normalize_telegram_details_tags(raw)
    assert "**Quiz Answers**" in normalized
    assert "<details" not in normalized
    assert "</details>" not in normalized
    assert "<summary" not in normalized


def test_is_telegram_html():
    """Only telegram-supported HTML tags should return True."""
    assert _is_telegram_html("<b>Bold</b> and <code>code</code>") is True
    assert _is_telegram_html("<details><summary>Foo</summary></details>") is False
    assert _is_telegram_html("<div>Not supported</div>") is False
    assert _is_telegram_html("Plain text with no tags") is False


def test_telegram_format_with_details_and_table(monkeypatch: pytest.MonkeyPatch):
    """Message containing tables and <details> should format as MarkdownV2, not HTML."""
    _install_telegram_mock(monkeypatch, MagicMock())

    msg = (
        "| Day | Topic |\n"
        "|---|---|\n"
        "| Mon | Math |\n\n"
        "<details><summary>Answers</summary>\n1. 42\n</details>"
    )
    formatted, mode, has_html = _telegram_format(msg)
    assert mode == "MarkdownV2"
    assert has_html is False
    assert "<details" not in formatted
    assert "Answers" in formatted


def test_telegram_format_with_valid_html(monkeypatch: pytest.MonkeyPatch):
    """Message containing only supported HTML tags should use ParseMode.HTML."""
    _install_telegram_mock(monkeypatch, MagicMock())

    msg = "<b>Title</b>\n<i>Italic</i>\n<a href=\"https://example.com\">link</a>"
    formatted, mode, has_html = _telegram_format(msg)
    assert mode == "HTML"
    assert has_html is True


@pytest.mark.asyncio
async def test_telegram_send_text_chunk_html_failure_retries_markdownv2(monkeypatch: pytest.MonkeyPatch):
    """If ParseMode.HTML fails with a parse error, it must retry with MarkdownV2 before plain text."""
    _install_telegram_mock(monkeypatch, MagicMock())

    bot = MagicMock()
    # 1st call (HTML): raises parse error
    # 2nd call (MarkdownV2): succeeds
    bot.send_message = AsyncMock(side_effect=[
        Exception("Can't parse entities: unsupported start tag 'details'"),
        SimpleNamespace(message_id=42),
    ])

    text_kwargs = {}
    chunk = "<details><summary>Quiz</summary>Body</details>"

    res = await _telegram_send_text_chunk(
        bot,
        chat_id=12345,
        chunk=chunk,
        parse_mode="HTML",
        has_html=True,
        text_kwargs=text_kwargs,
    )

    assert res.message_id == 42
    assert bot.send_message.call_count == 2
    # Verify second call was MarkdownV2
    second_call_kwargs = bot.send_message.call_args_list[1][1]
    assert second_call_kwargs["parse_mode"] == "MarkdownV2"
    assert second_call_kwargs["chat_id"] == 12345


@pytest.mark.asyncio
async def test_send_telegram_rich_messages_attempted(monkeypatch: pytest.MonkeyPatch):
    """When telegram.extra.rich_messages is enabled and message has rich constructs, sendRichMessage is called."""
    bot = MagicMock()
    # async coroutine for do_api_request
    bot.do_api_request = AsyncMock(return_value={"message_id": 999})
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=100))
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))

    # Mock load_config_readonly to enable rich_messages
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"telegram": {"extra": {"rich_messages": True}}},
    )

    msg = (
        "| Col1 | Col2 |\n"
        "|---|---|\n"
        "| A | B |\n\n"
        "<details><summary>Hidden</summary>\nSecret\n</details>"
    )

    res = await _send_telegram(
        token="FAKE_TOKEN",
        chat_id="123456",
        message=msg,
    )

    assert res.get("success") is True
    assert res.get("message_id") == "999"
    # do_api_request("sendRichMessage", ...) must have been called
    bot.do_api_request.assert_called_once()
    assert bot.do_api_request.call_args[0][0] == "sendRichMessage"
    # send_message should not have been called because rich succeeded
    bot.send_message.assert_not_called()
