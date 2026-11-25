"""``kanban notify-subscribe --platform`` must refuse a platform no notifier delivers (#159821).

``desktop`` is the reported case: it is not a gateway ``Platform`` (``Platform("desktop")`` raises
``ValueError``) and the TUI poller only claims ``platform='tui'`` rows, so the CLI accepted the row
with exit 0 and both notifiers then dropped it forever — the gateway at DEBUG, the TUI poller
because the platform never matches. These cases pin the early refusal and the alternatives its
message has to name, plus the two neighbours that must keep working: a gateway platform and a
bundled plugin platform. The latter matters because ``Platform`` is NOT a closed enum:
``Platform._missing_`` mints a pseudo-member for every bundled/registered plugin adapter, so a
validation built from a hardcoded literal list would refuse a legitimate platform.

The last case pins the second half of the issue: an orphan row already in the DB is surfaced ONCE
at WARNING (not on every poll, and not buried at DEBUG) without advancing its cursor.
"""
import argparse
import logging
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.kanban_watchers_notifier import _notifier_collect
from gateway.run import GatewayRunner
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

SESSION_KEY = "tui-session-key-abc123"


@pytest.fixture
def task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("PLATFORM", "CHAT_ID", "THREAD_ID", "USER_ID", "USER_ID_ALT", "SESSION_KEY"):
        monkeypatch.delenv(f"HERMES_SESSION_{name}", raising=False)
    kb.init_db()
    with kbc.connect() as conn:
        return kb.create_task(conn, title="platform validation card", assignee="worker")


def _parse(*argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["kanban", "notify-subscribe", *argv])


def _subscribe(*argv):
    return kc.kanban_command(_parse(*argv))


def _rows(task):
    with kbc.connect() as conn:
        return kbn.list_notify_subs(conn, task)


def test_desktop_platform_is_refused_before_any_row_is_written(task, capsys):
    """The reported command: exit non-zero, no row, and a message naming what does deliver."""
    with pytest.raises(SystemExit) as exit_info:
        _parse(task, "--platform", "desktop", "--chat-id", SESSION_KEY)
    assert exit_info.value.code != 0
    err = capsys.readouterr().err
    assert "desktop" in err
    assert "tui" in err and "--chat-id" in err, "the error must point Desktop/TUI callers at tui"
    assert "gateway platform" in err, "the error must offer a durable gateway platform too"
    assert _rows(task) == []


def test_subscribe_rechecks_the_platform_even_when_argparse_is_bypassed(task, capsys):
    """``_cmd_notify_subscribe`` is also reached programmatically; it must not trust the caller."""
    args = _parse(task, "--platform", "telegram", "--chat-id", "chat1")
    args.platform = "desktop"
    assert kc.kanban_command(args) != 0
    assert "desktop" in capsys.readouterr().err
    assert _rows(task) == []


def test_gateway_platform_subscription_is_unchanged(task, capsys):
    """Control: a platform with a gateway adapter keeps writing the row and the same confirmation."""
    assert _subscribe(task, "--platform", "telegram", "--chat-id", "chat1") == 0
    assert "Subscribed telegram:chat1" in capsys.readouterr().out
    assert [row["platform"] for row in _rows(task)] == ["telegram"]


def test_tui_with_a_session_key_is_accepted(task, capsys):
    """Control: ``tui`` is delivered by the TUI poller, not by a gateway adapter, and stays valid."""
    assert _subscribe(task, "--platform", "tui", "--chat-id", SESSION_KEY) == 0
    assert f"Subscribed tui:{SESSION_KEY}" in capsys.readouterr().out
    assert [row["platform"] for row in _rows(task)] == ["tui"]


def test_bundled_plugin_platform_stays_accepted(task, capsys):
    """A bundled plugin adapter is a dynamic ``Platform`` member, never an enum literal."""
    bundled, aliases = Platform._scan_bundled_plugin_platforms()
    assert "irc" in bundled and "irc" not in Platform.__members__, "fixture assumes irc is plugin-only"
    assert Platform("irc").value == "irc"
    assert _subscribe(task, "--platform", "irc", "--chat-id", "#channel") == 0
    assert [row["platform"] for row in _rows(task)] == ["irc"]
    capsys.readouterr()

    # A manifest ``name:`` alias resolves to its directory-name member, and the row stores the
    # canonical value the notifier routes on — an alias row would be a fresh orphan.
    alias, canonical = next(iter(aliases.items()))
    assert _subscribe(task, "--platform", alias, "--chat-id", "#other") == 0
    assert sorted(row["platform"] for row in _rows(task)) == sorted(["irc", canonical])


class RecordingAdapter:
    supports_async_delivery = True

    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: RecordingAdapter()}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._kanban_notifier_profile = "default"
    runner._kanban_dispatcher_lock_handle = object()
    runner.config = GatewayConfig()
    return runner


def _collect(runner):
    return _notifier_collect(runner, kb, notifier_profile="default", gc_due=False, gc_retention_days=30)


def test_orphan_desktop_row_warns_once_instead_of_debug_every_poll(task, caplog, monkeypatch):
    """A row an older CLI (or a direct write) left behind is a permanent dead-end: say so ONCE."""
    from gateway import kanban_watchers_notifier as notifier

    monkeypatch.setattr(notifier, "_UNROUTABLE_WARNED", set())
    runner = _runner()
    with kbc.connect() as conn:
        kbn.add_notify_sub(conn, task_id=task, platform="desktop", chat_id=SESSION_KEY,
                           notifier_profile="default")
        # A tui row is skipped by the gateway too, but it IS delivered elsewhere: it must stay quiet.
        kbn.add_notify_sub(conn, task_id=task, platform="tui", chat_id=SESSION_KEY,
                           notifier_profile="default")
        kb.complete_task(conn, task, result="finished")

    with caplog.at_level(logging.WARNING, logger=notifier.logger.name):
        assert not _collect(runner)
        assert not _collect(runner)  # second poll tick: the warning must not repeat

    warnings = [r for r in caplog.records
                if r.levelno == logging.WARNING and task in r.getMessage() and "desktop" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert "notify-unsubscribe" in warnings[0].getMessage()
    # Only the desktop row warned: the tui row is skipped by this gateway too, but it IS delivered
    # by the TUI poller, so it must stay quiet.
    assert len([r for r in caplog.records if task in r.getMessage()]) == 1

    # The skip is not a delivery attempt: the cursor never advanced, the event is still unseen.
    with kbc.connect() as conn:
        assert kbn.unseen_events_for_sub(conn, task_id=task, platform="desktop", chat_id=SESSION_KEY,
                                         thread_id="", kinds=["completed"])[1]
