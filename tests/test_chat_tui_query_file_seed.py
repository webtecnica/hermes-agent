"""``hermes chat --tui --query-file FILE`` seeds the session instead of opening an empty TUI.

``cmd_chat`` used to read ``--query-file`` *after* handing control to ``_launch_tui``, which never
returns (it ends in ``sys.exit``), so on the TUI path the read was dead code: the file was never
loaded, ``passthrough["query"]`` stayed ``None``, ``HERMES_TUI_QUERY`` was never exported and the
Ink session started with no first turn (#158875). The read now happens before the passthrough
snapshot, so the file content travels the same route as ``-q``.

These tests drive the real ``cmd_chat`` with a stubbed ``_launch_tui`` (the real one replaces the
process); no TUI, gateway or provider is started.
"""

from __future__ import annotations

import sys
import types

import pytest

QUERY = 'he said "hi" $(id) `uname`\nsecond line\n'


class _Launched(Exception):
    """Raised by the stubbed ``_launch_tui`` (the real one ``sys.exit``s and never returns)."""


def _parse(argv):
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    return parser.parse_args(argv)


def _args(argv):
    args = _parse(argv)
    import hermes_cli.main as main_mod

    args.func = main_mod.cmd_chat
    return args


@pytest.fixture
def harness(monkeypatch):
    """cmd_chat with every startup side effect stubbed; returns (main_mod, calls, seen, cli_calls)."""
    import hermes_cli.main as main_mod
    import hermes_cli.main_query_file as query_file_mod
    import hermes_cli.observability.shared_metrics_consent as consent
    import hermes_cli.observability.shared_metrics_process as proc

    calls: list[str] = []
    seen: dict = {}
    cli_calls: dict = {}

    def launch_tui(*_a, **kw):
        calls.append("launch_tui")
        seen.update(kw)
        raise _Launched

    def classic_main(**kw):
        calls.append("cli_main")
        cli_calls.update(kw)

    fake_cli = types.ModuleType("cli")
    fake_cli.main = classic_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    # Spy that keeps the real reader: the ordering assertion is about the real call, not a stand-in.
    real_read = query_file_mod._read_query_file

    def read_spy(args):
        calls.append("read_query_file")
        real_read(args)

    monkeypatch.setattr(query_file_mod, "_read_query_file", read_spy)
    monkeypatch.setattr(main_mod, "_launch_tui", launch_tui)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(main_mod, "_resolve_chat_session_args", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _args: None)
    monkeypatch.setattr(consent, "offer_consent_before_chat", lambda _args: None)
    monkeypatch.setattr(proc, "begin_process", lambda _name: None)

    return main_mod, calls, seen, cli_calls


@pytest.fixture(autouse=True)
def _no_tui_env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI", raising=False)


def test_tui_query_file_is_read_before_the_launch_and_seeds_it(tmp_path, harness):
    """The seed reaches ``_launch_tui(query=...)`` — the same kwarg ``-q`` fills."""
    main_mod, calls, seen, _cli_calls = harness
    qfile = tmp_path / "seed.txt"
    qfile.write_text(QUERY, encoding="utf-8")
    args = _args(["chat", "--tui", "--query-file", str(qfile)])

    with pytest.raises(_Launched):
        main_mod.cmd_chat(args)

    assert calls == ["read_query_file", "launch_tui"], calls
    assert args.query == QUERY  # filled before the launch, not after a dead return
    assert seen["query"] == QUERY


def test_tui_query_does_not_flip_use_tui_or_drop_the_seed(tmp_path, harness):
    """A plain ``--tui --query`` keeps working (control) and carries the seed the same way."""
    main_mod, calls, seen, _cli_calls = harness
    args = _args(["chat", "--tui", "-q", "inline"])

    with pytest.raises(_Launched):
        main_mod.cmd_chat(args)

    assert calls == ["read_query_file", "launch_tui"], calls
    assert seen["query"] == "inline"


def test_tui_empty_query_file_fails_before_any_launch(tmp_path, harness, capsys):
    main_mod, calls, _seen, _cli_calls = harness
    qfile = tmp_path / "empty.txt"
    qfile.write_text("  \n", encoding="utf-8")
    args = _args(["chat", "--tui", "--query-file", str(qfile)])

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert calls == ["read_query_file"], calls
    assert f"Error: --query-file {qfile} is empty" in capsys.readouterr().err


def test_tui_unreadable_query_file_fails_before_any_launch(tmp_path, harness, capsys):
    main_mod, calls, _seen, _cli_calls = harness
    missing = tmp_path / "nope.txt"
    args = _args(["chat", "--tui", "--query-file", str(missing)])

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)

    assert exc.value.code == 2
    assert calls == ["read_query_file"], calls
    assert f"Error: cannot read --query-file {missing}" in capsys.readouterr().err


def test_tui_query_file_dash_still_reads_stdin(monkeypatch, harness):
    import io

    main_mod, calls, seen, _cli_calls = harness
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped seed\n"))
    args = _args(["chat", "--tui", "--query-file", "-"])

    with pytest.raises(_Launched):
        main_mod.cmd_chat(args)

    assert calls == ["read_query_file", "launch_tui"], calls
    assert seen["query"] == "piped seed\n"


def test_query_and_query_file_stay_mutually_exclusive(tmp_path, harness, capsys):
    """Both refusals keep their rc=2: argparse at the CLI, the reader for programmatic callers."""
    main_mod, calls, _seen, _cli_calls = harness
    qfile = tmp_path / "seed.txt"
    qfile.write_text("hello", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        _args(["chat", "-q", "inline", "--query-file", str(qfile)])
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    # Programmatic caller that fills the namespace directly: the reader's own guard still refuses.
    args = _args(["chat", "--tui", "--query-file", str(qfile)])
    args.query = "inline"
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(args)
    assert exc.value.code == 2
    assert "Error: -q/--query and --query-file are mutually exclusive" in capsys.readouterr().err
    assert "launch_tui" not in calls and "cli_main" not in calls, calls


def test_classic_path_gets_the_file_as_query_and_never_launches_the_tui(tmp_path, harness):
    main_mod, calls, _seen, cli_calls = harness
    qfile = tmp_path / "seed.txt"
    qfile.write_text(QUERY, encoding="utf-8")
    args = _args(["chat", "--cli", "--query-file", str(qfile)])

    main_mod.cmd_chat(args)

    assert calls == ["read_query_file", "cli_main"], calls
    assert cli_calls["query"] == QUERY


def test_classic_path_without_query_file_is_unchanged(harness):
    main_mod, calls, _seen, cli_calls = harness
    args = _args(["chat", "--cli"])

    main_mod.cmd_chat(args)

    assert calls == ["read_query_file", "cli_main"], calls
    assert "query" not in cli_calls  # None kwargs are still filtered out of cli.main
    assert args.query is None
