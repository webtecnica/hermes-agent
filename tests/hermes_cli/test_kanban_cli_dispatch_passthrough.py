"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_cli
import pytest

# The hermes_cli object as loaded when this module is collected. The fixture
# below purges hermes_cli* from sys.modules so the test under it imports a
# copy bound to the temporary home; that copy must be dropped and this object
# put back on teardown, or every later import binds to the real HERMES_HOME
# (#159065).
_ORIGINAL_HERMES_CLI = sys.modules["hermes_cli"]


def _is_hermes_module(name):
    return (
        name.startswith("hermes_cli")
        or name.startswith("hermes_state")
        or name == "hermes_constants"
    )


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Snapshot the modules before purging them: monkeypatch restores
    # HERMES_HOME at teardown but knows nothing about sys.modules, so without
    # the restore below the re-imported copies outlive the fixture and bind
    # to whichever home is current then (the real one).
    saved = {name: mod for name, mod in sys.modules.items() if _is_hermes_module(name)}
    for name in saved:
        del sys.modules[name]
    try:
        yield test_home
    finally:
        for name in [name for name in sys.modules if _is_hermes_module(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_01_isolated_kanban_home_drops_and_reimports_hermes_modules(isolated_kanban_home):
    """Enter the fixture and import the kanban CLI under the temporary home."""
    from hermes_cli import kanban as kb_cli

    assert kb_cli is not None
    # The purge really happened: this is a fresh copy, not the collection-time
    # one (otherwise the fixture would not be exercising the re-import path).
    assert sys.modules["hermes_cli"] is not _ORIGINAL_HERMES_CLI


def test_02_isolated_kanban_home_restores_hermes_modules():
    """#159065: the fixture must put the collection-time modules back.

    Runs without the fixture on purpose: it inspects what the previous
    test's fixture teardown left behind. The re-imported copies must be
    gone and the collection-time objects back, or later modules in the same
    process bind to the fixture's temporary home (or to the real home once
    monkeypatch restores HERMES_HOME) instead of the ones every other test
    imported.
    """
    leaked = sys.modules.get("hermes_cli")
    assert leaked is _ORIGINAL_HERMES_CLI, (
        "isolated_kanban_home left a different hermes_cli object in "
        "sys.modules; later imports would bind to the real HERMES_HOME "
        "(#159065)"
    )
    import hermes_cli as reimported

    assert reimported is _ORIGINAL_HERMES_CLI


def test_cli_dispatch_passes_max_in_progress_from_config(isolated_kanban_home, monkeypatch):
    """#33488: hermes kanban dispatch must pass kanban.max_in_progress from
    config to dispatch_once. Without this, the global concurrency cap is
    unreachable from the CLI even though it works from the gateway."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd

    # Configure max_in_progress in the loaded config.
    fake_config = {
        "kanban": {
            "max_in_progress": 3,
            "max_spawn": 5,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: fake_config
    )

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    # Every config value must have reached dispatch_once.
    assert captured.get("max_in_progress") == 3, (
        f"CLI must pass kanban.max_in_progress from config; got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("max_spawn") == 5, (
        f"CLI must pass kanban.max_spawn from config when --max is not provided; got {captured.get('max_spawn')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd

    fake_config = {"kanban": {"max_spawn": 10}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    monkeypatch.setattr(
        kbd, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    args = argparse.Namespace(dry_run=True, max=2, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    assert captured.get("max_spawn") == 2, (
        f"CLI --max=2 must override config kanban.max_spawn=10; got {captured.get('max_spawn')!r}"
    )


