"""A Kanban worker's toolsets are resolved WITHOUT importing the assignee profile's plugins.

``_resolve_worker_cli_toolsets`` swaps ``HERMES_HOME`` to the assignee's profile and asks
``_get_platform_tools`` for its CLI toolsets, which needs that profile's plugin toolset keys. Serving
them from a full plugin discovery of the foreign home ran every one of that profile's ``register()``
hooks — threads, DB handles — inside the dispatcher's process, and left a ``PluginManager`` in
``_plugin_managers_by_home`` (plus everything it imported) for the rest of the gateway's life: one
per dispatched profile, never evicted, still there after the profile was deleted (#146992).

The inventory those keys come from is the one the profile persists when IT discovers its plugins
(``cache/plugin_toolset_keys.json``), so the pin the worker receives is unchanged.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import yaml

from hermes_cli import plugins as plugins_mod
from hermes_cli import profiles
from hermes_cli.kanban_db_dispatch import _resolve_worker_cli_toolsets
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

_PLUGIN_BODY = '''
import threading


def register(ctx):
    ctx.register_tool(
        name="probe_{index}_ping",
        toolset="{key}",
        schema={{"name": "probe_{index}_ping", "description": "probe",
                 "parameters": {{"type": "object", "properties": {{}}}}}},
        handler=lambda **kwargs: "pong",
    )
{thread}'''

_THREAD_BODY = '''    # A resident plugin thread: what a real plugin's register() leaves behind (gateway adapters,
    # schedulers). Started only by an import, so its absence proves no import happened.
    threading.Thread(target=threading.Event().wait, name="{thread_name}", daemon=True).start()
'''


def _write_plugin(home: Path, index: int, *, thread: bool) -> str:
    """Install one plugin that contributes toolset ``probe<index>`` under *home*. Returns the key."""
    key = f"probe{index}"
    plugin = home / "plugins" / key
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(
        json.dumps({"name": key, "version": "1.0.0"}), encoding="utf-8")
    (plugin / "__init__.py").write_text(
        _PLUGIN_BODY.format(
            index=index, key=key,
            thread=_THREAD_BODY.format(thread_name=_thread_name(index)) if thread else "",
        ),
        encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [key]}}), encoding="utf-8")
    return key


def _thread_name(index: int) -> str:
    return f"probe-plugin-thread-{index}"


def _persist_inventory(home: Path, keys: list[str]) -> None:
    """Write the inventory the profile itself persists when it discovers its plugins."""
    cache = home / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "plugin_toolset_keys.json").write_text(
        json.dumps({"toolset_keys": sorted(keys), "portable_mcp": []}), encoding="utf-8")


def _modules_under(home: Path) -> list[str]:
    """Imported modules whose source file lives under *home* (a foreign profile's plugin code)."""
    root = str(home.resolve())
    return sorted(
        name for name, module in list(sys.modules.items())
        if str(getattr(module, "__file__", None) or "").startswith(root)
    )


def test_foreign_profiles_leave_no_plugin_manager_threads_or_imports(tmp_path):
    """Three dispatched profiles: three toolset pins, zero resident plugin state."""
    homes = []
    for index in range(3):
        home = tmp_path / f"profile{index}"
        key = _write_plugin(home, index, thread=True)
        _persist_inventory(home, [key])
        homes.append((home, index, key))

    for home, index, key in homes:
        toolsets = _resolve_worker_cli_toolsets(str(home))

        # The plugin toolset still reaches the worker: the pin is not silently narrowed.
        assert toolsets, f"{home.name}: no toolsets resolved"
        assert key in toolsets, f"{home.name}: plugin toolset missing from {sorted(toolsets)}"

        # ... but resolving it left nothing of that profile in this process.
        assert not [
            name for name in plugins_mod._plugin_managers_by_home
            if str(name).startswith(str(home.resolve()))
        ], f"{home.name}: a PluginManager stayed resident"
        assert _modules_under(home) == [], f"{home.name}: imported {_modules_under(home)}"
        assert _thread_name(index) not in {t.name for t in threading.enumerate()}, \
            f"{home.name}: its plugin thread is still running"

    assert plugins_mod._plugin_managers_by_home == {}


def test_the_dispatchers_own_home_keeps_its_manager_and_plugin(tmp_path):
    """Control: the process's own profile is served exactly as before — no eviction, no suppression."""
    own = get_hermes_home()
    key = _write_plugin(own, 9, thread=False)

    manager = plugins_mod.get_plugin_manager()
    toolsets = _resolve_worker_cli_toolsets(str(own))

    assert key in toolsets
    assert key in {ts_key for ts_key, _, _ in plugins_mod.get_plugin_toolsets()}
    assert plugins_mod.get_plugin_manager() is manager
    assert plugins_mod._plugin_managers_by_home.get(Path(own).resolve()) is manager


def test_discovery_persists_the_inventory_the_dispatcher_reads_instead_of_importing(tmp_path):
    """The profile's own discovery must write the inventory a foreign reader is served from."""
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / "profile-owner"
    key = _write_plugin(home, 5, thread=False)

    token = set_hermes_home_override(str(home))
    try:
        discover_plugins()
    finally:
        reset_hermes_home_override(token)

    persisted = home / "cache" / "plugin_toolset_keys.json"
    assert key in json.loads(persisted.read_text(encoding="utf-8"))["toolset_keys"]
    assert key in _resolve_worker_cli_toolsets(str(home))


def test_deleting_a_profile_drops_its_cached_plugin_manager(tmp_path, monkeypatch):
    """A deleted profile must not keep a manager (and its plugin threads) in this process."""
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)

    home = profiles.create_profile("probe-plugin-delete", no_alias=True)
    token = set_hermes_home_override(str(home))
    try:
        manager = plugins_mod.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    assert plugins_mod._plugin_managers_by_home.get(Path(home).resolve()) is manager

    profiles.delete_profile("probe-plugin-delete", yes=True)

    assert Path(home).resolve() not in plugins_mod._plugin_managers_by_home
