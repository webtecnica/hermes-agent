"""Two profile homes scanned at the same time each land in ``CANONICAL_PROVIDERS`` (#159442).

Every session open scans its own profile home in its own thread. When two homes scanned at once they
shared one process-wide ``_discovering`` flag: it was saved, overwritten and restored by both scans, so
the registry->catalog mirror was skipped for one home (its row never reached the picker, Settings >
Providers > Accounts or Edit models) and, when the thread that restored last had saved ``True``, the
flag stayed stuck for the rest of the process.

These tests pin the interleavings the issue measured (V1: one home finishes entirely inside the other
home's scan; V2: the first home to enter ends first and the last one ends last), in both role orders.
Each fake plugin's import blocks on an event, so the overlap is forced instead of timed.
"""

from __future__ import annotations

import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

A = "fake-prov-a"
B = "fake-prov-b"

_GATE_MODULE = "_provider_scan_gate_159442"
_GATE_TIMEOUT = 30

# The plugin parks inside its import until the test releases it, so the scan that imports it stays
# open while another home scans.
_PLUGIN = textwrap.dedent(
    '''\
    import sys

    sys.modules["{gate}"].arrive("{name}")

    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(
        ProviderProfile(
            name="{name}",
            base_url="process://{name}",
            api_mode="chat_completions",
            auth_type="external_process",
        )
    )
    '''
)


class _ScanGate:
    """One event pair per plugin: ``arrive`` reports that its import began and then blocks."""

    def __init__(self) -> None:
        self.entered: dict[str, threading.Event] = {}
        self.release: dict[str, threading.Event] = {}

    def arm(self, *names: str) -> None:
        for name in names:
            self.entered[name] = threading.Event()
            self.release[name] = threading.Event()

    def arrive(self, name: str) -> None:
        self.entered[name].set()
        if not self.release[name].wait(_GATE_TIMEOUT):
            raise RuntimeError(f"the scan gate for {name} was never released")


def _install(home: Path, name: str, gate: str) -> None:
    plugin = home / "plugins" / "model-providers" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN.format(name=name, gate=gate), encoding="utf-8")


def _scan(home: Path, errors: list[BaseException]) -> None:
    """Bind *home* in this thread and scan it, exactly like a session open does."""
    import providers

    token = set_hermes_home_override(home)
    try:
        providers.list_providers()
    except BaseException as exc:  # handed to the test thread, never swallowed
        errors.append(exc)
    finally:
        reset_hermes_home_override(token)


@pytest.fixture()
def race(tmp_path, monkeypatch):
    """Two homes with a gated provider plugin each, plus a plugin-less home for the warm-up."""
    import providers
    from providers import plugin_reload
    from hermes_cli import models_catalog_static as catalog

    process_home = tmp_path / "process-home"
    process_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    # The tables and layers this test fills are process state: keep every change to this test.
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_SOURCES", dict(providers._SOURCES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    # Snapshot the discovery flags too: the warm-up below discovers into the copied registry, and
    # leaving ``_discovered`` True over a restored (possibly empty) registry would hide every bundled
    # profile from the tests that run after this one.
    monkeypatch.setattr(providers, "_discovered", providers._discovered)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    monkeypatch.setattr(catalog, "CANONICAL_PROVIDERS", list(catalog.CANONICAL_PROVIDERS))
    monkeypatch.setattr(catalog, "_canonical_slugs", set(catalog._canonical_slugs))
    monkeypatch.setattr(catalog, "_PROVIDER_LABELS", dict(catalog._PROVIDER_LABELS))
    # Frozen plugin copies under this test's home, not the root an earlier test happened to pick.
    monkeypatch.setattr(plugin_reload, "_snapshot_root", None, raising=False)

    gate = _ScanGate()
    gate.arm(A, B)
    gate_module = types.ModuleType(_GATE_MODULE)
    gate_module.arrive = gate.arrive
    sys.modules[_GATE_MODULE] = gate_module

    homes = {"warm": tmp_path / "home-warm", A: tmp_path / "home-a", B: tmp_path / "home-b"}
    for name in (A, B):
        _install(homes[name], name, _GATE_MODULE)

    auth = sys.modules.get("hermes_cli.auth")
    auth_keys = set(auth.PROVIDER_REGISTRY) if auth is not None else set()
    mirrored = None
    if "hermes_cli.auth" in sys.modules:
        from hermes_cli.auth_plugin_providers import PLUGIN_MIRRORED_PROVIDERS

        mirrored = set(PLUGIN_MIRRORED_PROVIDERS)

    yield gate, homes

    if auth is not None:
        for key in set(auth.PROVIDER_REGISTRY) - auth_keys:
            del auth.PROVIDER_REGISTRY[key]
    if mirrored is not None:
        PLUGIN_MIRRORED_PROVIDERS.intersection_update(mirrored)
    sys.modules.pop(_GATE_MODULE, None)
    for module in [m for m in sys.modules if m.startswith(("_hermes_user_provider", "_provider_scan_gate"))]:
        del sys.modules[module]


def _warm_up(homes, gate) -> None:
    """Run the one-shot discovery pass on a plugin-less home before the racing scans start."""
    import providers

    token = set_hermes_home_override(homes["warm"])
    try:
        providers.list_providers()
    finally:
        reset_hermes_home_override(token)
    assert providers._discovered, "the warm-up must complete process-wide discovery"
    assert not gate.entered[A].is_set() and not gate.entered[B].is_set()


@pytest.mark.parametrize("opener", [A, B], ids=["a_opens", "b_opens"])
@pytest.mark.parametrize("variant", ["finishes_inside", "ends_first"])
def test_two_homes_scanned_concurrently_both_reach_the_catalog(race, variant, opener):
    """Both providers are in ``CANONICAL_PROVIDERS`` afterwards, whichever order the scans end in."""
    gate, homes = race
    finisher = B if opener == A else A
    _warm_up(homes, gate)

    import providers
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    errors: list[BaseException] = []
    opener_thread = threading.Thread(target=_scan, args=(homes[opener], errors), daemon=True)
    opener_thread.start()
    assert gate.entered[opener].wait(_GATE_TIMEOUT), "the opening scan never reached its plugin import"

    if variant == "finishes_inside":
        gate.release[finisher].set()  # the second home scans and completes inside the first one's scan
    finisher_thread = threading.Thread(target=_scan, args=(homes[finisher], errors), daemon=True)
    finisher_thread.start()

    if variant == "ends_first":
        assert gate.entered[finisher].wait(_GATE_TIMEOUT), "the second scan never reached its plugin import"
        gate.release[opener].set()
        opener_thread.join(_GATE_TIMEOUT)
        assert not opener_thread.is_alive(), "the first scan to enter never finished"
        gate.release[finisher].set()

    finisher_thread.join(_GATE_TIMEOUT)
    assert not finisher_thread.is_alive(), f"{variant}: the second scan never finished"
    if variant == "finishes_inside":
        assert opener_thread.is_alive(), f"{variant}: the second scan did not run inside the first one's"
        gate.release[opener].set()
        opener_thread.join(_GATE_TIMEOUT)
        assert not opener_thread.is_alive(), "the opening scan never finished"

    assert not errors, errors
    slugs = {entry.slug for entry in CANONICAL_PROVIDERS}
    missing = sorted({A, B} - slugs)
    stuck = bool(providers._discovering)
    assert not missing and not stuck, (
        f"{variant} with {opener} opening: missing={missing} _discovering_stuck={stuck}"
    )
