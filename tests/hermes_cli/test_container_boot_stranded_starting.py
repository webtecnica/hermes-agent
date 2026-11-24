"""Bounded one-time autostart retry for a stranded legacy ``gateway_state=starting`` marker.

A gateway hard-killed between its first ``gateway_state=starting`` write (``gateway/run_startup.py``)
and the ``running`` write leaves ``{"gateway_state": "starting"}`` with no ``desired_state`` — the
``desired_state`` key is only stamped by the s6 lifecycle commands, so a fresh volume or a
pre-lifecycle install has none. Reading that literal marker as the prior state registered the root
slot DOWN on every boot, and nothing ever rewrote the file: one interrupted boot disabled gateway
autostart permanently (observed as 13+ boots at ``action=registered prior_exit=unclean``).

The reconciler now spends ONE retry and records it in the same ``gateway_state.json`` (#159527). A
later boot that still finds the stranded marker with the retry already spent registers down, so a
genuine crash loop stays bounded and never becomes an unbounded autostart loop.

Hermetic: fake ``$HERMES_HOME`` under ``tmp_path``, no real s6 supervision tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.container_boot import (
    ReconcileAction,
    reconcile_profile_gateways,
)

pytestmark = pytest.mark.platforms("linux")

_RETRY_KEY = "stranded_starting_retry"


@pytest.fixture(autouse=True)
def _hermetic_container_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ``_read_container_argv()`` to empty and neuter the chown hook.

    ``_read_container_argv()`` walks the whole ``/proc`` table looking for a ``main-wrapper.sh``
    argv; on a host that also runs hermes containers that scan can pick up a foreign ``gateway run``
    and make ``_maybe_migrate_legacy_gateway_run_state`` synthesize state (same fixture as
    ``test_container_boot.py``). No test here relies on a real container CMD.
    """
    monkeypatch.setattr("hermes_cli.container_boot._read_container_argv", lambda: ())
    monkeypatch.setattr("hermes_cli.service_manager._chown_hermes", lambda path: None)


def _write_state(home: Path, payload: dict) -> Path:
    state_file = home / "gateway_state.json"
    state_file.write_text(json.dumps(payload), encoding="utf-8")
    return state_file


def _read_state(home: Path) -> dict:
    return json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))


def _unclean_sentinel(home: Path) -> None:
    """The lifecycle sentinel a SIGKILLed gateway leaves behind (``phase=running`` → unclean)."""
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "gateway.lifecycle.json").write_text(json.dumps({"phase": "running"}),
                                                      encoding="utf-8")


def _reconcile(home: Path, scandir: Path, *, dry_run: bool = False) -> ReconcileAction:
    actions = reconcile_profile_gateways(hermes_home=home, scandir=scandir, dry_run=dry_run,
                                         container_argv=())
    return next(a for a in actions if a.profile == "default")


def test_first_boot_spends_one_retry_and_starts_the_root_slot(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": "starting", "timestamp": 1234567890})
    _unclean_sentinel(home)

    action = _reconcile(home, tmp_path / "service")

    assert action.action == "started"
    assert action.prior_state == "running"
    assert action.prior_exit == "unclean"
    # The retry is recorded in the SAME file — no new state file, no migration.
    state = _read_state(home)
    assert state[_RETRY_KEY]["count"] == 1
    assert isinstance(state[_RETRY_KEY]["at"], str) and state[_RETRY_KEY]["at"]
    # The legacy marker itself is left alone: `gateway_state` is the runtime's field.
    assert state["gateway_state"] == "starting"
    assert "desired_state" not in state


def test_second_boot_on_the_same_stranded_marker_registers_down_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": "starting", "timestamp": 1234567890})
    _unclean_sentinel(home)

    assert _reconcile(home, tmp_path / "service").action == "started"

    # The retry did not help (the gateway died mid-boot again): the marker is unchanged, so the
    # second boot must NOT autostart. Bounded, not a crash loop.
    with caplog.at_level("WARNING", logger="hermes_cli.container_boot"):
        second = _reconcile(home, tmp_path / "service")

    assert second.action == "registered"
    assert second.prior_state == "starting"
    assert second.prior_exit == "unclean"
    assert _read_state(home)[_RETRY_KEY]["count"] == 1  # not incremented into an endless spin
    warning = " ".join(record.getMessage() for record in caplog.records)
    assert "stranded gateway_state=starting without desired_state" in warning
    assert "retry already spent" in warning
    # And it stays down on every later boot too.
    assert _reconcile(home, tmp_path / "service").action == "registered"


def test_dry_run_reports_the_start_without_persisting_the_retry(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    state_file = _write_state(home, {"gateway_state": "starting", "timestamp": 1234567890})
    before = state_file.read_bytes()
    _unclean_sentinel(home)

    action = _reconcile(home, tmp_path / "service", dry_run=True)

    assert action.action == "started"
    assert action.prior_state == "running"
    assert state_file.read_bytes() == before  # dry_run never touches the file


def test_desired_state_running_still_starts_the_root_slot(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": "starting", "desired_state": "running"})

    action = _reconcile(home, tmp_path / "service")

    assert action.action == "started"
    assert action.prior_state == "running"
    # Operator intent wins: no retry bookkeeping is involved at all.
    assert _RETRY_KEY not in _read_state(home)


@pytest.mark.parametrize("transient", ["draining", "degraded"])
def test_transient_running_substates_still_start(tmp_path: Path, transient: str) -> None:
    """Regression guard for #42675: `draining`/`degraded` stay mapped to `running`."""
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": transient, "timestamp": 1234567890})

    action = _reconcile(home, tmp_path / "service")

    assert action.action == "started"
    assert action.prior_state == "running"
    assert _RETRY_KEY not in _read_state(home)


def test_desired_state_stopped_never_starts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": "starting", "desired_state": "stopped"})

    assert _reconcile(home, tmp_path / "service").action == "registered"
    assert _RETRY_KEY not in _read_state(home)


def test_a_start_that_reached_running_earns_a_fresh_retry(tmp_path: Path) -> None:
    """The retry record is cleared once the marker moved on, so every interrupted boot gets one."""
    home = tmp_path / "home"
    home.mkdir()
    _write_state(home, {"gateway_state": "starting", "timestamp": 1234567890})

    assert _reconcile(home, tmp_path / "service").action == "started"
    assert _read_state(home)[_RETRY_KEY]["count"] == 1

    # The gateway came up: the runtime writer merges `gateway_state=running` over the file, keeping
    # the retry record. The next boot sees a healthy marker and clears the spent record.
    _write_state(home, {**_read_state(home), "gateway_state": "running"})
    assert _reconcile(home, tmp_path / "service").action == "started"
    assert _RETRY_KEY not in _read_state(home)

    # A LATER interrupted boot therefore gets its own one-time retry.
    _write_state(home, {**{k: v for k, v in _read_state(home).items()},
                        "gateway_state": "starting"})
    fresh = _reconcile(home, tmp_path / "service")
    assert fresh.action == "started"
    assert _read_state(home)[_RETRY_KEY]["count"] == 1
