"""``cua-driver doctor --json`` costs ~1.8 s on Windows, so ``computer_use_status()`` memoizes it (#157424).

Every test drives a real fake driver script rather than patching ``_run``: invocation counting is the measurement
(a second ``doctor`` spawn is the regression), and the payload keys/order are the Desktop/CLI contract.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from tools.computer_use import cua_backend_driver, permissions

pytestmark = pytest.mark.platforms("linux", "macos")

CHECK = {"label": "Driver", "status": "pass", "message": "Driver is healthy"}
TCC_OK = {"accessibility": True, "screen_recording": True, "screen_recording_capturable": True}
# The payload shape and this order are an API contract (Desktop card + `hermes computer-use permissions --json`).
CONTRACT_KEYS = ["platform", "platform_supported", "installed", "version", "ready", "can_grant", "checks", "source",
                 "error", "accessibility", "screen_recording", "screen_recording_capturable"]


def _write_driver(path: Path, calls: Path, *, doctor_ok: bool = True, label: str = "9.9.9") -> Path:
    """A fake cua-driver that logs each invocation; *calls* is baked in so the child env cannot drop it."""
    doctor_out = json.dumps({"ok": True, "probes": [CHECK]}) if doctor_ok else ""
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{calls}'\n"
        'case "$1" in\n'
        f'  --version) printf \'%s\\n\' \'cua-driver {label} (fake)\' ;;\n'
        f"  doctor) printf '%s\\n' '{doctor_out}' ;;\n"
        f"  permissions) printf '%s\\n' '{json.dumps(TCC_OK)}' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _fake_driver(tmp_path: Path, **kw) -> tuple[Path, Path]:
    calls = tmp_path / "calls.log"
    return _write_driver(tmp_path / "cua-driver", calls, **kw), calls


def _invocations(calls: Path, verb: str) -> int:
    if not calls.exists():
        return 0
    return sum(1 for line in calls.read_text(encoding="utf-8").splitlines() if line.split(" ")[0] == verb)


def test_doctor_runs_once_within_ttl_and_again_after(tmp_path, monkeypatch):
    driver, calls = _fake_driver(tmp_path)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    clock = {"now": 1000.0}  # only permissions' view of time moves; subprocess keeps the real clock
    monkeypatch.setattr(permissions, "time", types.SimpleNamespace(monotonic=lambda: clock["now"]))

    first = permissions.computer_use_status()
    second = permissions.computer_use_status()

    assert _invocations(calls, "doctor") == 1
    assert second == first
    assert first["installed"] is True and first["ready"] is True
    assert first["checks"] == [CHECK]

    clock["now"] += permissions._STATUS_CACHE_TTL + 1
    permissions.computer_use_status()
    assert _invocations(calls, "doctor") == 2


def test_payload_keeps_contract_keys_in_order(tmp_path, monkeypatch):
    driver, _ = _fake_driver(tmp_path)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))

    payload = permissions.computer_use_status()

    assert list(payload) == CONTRACT_KEYS
    assert payload["version"] == "cua-driver 9.9.9 (fake)"


def test_missing_driver_still_returns_the_full_payload(monkeypatch):
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda *a, **kw: None)

    payload = permissions.computer_use_status()

    assert list(payload) == CONTRACT_KEYS
    assert payload["installed"] is False
    assert payload["version"] is None
    assert payload["ready"] is None
    assert payload["checks"] == []


def test_failed_doctor_is_not_cached(tmp_path, monkeypatch):
    """A dead/timing-out driver must never sit behind the TTL as a blank-but-fresh answer."""
    driver, calls = _fake_driver(tmp_path, doctor_ok=False)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))

    for _ in range(2):
        payload = permissions.computer_use_status()
        assert payload["ready"] is None
        assert payload["checks"] == []

    assert _invocations(calls, "doctor") == 2


def test_permissions_grant_invalidates_the_cache(tmp_path, monkeypatch):
    driver, calls = _fake_driver(tmp_path)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    monkeypatch.setattr(permissions.sys, "platform", "darwin")

    assert permissions.computer_use_status()["ready"] is True
    assert _invocations(calls, "doctor") == 1

    assert permissions.request_permissions_grant() == 0  # rewrote the TCC rows: the cached readiness is stale
    permissions.computer_use_status()
    assert _invocations(calls, "doctor") == 2


def test_switching_backend_does_not_reuse_the_cua_entry(tmp_path, monkeypatch):
    import plugins.computer_use as cu_plugins

    driver, calls = _fake_driver(tmp_path)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    assert permissions.computer_use_status()["ready"] is True
    assert _invocations(calls, "doctor") == 1

    class _Other:
        name = "other"
        display_name = "Other Driver"
        is_available = staticmethod(lambda: True)

    monkeypatch.setattr(cu_plugins, "configured_backend_name", lambda: "other")
    monkeypatch.setattr(cu_plugins, "get_active_provider", lambda: _Other())
    payload = permissions.computer_use_status()

    assert payload["version"] == "Other Driver (computer_use.backend: other)"
    assert payload["ready"] is True
    assert payload["checks"] == []  # never the cua driver's checks
    assert _invocations(calls, "doctor") == 1


def test_another_binary_gets_its_own_entry(tmp_path, monkeypatch):
    first, first_calls = _fake_driver(tmp_path)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    second, second_calls = _fake_driver(other_dir, label="1.0.0-other")
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(first))

    assert permissions.computer_use_status()["version"] == "cua-driver 9.9.9 (fake)"
    payload = permissions.computer_use_status(driver_cmd=str(second))

    assert payload["version"] == "cua-driver 1.0.0-other (fake)"  # a shared entry would report the wrong driver
    assert _invocations(first_calls, "doctor") == 1
    assert _invocations(second_calls, "doctor") == 1


def test_driver_upgrade_retires_the_entry(tmp_path, monkeypatch):
    driver, calls = _fake_driver(tmp_path)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(driver))
    assert permissions.computer_use_status()["version"] == "cua-driver 9.9.9 (fake)"

    _write_driver(driver, calls, label="10.0.0")
    os.utime(driver, None)  # mtime/size are part of the key, so an in-place upgrade re-probes

    assert permissions.computer_use_status()["version"] == "cua-driver 10.0.0 (fake)"
    assert _invocations(calls, "doctor") == 2
