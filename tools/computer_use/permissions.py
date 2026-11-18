"""Cross-platform Computer Use readiness + macOS permission helpers. "Ready to drive" differs per platform: macOS
needs explicit TCC grants (Accessibility + Screen Recording) via cua-driver ``permissions status`` / ``permissions
grant``; Windows/Linux have no TCC toggles, so readiness == driver health. The grants attach to cua-driver's OWN
identity (``com.trycua.driver``), not Hermes, so ``grant`` launches CuaDriver via LaunchServices for correct dialog
attribution. ``cua-driver doctor --json`` is the universal signal; ``computer_use_status`` folds it with the macOS
detail into one payload for the desktop card, the ``permissions`` CLI and ``/api/tools/computer-use/status``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import suppress
from typing import Any, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

_RUNTIME_PLATFORMS = frozenset({"darwin", "win32", "linux"})  # mirrors the toolset platform_gate
_BOOLS = ("accessibility", "screen_recording", "screen_recording_capturable")
CUA_DRIVER_BUNDLE_ID = "com.trycua.driver"  # the only identity TCC rows and the private daemon are keyed to
_TCC_SERVICES = {"accessibility": "Accessibility", "screen_recording": "ScreenCapture"}  # status field -> tccutil service
TCC_FIELDS = tuple(_TCC_SERVICES)

# ``cua-driver doctor --json`` costs ~1.8 s on Windows (#157424) and the desktop Capabilities tab re-reads the
# status every time it appears, so the probe half of the payload is memoized for a few seconds. Only a HEALTHY
# read is cached: a timeout or a failed doctor must never sit behind the TTL as a stale/blank answer.
_STATUS_CACHE_TTL = 30.0
_status_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}
_status_cache_lock = threading.Lock()

def stale_tcc_grant_hint(*missing: str) -> str:
    """Recovery text for grants the daemon reports missing while System Settings shows the CuaDriver toggle ON.

    TCC keys each row to the app's code-signing requirement, so a row written for an earlier CuaDriver build can
    stop matching after a driver update: the toggle stays ON, the daemon is denied, and flipping the toggle does
    not rewrite the requirement (trycua/cua#3170). Only a reset + re-grant recovers it, so the hint names the
    exact rows for the *missing* status fields.
    """
    resets = " && ".join(f"tccutil reset {_TCC_SERVICES[f]} {CUA_DRIVER_BUNDLE_ID}" for f in missing if f in _TCC_SERVICES)
    if not resets:
        return ""
    return ("If System Settings already shows CuaDriver ON, the stored grant is stale (it no longer matches the installed "
            f"driver's signature): run `{resets}`, then `hermes computer-use permissions grant`.")

def _child_env() -> dict[str, str]:
    """cua-driver child env (telemetry policy + provider secrets stripped); ``os.environ`` on import error.

    cua-driver is a third-party binary — it must never inherit provider API keys (#53503/#55709/#58889
    lineage). Each layer degrades gracefully so permission probes never break on a helper import error.
    """
    try:
        from tools.computer_use.cua_backend import sanitized_cua_driver_env
        return sanitized_cua_driver_env()
    except Exception:
        return dict(os.environ)

def _run(binary: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run([binary, *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout, env=_child_env(), stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(), check=False)

def _json_out(binary: str, *args: str, timeout: float) -> Any:
    """Run ``binary args`` and parse stdout as JSON (``None`` on empty output)."""
    raw = (_run(binary, *args, timeout=timeout).stdout or "").strip()
    return json.loads(raw) if raw else None

def _doctor(binary: str) -> Optional[dict[str, Any]]:
    """``cua-driver doctor --json`` → ``{ok, checks:[{label,status,message}]}`` (None on any failure)."""
    try:
        data = _json_out(binary, "doctor", "--json", timeout=12)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None
    checks = [{k: str(p.get(k, "")) for k in ("label", "status", "message")} for p in data.get("probes", []) if isinstance(p, dict)]
    return {"ok": bool(data.get("ok")), "checks": checks}

def _mac_permissions(binary: str, out: dict[str, Any]) -> None:
    """Fold ``cua-driver permissions status --json`` booleans (+ ``source``) into ``out``."""
    try:
        data = _json_out(binary, "permissions", "status", "--json", timeout=10)
    except subprocess.TimeoutExpired:
        out["error"] = "cua-driver permissions status timed out"
    except Exception as exc:  # spawn failure or malformed JSON
        out["error"] = f"cua-driver permissions status failed: {exc}"
    else:
        if isinstance(data, dict):
            out.update({k: data[k] for k in _BOOLS if isinstance(data.get(k), bool)})
            if isinstance(data.get("source"), dict):
                out["source"] = data["source"]

def _status_cache_key(backend: str, binary: str, plat: str) -> tuple:
    """``(backend, binary, platform, binary stat)`` — the inputs the probe half of the payload derives from.

    The profile is not keyed separately: ``computer_use.backend`` and the resolved binary are the only
    profile-scoped inputs folded into the payload, and two profiles that share both share the payload (they
    differ elsewhere in the dict only through values recomputed on every call). ``mtime_ns``/``size`` change on
    every driver install or upgrade, which retires the entry without a hook inside the installer.
    """
    stamp = None
    with suppress(OSError):
        info = os.stat(binary)
        stamp = (info.st_mtime_ns, info.st_size)
    return (backend, binary, plat, stamp)


def invalidate_status_cache() -> None:
    """Drop every cached probe: a permission grant, a driver upgrade or a backend switch can change the answer."""
    with _status_cache_lock:
        _status_cache.clear()


def _probe(binary: str, plat: str) -> dict[str, Any]:
    """The expensive half of the payload: ``--version``, ``doctor`` and (macOS) the TCC rows."""
    version = None
    with suppress(Exception):
        version = (_run(binary, "--version", timeout=5).stdout or "").strip() or None
    doctor = _doctor(binary)
    tcc: dict[str, Any] = {"source": None, "error": None, **{k: None for k in _BOOLS}}
    if plat == "darwin":
        _mac_permissions(binary, tcc)
    return {"version": version, "doctor": doctor, "source": tcc["source"], "error": tcc["error"],
            "bools": {k: tcc[k] for k in _BOOLS}}


def _probe_cached(backend: str, binary: str, plat: str) -> dict[str, Any]:
    """``_probe`` behind a short TTL, under a lock (the gateway serves requests on threads).

    Failures bypass the cache entirely: ``doctor`` returning nothing, or the macOS TCC probe erroring, is
    reported as-is on every call so a broken driver cannot read as healthy for the whole TTL. Two requests that
    miss at the same moment may both probe — the lock never covers a subprocess, so one slow probe cannot block
    another profile's status read.
    """
    key = _status_cache_key(backend, binary, plat)
    with _status_cache_lock:
        hit = _status_cache.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return hit[1]
    probes = _probe(binary, plat)
    if probes["doctor"] is None or probes["error"] is not None:
        with _status_cache_lock:
            _status_cache.pop(key, None)
        return probes
    with _status_cache_lock:
        _status_cache[key] = (time.monotonic() + _STATUS_CACHE_TTL, probes)
    return probes


def computer_use_status(driver_cmd: Optional[str] = None) -> dict[str, Any]:
    """OS-aware readiness for the desktop card; key order is an API payload contract. ``ready`` is the single signal the
    UI keys off: macOS = both TCC grants, elsewhere = driver health (no TCC model); ``None`` = unknown (binary missing /
    probe failed). ``can_grant`` is macOS-only. The driver probes behind the cua path are memoized for
    ``_STATUS_CACHE_TTL`` seconds (see ``_probe_cached``); the dict itself is rebuilt on every call."""
    from plugins.computer_use import DEFAULT_BACKEND, configured_backend_name, get_active_provider
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd  # same resolver as the tool itself
    plat, name = sys.platform, configured_backend_name()
    binary = resolve_cua_driver_cmd(driver_cmd) if name == DEFAULT_BACKEND else None
    out: dict[str, Any] = {"platform": plat, "platform_supported": plat in _RUNTIME_PLATFORMS,
                           "installed": bool(binary), "version": None, "ready": None, "can_grant": plat == "darwin",
                           "checks": [], "source": None, "error": None, **{k: None for k in _BOOLS}}
    if name != DEFAULT_BACKEND:  # another driver is selected: cua-driver probes and TCC grants do not apply
        out.update(platform_supported=True, can_grant=False)
        try:
            provider = get_active_provider()
        except LookupError as e:
            out["error"] = str(e)
            return out
        out.update(installed=True, version=f"{provider.display_name} (computer_use.backend: {provider.name})",
                   ready=provider.is_available())
        return out
    if not binary:
        return out
    probes = _probe_cached(name, binary, plat)
    out["version"] = probes["version"]
    out["source"] = probes["source"]
    out["error"] = probes["error"]
    out.update({k: v for k, v in probes["bools"].items() if v is not None})
    if probes["doctor"] is not None:
        out["checks"] = [dict(check) for check in probes["doctor"]["checks"]]  # copies: callers never share cache state
    if plat == "darwin":
        if out["error"] is None:
            out["ready"] = out["accessibility"] is True and out["screen_recording"] is True
    elif probes["doctor"] is not None:
        out["ready"] = probes["doctor"]["ok"]  # no TCC model off macOS
    return out

def request_permissions_grant(driver_cmd: Optional[str] = None) -> int:
    """Run ``cua-driver permissions grant`` (macOS), streaming its output. Returns the driver's exit code (0 ok), 2 if
    the binary is missing, 64 on a non-macOS platform (no TCC model to grant)."""
    if sys.platform != "darwin":
        print("Computer Use permissions are a macOS concept; nothing to grant here.")
        return 64
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd
    binary = resolve_cua_driver_cmd(driver_cmd)
    if not binary:
        print("cua-driver: not installed. Run: hermes computer-use install")
        return 2
    print("Requesting Accessibility + Screen Recording for CuaDriver.\n"
          f"macOS will show a dialog attributed to CuaDriver ({CUA_DRIVER_BUNDLE_ID}) — approve it, then return here.")
    try:
        return int(subprocess.run([binary, "permissions", "grant"], env=_child_env(), stdin=subprocess.DEVNULL, check=False).returncode)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except Exception as exc:  # pragma: no cover - defensive
        print(f"cua-driver permissions grant failed: {exc}", file=sys.stderr)
        return 2
    finally:
        # A grant attempt rewrites the TCC rows (or leaves them denied) either way: no cached probe may survive it.
        invalidate_status_cache()
