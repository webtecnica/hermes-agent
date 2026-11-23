"""Non-finite floats in a handler result must never reach the wire as bare ``Infinity``/``NaN``.

``json.dumps`` emits ``inf``/``-inf``/``nan`` as those bare tokens by default, and none of them is
valid JSON: a strict client (iOS ``JSONDecoder``, Java, Swift) rejects the whole frame, drops the
socket and reconnects forever. A ``profile.yaml`` ``ui_meta`` value written as an unquoted YAML 1.1
mantissa (``20260819_045052_92e993``) overflows to ``inf`` on load, which is how this reached
``profiles.list`` (#159335).

The contract pinned here: ``null`` on the wire, at both boundaries — the profile.yaml read and the
frame encoder every transport serializes through. The real iOS / Java app's reconnect loop is NOT
measured here (we don't have the app): the proof is that the payload a strict JSON parser accepts,
which is what its decoder does.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path

import pytest

import hermes_yaml
import tui_gateway.server as srv
from tui_gateway import profile_roster_cache as cache
from tui_gateway.transport import serialize_frame
from tui_gateway.ws import WSTransport

_LOG = logging.getLogger("test-nonfinite")

# The unquoted token from the issue: YAML 1.1 resolves it as a float, the mantissa overflows to inf.
MANTISSA = "20260819_045052_92e993"


def _reject_constant(value):
    raise AssertionError(f"non-finite JSON constant on the wire: {value}")


def _strict_loads(payload: str):
    """``json.loads`` the way a strict client does: any ``Infinity``/``NaN`` token fails the parse."""
    return json.loads(payload, parse_constant=_reject_constant)


@pytest.fixture(autouse=True)
def _clear_memo():
    cache.invalidate()
    yield
    cache.invalidate()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    bob = tmp_path / "profiles" / "bob"
    bob.mkdir(parents=True)
    (bob / "config.yaml").write_text("model:\n  provider: openai\n", encoding="utf-8")
    (bob / "profile.yaml").write_text(
        "display_name: Bob\n"
        "ui_meta:\n"
        "  hermes-bots:\n"
        "    title: Bob\n"
        f"    created: {MANTISSA}\n"
        "    floor: -.inf\n"
        "    ratio: .nan\n"
        "_ui_meta_revisions:\n"
        "  hermes-bots: 1\n",
        encoding="utf-8")
    return tmp_path


def _row(**params):
    envelope = srv._methods["profiles.list"](1, {"include_sessions": False, **params})
    return envelope, next(p for p in envelope["result"]["profiles"] if p["name"] == "bob")


def test_the_yaml_mantissa_really_loads_as_a_non_finite_float(home):
    """Precondition for the regression: without it this test would pass for the wrong reason."""
    raw = hermes_yaml.safe_load((home / "profiles" / "bob" / "profile.yaml").read_text(encoding="utf-8"))
    bots = raw["ui_meta"]["hermes-bots"]
    assert bots["created"] == math.inf
    assert bots["floor"] == -math.inf
    assert math.isnan(bots["ratio"])


@pytest.mark.filterwarnings("ignore::ruamel.yaml.error.MantissaNoDotYAML1_1Warning")
def test_profiles_list_emits_null_for_every_non_finite(home):
    """The listing payload is strict JSON, and the user's other keys survive untouched."""
    envelope, row = _row()

    assert row["ui_meta"] == {"hermes-bots": {"title": "Bob", "created": None,
                                              "floor": None, "ratio": None}}
    # Wire contract: ui_meta_revisions precedes ui_meta on the row.
    keys = list(row)
    assert keys.index("ui_meta_revisions") < keys.index("ui_meta")

    payload = serialize_frame(envelope, "test", _LOG)
    assert "Infinity" not in payload and "NaN" not in payload
    decoded = _strict_loads(payload)["result"]["profiles"]
    bob = next(p for p in decoded if p["name"] == "bob")
    assert bob["ui_meta"]["hermes-bots"]["created"] is None
    assert bob["ui_meta_revisions"] == {"hermes-bots": 1}


def test_serialize_frame_scrubs_non_finite_and_keeps_everything_else():
    """The frame encoder is the last line of defence: any handler result may carry a client float."""
    frame = {"jsonrpc": "2.0", "id": 9, "result": {
        "nested": {"up": math.inf, "down": -math.inf, "none": math.nan},
        "items": [math.inf, 1.5, -2, "1e999", True, None],
    }}
    payload = serialize_frame(frame, "test", _LOG)
    decoded = _strict_loads(payload)["result"]
    assert decoded["nested"] == {"up": None, "down": None, "none": None}
    assert decoded["items"] == [None, 1.5, -2, "1e999", True, None]
    # Scrubbing copies: the caller's payload is never mutated under a cache that shares it.
    assert frame["result"]["nested"]["up"] == math.inf


def test_a_finite_payload_is_byte_identical():
    """No non-finite float means no rewrite — this rides every roster paint."""
    frame = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True, "n": 1.5, "s": "é"}}
    assert serialize_frame(frame, "test", _LOG) == json.dumps(frame, ensure_ascii=False)


def test_ws_transport_sends_a_strict_json_frame():
    """The WS emission path (``ws.write_async`` -> ``serialize_frame``) puts valid JSON on the socket."""
    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, line: str) -> None:
            self.sent.append(line)

    async def _run() -> None:
        ws = _FakeWS()
        transport = WSTransport(ws, asyncio.get_running_loop(), peer="127.0.0.1:1")
        frame = {"jsonrpc": "2.0", "id": 7,
                 "result": {"v": [math.inf, -math.inf, math.nan]}}
        assert await transport.write_async(frame) is True
        [payload] = ws.sent
        assert _strict_loads(payload)["result"]["v"] == [None, None, None]

    asyncio.run(_run())
