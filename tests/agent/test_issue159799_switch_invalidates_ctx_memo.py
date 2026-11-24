"""#159799: a successful /model switch drops the context-length memo for both ends of the switch.

The picker (``hermes_cli.model_switch.resolve_display_context_length``) probes live, but the
request-time validator reads ``agent.model_metadata.get_model_context_length``, which serves the
in-process local-probe memo (``_LOCAL_CTX_PROBE_CACHE``, 30 s) and the endpoint-metadata memo
(300 s). Both outlived a switch, so an operator restarting a local vLLM/llama.cpp/Ollama server
with a larger window under the same model id kept being validated against the window probed
BEFORE the restart: a stale 32,768 made the live 65,536 fail the "below minimum 64,000" check.

``hermes_cli.model_switch.switch_model`` owns the applied selection for every frontend (CLI,
gateway, TUI, ACP), so it is the callable that invalidates the outgoing and the incoming
``(model, base_url)`` memos. Nothing here opens a socket: the live probe is stubbed.
"""

import time

import pytest

import agent.model_metadata as mm
from hermes_cli.model_switch import switch_model

BASE_URL = "http://127.0.0.1:8080/v1"
STALE_WINDOW = 32_768
LIVE_WINDOWS = {"model-x": 65_536, "model-y": 131_072}


@pytest.fixture(autouse=True)
def _restore_probe_memo():
    """The probe memo is process-global; keep this file from leaking into its neighbours."""
    before = dict(mm._LOCAL_CTX_PROBE_CACHE)
    yield
    mm._LOCAL_CTX_PROBE_CACHE.clear()
    mm._LOCAL_CTX_PROBE_CACHE.update(before)


@pytest.fixture
def live_probe(monkeypatch):
    """Seed the memo with the pre-restart window and make the live probe report the new one."""
    probes: list[tuple[str, str]] = []

    def _probe(model, base_url, api_key=""):
        probes.append((model, base_url))
        return LIVE_WINDOWS.get(model)

    monkeypatch.setattr(mm, "_query_local_context_length_uncached", _probe)
    # The other two ways a custom endpoint answers for a window: both miss in this scenario.
    monkeypatch.setattr(mm, "_resolve_endpoint_context_length", lambda *a, **k: None)
    monkeypatch.setattr(mm, "_query_ollama_api_show", lambda *a, **k: None)
    _seed_memo("model-x", STALE_WINDOW)
    return probes


def _seed_memo(model: str, window: int) -> None:
    """Memoize *window* exactly like a probe that ran before the server restart."""
    mm._LOCAL_CTX_PROBE_CACHE[(mm._strip_provider_prefix(model), BASE_URL)] = (window, time.monotonic())


def _stub_switch_metadata(monkeypatch) -> None:
    """The non-routing half of a switch: provider credential resolution and models.dev metadata."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {"provider": "custom", "api_key": "", "base_url": BASE_URL,
                          "api_mode": "chat_completions"})
    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": ""})
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_picker_usage.large_model_notice", lambda *a, **k: "")


def test_switch_reevaluates_the_window_of_a_restarted_server(monkeypatch, live_probe):
    """Same model id, same endpoint, new window: the switch must drop the stale 32,768."""
    _stub_switch_metadata(monkeypatch)
    # The request-time read before the switch serves the stale memo — the reported failure.
    assert mm.get_model_context_length("model-x", base_url=BASE_URL, provider="custom") == STALE_WINDOW

    result = switch_model("model-x", "custom", "model-x", BASE_URL, "")

    assert result.success, result.error_message
    assert (result.new_model, result.base_url) == ("model-x", BASE_URL)
    assert mm.get_model_context_length("model-x", base_url=BASE_URL, provider="custom") == LIVE_WINDOWS["model-x"]


def test_switch_to_another_model_resolves_its_own_window(monkeypatch, live_probe):
    """A different model must never inherit the outgoing model's window."""
    _stub_switch_metadata(monkeypatch)

    result = switch_model("model-y", "custom", "model-x", BASE_URL, "")

    assert result.success, result.error_message
    assert mm.get_model_context_length("model-y", base_url=BASE_URL, provider="custom") == LIVE_WINDOWS["model-y"]
    assert ("model-y", BASE_URL) in live_probe


def test_plain_reads_still_hit_the_memo(monkeypatch, live_probe):
    """No switch, no probe: the cache keeps doing its job (invalidation must not move into reads)."""
    for _ in range(3):
        assert mm.get_model_context_length("model-x", base_url=BASE_URL, provider="custom") == STALE_WINDOW
    assert live_probe == []
