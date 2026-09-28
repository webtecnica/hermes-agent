import pytest
from agent.auxiliary_client import _effective_aux_timeout, _get_task_timeout
from agent.conversation_compression import resolve_context_compression_timeouts


def test_effective_aux_timeout_compression_default_floor(monkeypatch):
    # When unset, compression timeout defaults to 300s floor
    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
    assert _effective_aux_timeout("compression", None) == 300.0
    assert _get_task_timeout("compression") == 300.0


def test_effective_aux_timeout_compression_honours_explicit_config_below_300(monkeypatch):
    # When explicitly configured below 300s, config must be honoured
    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {"timeout": 120.0})
    assert _effective_aux_timeout("compression", None) == 120.0
    assert _get_task_timeout("compression") == 120.0

    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {"timeout": 45})
    assert _effective_aux_timeout("compression", None) == 45.0
    assert _get_task_timeout("compression") == 45.0


def test_effective_aux_timeout_per_call_override_wins():
    assert _effective_aux_timeout("compression", 15.0) == 15.0
    assert _effective_aux_timeout("compression", 600.0) == 600.0


def test_resolve_context_compression_timeouts_honours_explicit_aux_timeout(monkeypatch):
    # With explicit auxiliary.compression.timeout: 120, host ceiling is not forced up to 300s
    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {"timeout": 120.0})

    cfg = {
        "context_timeout_seconds": 60,
        "context_total_ceiling_seconds": 120,
    }
    idle, ceiling = resolve_context_compression_timeouts(cfg)
    assert idle == 120.0
    assert ceiling == 120.0
