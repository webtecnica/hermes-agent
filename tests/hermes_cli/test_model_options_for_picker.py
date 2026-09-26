import pytest
from unittest.mock import MagicMock, patch
from hermes_cli.inventory import ConfigContext, build_model_options_payload
from hermes_cli.cli_model_switch_mixin import _show_model_picker


def _make_ctx(provider="", model="", base_url=""):
    return ConfigContext(
        current_provider=provider,
        current_model=model,
        current_base_url=base_url,
        user_providers={},
        custom_providers=[],
    )


def test_build_model_options_payload_defaults_to_for_picker_true():
    ctx = _make_ctx()
    with patch("hermes_cli.inventory.build_models_payload") as mock_build:
        mock_build.return_value = {"providers": []}
        build_model_options_payload(ctx)
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("for_picker") is True


def test_build_model_options_payload_forwards_for_picker_flag():
    ctx = _make_ctx()
    with patch("hermes_cli.inventory.build_models_payload") as mock_build:
        mock_build.return_value = {"providers": []}
        build_model_options_payload(ctx, for_picker=False)
        _, kwargs = mock_build.call_args
        assert kwargs.get("for_picker") is False


def test_build_model_options_payload_includes_oauth_subscription_provider(monkeypatch, tmp_path):
    """When an OAuth-subscription provider (openai-codex) has credentials in pool,
    build_model_options_payload must include it because for_picker defaults to True."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **kw: {})
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *a, **kw: {})

    fake_pool = MagicMock()
    fake_pool.has_credentials.return_value = True
    fake_pool.has_available.return_value = False  # in cooldown or flat subscription check

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda slug: fake_pool)
    monkeypatch.setattr("hermes_cli.model_switch_providers._credential_pool_is_usable", lambda slug, **kw: False)
    monkeypatch.setattr("hermes_cli.model_switch_providers._auth_store_has_provider", lambda *keys: False)
    monkeypatch.setattr("hermes_cli.model_switch_providers._live_or_curated_ids", lambda *a, **kw: ["gpt-5.5"])

    ctx = _make_ctx()

    payload_picker = build_model_options_payload(ctx)
    codex_picker = [p for p in payload_picker["providers"] if p.get("slug") == "openai-codex"]
    assert len(codex_picker) == 1, "openai-codex must be included in desktop/dashboard model options payload"

    payload_no_picker = build_model_options_payload(ctx, for_picker=False)
    codex_no_picker = [p for p in payload_no_picker["providers"] if p.get("slug") == "openai-codex"]
    assert len(codex_no_picker) == 0, "openai-codex must be omitted when for_picker=False"


def test_cli_show_model_picker_passes_for_picker_true():
    ctx = _make_ctx()
    cli = MagicMock()
    with patch("hermes_cli.inventory.build_models_payload") as mock_build:
        mock_build.return_value = {"providers": [{"slug": "openai-codex", "name": "Codex", "models": ["gpt-5.5"]}]}
        _show_model_picker(cli, ctx, force_refresh=False)
        mock_build.assert_called_once()
        _, kwargs = mock_build.call_args
        assert kwargs.get("for_picker") is True
