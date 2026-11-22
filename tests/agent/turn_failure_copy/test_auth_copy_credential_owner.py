"""Auth-failure copy names the login that owns the credential (#158828).

A bare Anthropic 401 is not necessarily an API-key problem: when the pool's active
row is the Claude Code login Hermes borrowed from ``~/.claude``, the fix is that
CLI's own ``claude auth login`` — not ``hermes setup`` and not ``hermes auth add``
(a Hermes-store sign-in that would not touch the borrowed credential).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.credential_pool import PooledCredential
from agent.error_surface import active_credential, auth_kind
from agent.turn_failure_copy import nonretryable_copy, oauth_relogin_command, relogin_command_hint


class _Pool:
    """The slice of ``CredentialPool`` the copy path reads: the row in use."""

    def __init__(self, entry=None, provider=None):
        self._entry = entry
        self.provider = provider

    def current(self):
        return self._entry


def _entry(**overrides):
    payload = {"source": "claude_code", "auth_type": "oauth", "access_token": "***",
               "priority": 0, "label": "Claude Code"}
    payload.update(overrides)
    return PooledCredential.from_dict("anthropic", payload)


CLAUDE_CODE_POOL = _Pool(_entry())
API_KEY_POOL = _Pool(_entry(source="manual", auth_type="api_key", label="manual"))
EMPTY_POOL = _Pool()


def _auth_classified():
    return SimpleNamespace(is_auth=bool(1))


@pytest.fixture(autouse=True)
def _profile_selector(monkeypatch):
    # A named profile, so a leaked ``-p <profile>`` in a vendor CLI command would show.
    monkeypatch.setattr("hermes_constants.profile_cli_selector", lambda: "-p demo ")


def test_active_credential_reads_the_pools_current_row():
    assert active_credential("anthropic", CLAUDE_CODE_POOL) == ("claude_code", "oauth")
    assert active_credential("anthropic", API_KEY_POOL) == ("manual", "api_key")
    assert active_credential("anthropic", EMPTY_POOL) is None
    assert active_credential("", CLAUDE_CODE_POOL) is None


def test_borrowed_oauth_row_makes_anthropic_an_oauth_provider():
    assert auth_kind("anthropic", CLAUDE_CODE_POOL) == "oauth"
    assert auth_kind("anthropic", API_KEY_POOL) == "api_key"
    # No pool at all keeps the catalog verdict.
    assert auth_kind("anthropic") == "api_key"


def test_borrowed_oauth_row_relogin_is_the_cli_that_owns_it():
    assert oauth_relogin_command("anthropic", CLAUDE_CODE_POOL) == "claude auth login"
    assert relogin_command_hint("anthropic", CLAUDE_CODE_POOL) == "claude auth login"


def test_borrowed_oauth_401_copy_sends_the_user_to_the_owning_login():
    copy = nonretryable_copy(
        _auth_classified(), provider="anthropic", model="claude-sonnet-4-5",
        summary="401 invalid bearer token", pool=CLAUDE_CODE_POOL,
    )
    assert "rejected your sign-in" in copy
    assert "`claude auth login`" in copy
    assert "hermes setup" not in copy
    assert "hermes auth add" not in copy


def test_api_key_credential_keeps_todays_copy():
    assert relogin_command_hint("anthropic", API_KEY_POOL) == "hermes -p demo auth add anthropic"
    copy = nonretryable_copy(
        _auth_classified(), provider="anthropic", model="claude-sonnet-4-5",
        summary="401 invalid x-api-key", pool=API_KEY_POOL,
    )
    assert "rejected your API key" in copy
    assert "hermes setup" in copy
    assert "claude auth login" not in copy


def test_empty_or_unreadable_pool_falls_back_to_the_catalog_rule(monkeypatch):
    assert auth_kind("anthropic", EMPTY_POOL) == "api_key"
    assert nonretryable_copy(
        _auth_classified(), provider="anthropic", model="claude-sonnet-4-5",
        summary="401", pool=EMPTY_POOL,
    ).count("rejected your API key") == 1

    def _broken(_provider=None):
        raise RuntimeError("auth store unreadable")

    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", _broken)
    assert auth_kind("anthropic") == "api_key"
    assert active_credential("anthropic") is None


def test_without_a_pool_the_persisted_pool_decides(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.read_credential_pool",
        lambda provider=None: [{"source": "claude_code", "auth_type": "oauth",
                                "access_token": "***", "priority": 0, "label": "Claude Code"}],
    )
    assert active_credential("anthropic") == ("claude_code", "oauth")
    assert auth_kind("anthropic") == "oauth"
    assert relogin_command_hint("anthropic") == "claude auth login"


def test_persisted_pool_prefers_the_highest_priority_row(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.read_credential_pool",
        lambda provider=None: [
            {"source": "claude_code", "auth_type": "oauth", "access_token": "***",
             "priority": 1, "label": "Claude Code"},
            {"source": "manual", "auth_type": "api_key", "access_token": "***",
             "priority": 0, "label": "manual"},
        ],
    )
    assert active_credential("anthropic") == ("manual", "api_key")
    assert auth_kind("anthropic") == "api_key"


def test_api_key_row_never_downgrades_a_catalog_oauth_provider():
    """A pool row that is an API key must not flip a catalog OAuth provider (Nous) to api_key."""
    assert auth_kind("nous", API_KEY_POOL) == "oauth"
    assert nonretryable_copy(
        _auth_classified(), provider="nous", model="Hermes-4-405B",
        summary="401", pool=API_KEY_POOL,
    ).count("rejected your sign-in") == 1


def test_pool_for_another_provider_is_ignored():
    """After a fallback switch the agent's live pool can belong to a different provider."""
    other = _Pool(_entry(), provider="openrouter")
    assert active_credential("anthropic", other) is None
    assert auth_kind("anthropic", other) == "api_key"
    assert active_credential("anthropic", _Pool(_entry(), provider="anthropic")) == ("claude_code", "oauth")


def test_other_sources_keep_the_profile_scoped_hermes_command():
    oauth_manual = _Pool(_entry(source="hermes_pkce", auth_type="oauth", label="Hermes OAuth"))
    assert auth_kind("anthropic", oauth_manual) == "oauth"
    assert oauth_relogin_command("anthropic", oauth_manual) == "hermes -p demo auth add anthropic --type oauth"
    assert oauth_relogin_command("nous", CLAUDE_CODE_POOL) == "hermes -p demo portal"


def _guidance_lines(pool, provider="anthropic"):
    from agent.turn_recovery import _print_nonretryable_auth_guidance

    lines: list[str] = []
    agent = SimpleNamespace(
        log_prefix="", _credential_pool=pool,
        _vprint=lambda text, **_kwargs: lines.append(text),
    )
    _print_nonretryable_auth_guidance(
        agent, SimpleNamespace(is_auth=bool(1), reason=None, error_context={}, billing_unverified=False),
        status_code=401, provider=provider, base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
    )
    return lines


def test_cli_auth_guidance_names_the_owning_login():
    assert _guidance_lines(CLAUDE_CODE_POOL) == [
        "   💡 anthropic sign-in was rejected. Sign in again: `claude auth login`",
    ]


def test_cli_auth_guidance_keeps_the_api_key_lines_for_a_key():
    lines = _guidance_lines(API_KEY_POOL)
    assert any("Your API key was rejected" in line for line in lines)
    assert any("hermes setup" in line for line in lines)


def test_catalog_oauth_provider_with_a_key_row_keeps_the_api_key_guidance():
    """Only a borrowed external login changes the CLI lines; a key row never does."""
    from agent.turn_failure_copy import external_source_login

    assert external_source_login("anthropic", API_KEY_POOL) is None
    lines = _guidance_lines(API_KEY_POOL, provider="minimax-oauth")
    assert any("Your API key was rejected" in line for line in lines)
