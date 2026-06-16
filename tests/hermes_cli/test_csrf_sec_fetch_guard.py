"""Sec-Fetch-Site CSRF guard on mutating /api/* routes.

The guard replaces the legacy ``_SESSION_TOKEN``'s only robust
contribution — blocking drive-by CSRF from a web page the user visits —
with a credential-free, browser-asserted check that applies in BOTH auth
regimes. ``Sec-Fetch-Site`` is a forbidden header name (JS cannot forge
it), so a cross-origin page cannot spoof ``same-origin``.

Scope decision (plan Q2): mutating methods only. Reads are already
neutralised by the CORSMiddleware (localhost-only origin regex,
allow_credentials off), which prevents a foreign origin from reading any
``/api/*`` response body.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.xdist_group("dashboard_auth_app_state")

from fastapi.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def loopback_client():
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


# A real state-changing route. The CSRF guard runs BEFORE auth, so the
# blocked cases 403 regardless of token; the allowed cases carry a valid
# token so a non-403 proves the guard let them through to auth+handler.
_MUTATING_ROUTE = "/api/providers/validate"


@pytest.mark.parametrize("sfs", ["cross-site", "same-site"])
def test_cross_origin_mutation_blocked(loopback_client, sfs):
    r = loopback_client.post(
        _MUTATING_ROUTE,
        headers={
            "X-Hermes-Session-Token": web_server._SESSION_TOKEN,
            "Sec-Fetch-Site": sfs,
        },
        json={"key": "OPENAI_API_KEY", "value": "x"},
    )
    assert r.status_code == 403
    assert r.json().get("error") == "cross_origin_blocked"


@pytest.mark.parametrize("sfs", ["same-origin", "none"])
def test_same_origin_mutation_allowed(loopback_client, sfs):
    r = loopback_client.post(
        _MUTATING_ROUTE,
        headers={
            "X-Hermes-Session-Token": web_server._SESSION_TOKEN,
            "Sec-Fetch-Site": sfs,
        },
        json={"key": "OPENAI_API_KEY", "value": "x"},
    )
    # Reaches the handler (any non-403): the CSRF guard let it through.
    assert r.status_code != 403


def test_absent_header_fails_open(loopback_client):
    """Non-browser clients (curl, NAS probe, desktop) send no
    Sec-Fetch-Site and must NOT be blocked."""
    r = loopback_client.post(
        _MUTATING_ROUTE,
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
        json={"key": "OPENAI_API_KEY", "value": "x"},
    )
    assert r.status_code != 403


def test_cross_site_get_not_blocked(loopback_client):
    """Reads are CORS-covered, not CSRF-guarded (mutations-only scope)."""
    r = loopback_client.get(
        "/api/status", headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert r.status_code == 200


def test_guard_applies_in_gated_mode():
    """The guard is mode-agnostic: a cross-site mutation from an
    AUTHENTICATED session is still blocked in gated mode by the CSRF guard.

    A cookieless gated request 401s at the cookie gate before the CSRF
    guard runs (Starlette runs last-registered-middleware outermost, so
    the auth gate is outer). To prove the CSRF guard actually fires in
    gated mode we must carry a valid session cookie so the request gets
    past the gate and reaches the guard, which then 403s the cross-site
    mutation.
    """
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    clear_providers()
    provider = StubAuthProvider()
    register_provider(provider)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = "fly-app.fly.dev"
    try:
        # Mint a real session via the stub's login round trip.
        start = provider.start_login(redirect_uri="https://fly-app.fly.dev/auth/callback")
        state = start.cookie_payload["hermes_session_pkce"].split("state=")[1].split(";")[0]
        verifier = start.cookie_payload["hermes_session_pkce"].split("verifier=")[1]
        session = provider.complete_login(
            code="stub_code", state=state, code_verifier=verifier,
            redirect_uri="https://fly-app.fly.dev/auth/callback",
        )
        client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
        client.cookies.set(SESSION_AT_COOKIE, session.access_token)
        r = client.post(
            _MUTATING_ROUTE,
            headers={"Sec-Fetch-Site": "cross-site"},
            json={"key": "OPENAI_API_KEY", "value": "x"},
        )
        assert r.status_code == 403
        assert r.json().get("error") == "cross_origin_blocked"
    finally:
        clear_providers()
        web_server.app.state.auth_required = prev_required
        web_server.app.state.bound_host = prev_host
