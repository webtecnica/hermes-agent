"""The dashboard Basic-Auth username never reaches a child, and neither does its password.

#156922: the child-environment policy stripped ``HERMES_DASHBOARD_BASIC_AUTH_PASSWORD``
and ``..._SECRET`` but handed ``..._USERNAME`` down, so a spawned Hermes saw half a
credential and logged "username is set but neither password_hash nor password is
configured" on every start. Half a credential authenticates nothing, so dropping it
costs the child no capability.
"""

import os

from tests.tools._child_env_fixtures import child_env, observe_child, observe_terminal
from tools.environments import local
from tools.environments.local_env_policy import (
    _ALWAYS_STRIP_KEYS,
    _STATIC_PROVIDER_ENV_BLOCKLIST,
)

BASIC_AUTH = (
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
    "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
)
# An operator var that is not a credential: it must survive every scrub path, so a
# policy that stripped the whole environment would still fail these tests.
INNOCENT = "HERMES_HOME"


def _plant(monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "dashboard-operator")
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "correct-horse-battery")
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", "s" * 43)


def _assert_scrubbed(observed, parent):
    for name in BASIC_AUTH:
        assert observed[name] is None, f"{name} leaked to the child"
    assert observed[INNOCENT] == parent[INNOCENT], "an innocent operator var was scrubbed"
    # The parent is untouched: the scrub copies the env it hands out.
    for name in BASIC_AUTH:
        assert os.environ.get(name) == parent[name]


def test_policy_strips_username_wherever_it_strips_the_password():
    """Both lists that already scrub the password carry the username too."""
    for name in ("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "HERMES_DASHBOARD_BASIC_AUTH_SECRET"):
        assert name in _STATIC_PROVIDER_ENV_BLOCKLIST
        assert name in _ALWAYS_STRIP_KEYS
    assert "HERMES_DASHBOARD_BASIC_AUTH_USERNAME" in _STATIC_PROVIDER_ENV_BLOCKLIST
    assert "HERMES_DASHBOARD_BASIC_AUTH_USERNAME" in _ALWAYS_STRIP_KEYS


def test_terminal_child_drops_the_whole_basic_auth_triplet(child_env, monkeypatch):
    """Terminal/browser/execute_code path: ``_STATIC_PROVIDER_ENV_BLOCKLIST`` governs it."""
    _plant(monkeypatch)
    parent = dict(os.environ)
    env = local.LocalEnvironment(cwd=str(child_env), timeout=30)
    observed = observe_terminal(env, [*BASIC_AUTH, INNOCENT, "PATH"])

    _assert_scrubbed(observed, parent)
    # The builder may reorder PATH (it hoists its own bin dir), but it must not drop an
    # operator entry: the child still resolves the same toolchain the parent had.
    child_path = observed["PATH"].split(os.pathsep)
    assert set(parent["PATH"].split(os.pathsep)) - set(child_path) == set()


def test_inherit_credentials_child_drops_the_whole_basic_auth_triplet(child_env, monkeypatch):
    """claude/codex/gemini path: Tier 1 (``_ALWAYS_STRIP_KEYS``) applies even with inheritance."""
    _plant(monkeypatch)
    parent = dict(os.environ)
    observed = observe_child(local.hermes_subprocess_env(inherit_credentials=True),
                             [*BASIC_AUTH, INNOCENT, "PATH"])

    _assert_scrubbed(observed, parent)
    assert observed["PATH"] == parent["PATH"]


def test_default_child_drops_the_whole_basic_auth_triplet(child_env, monkeypatch):
    """The same triplet stays out when credentials are not inherited at all."""
    _plant(monkeypatch)
    parent = dict(os.environ)
    observed = observe_child(local.hermes_subprocess_env(), [*BASIC_AUTH, INNOCENT, "PATH"])

    _assert_scrubbed(observed, parent)
    assert observed["PATH"] == parent["PATH"]
