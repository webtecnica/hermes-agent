"""Tests for profile-scoped AGENT_BROWSER_EXECUTABLE_PATH resolution in routed profiles (#143170)."""

import os
import pm
import pytest

from agent.secret_scope import (
    get_secret,
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from hermes_cli.browser_runtime import chromium_executable
from tools.browser_tool import _build_browser_env
from tools.browser_tool_session import _agent_browser_command_env
from tools.browser_use_cli import _base_subprocess_env


@pytest.fixture(autouse=True)
def _cleanup_scope():
    """Ensure multiplex context and secret scope are cleaned up after each test."""
    yield
    try:
        set_multiplex_context(False)
    except Exception:
        pass
    try:
        set_secret_scope(None)
    except Exception:
        pass


def test_scoped_browser_executable_path_honored_in_routed_profile(tmp_path, monkeypatch):
    """Reproduction for #143170: A secondary profile's AGENT_BROWSER_EXECUTABLE_PATH
    must be honored in _build_browser_env, chromium_executable, and _agent_browser_command_env."""
    renderer = str(tmp_path / "profile-owned-chrome")
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)

    m_token = set_multiplex_context(True)
    s_token = set_secret_scope(
        {"AGENT_BROWSER_EXECUTABLE_PATH": renderer},
        profile_home=str(tmp_path / "secondary-profile"),
    )
    try:
        assert get_secret("AGENT_BROWSER_EXECUTABLE_PATH") == renderer
        assert _build_browser_env().get("AGENT_BROWSER_EXECUTABLE_PATH") == renderer
        assert chromium_executable() == renderer

        socket_dir = str(tmp_path / "probe-socket")
        cmd_env = _agent_browser_command_env(socket_dir)
        assert cmd_env.get("AGENT_BROWSER_EXECUTABLE_PATH") == renderer

        base_env = _base_subprocess_env()
        assert base_env.get("AGENT_BROWSER_EXECUTABLE_PATH") == renderer
    finally:
        reset_secret_scope(s_token)
        reset_multiplex_context(m_token)


def test_conflicting_launch_and_secondary_profile_browser_executable(tmp_path, monkeypatch):
    """When the gateway process has a launch-profile AGENT_BROWSER_EXECUTABLE_PATH,
    a secondary profile with its own override must resolve its own value."""
    launch_chrome = str(tmp_path / "launch-chrome")
    secondary_chrome = str(tmp_path / "secondary-chrome")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", launch_chrome)

    m_token = set_multiplex_context(True)
    s_token = set_secret_scope(
        {"AGENT_BROWSER_EXECUTABLE_PATH": secondary_chrome},
        profile_home=str(tmp_path / "secondary-profile"),
    )
    try:
        assert _build_browser_env().get("AGENT_BROWSER_EXECUTABLE_PATH") == secondary_chrome
        assert chromium_executable() == secondary_chrome

        socket_dir = str(tmp_path / "probe-socket")
        cmd_env = _agent_browser_command_env(socket_dir)
        assert cmd_env.get("AGENT_BROWSER_EXECUTABLE_PATH") == secondary_chrome
    finally:
        reset_secret_scope(s_token)
        reset_multiplex_context(m_token)


def test_missing_scoped_override_does_not_inherit_launch_environ(tmp_path, monkeypatch):
    """When the secondary profile has NO override, it must NOT inherit the launch profile's
    AGENT_BROWSER_EXECUTABLE_PATH from os.environ. It must fall back to PM Chromium or None."""
    launch_chrome = str(tmp_path / "launch-chrome")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", launch_chrome)

    m_token = set_multiplex_context(True)
    s_token = set_secret_scope(
        {},
        profile_home=str(tmp_path / "secondary-profile"),
    )
    try:
        # _build_browser_env must not contain launch_chrome
        assert _build_browser_env().get("AGENT_BROWSER_EXECUTABLE_PATH") is None

        # chromium_executable must never return launch_chrome
        resolved = chromium_executable()
        assert resolved != launch_chrome

        # _agent_browser_command_env must never contain launch_chrome
        socket_dir = str(tmp_path / "probe-socket")
        cmd_env = _agent_browser_command_env(socket_dir)
        assert cmd_env.get("AGENT_BROWSER_EXECUTABLE_PATH") != launch_chrome
    finally:
        reset_secret_scope(s_token)
        reset_multiplex_context(m_token)


def test_single_profile_unrouted_honors_environ(tmp_path, monkeypatch):
    """In single-profile mode (unrouted), os.environ override is honored."""
    single_chrome = str(tmp_path / "single-chrome")
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", single_chrome)

    set_multiplex_context(False)
    set_secret_scope(None)

    assert _build_browser_env().get("AGENT_BROWSER_EXECUTABLE_PATH") == single_chrome
    assert chromium_executable() == single_chrome

    socket_dir = str(tmp_path / "probe-socket")
    cmd_env = _agent_browser_command_env(socket_dir)
    assert cmd_env.get("AGENT_BROWSER_EXECUTABLE_PATH") == single_chrome
