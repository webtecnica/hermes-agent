"""Read-only full-Chromium selection shared by browser launchers."""

from __future__ import annotations

import os

import pm


def chromium_executable(*, allow_override: bool = True) -> str | None:
    """Prefer an explicit override unless the caller needs PM's own Chromium."""
    override = None
    if allow_override:
        try:
            from agent.secret_scope import current_secret_scope, get_secret, serves_routed_profile

            if serves_routed_profile():
                override = (current_secret_scope() or {}).get("AGENT_BROWSER_EXECUTABLE_PATH")
            else:
                override = get_secret("AGENT_BROWSER_EXECUTABLE_PATH")
        except Exception:
            override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH")
    if override and str(override).strip():
        return str(override).strip()
    installed = pm.installed_package("chromium")
    return str(installed.binary) if installed and installed.binary is not None else None
