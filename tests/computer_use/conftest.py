"""The status probe cache is process-global: every Computer Use test starts and ends with it empty."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_computer_use_status_cache():
    from tools.computer_use import permissions

    permissions.invalidate_status_cache()
    yield
    permissions.invalidate_status_cache()
