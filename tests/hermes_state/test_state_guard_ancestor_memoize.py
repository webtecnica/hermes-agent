import pytest
from unittest.mock import MagicMock
import hermes_state_guard


def test_has_pytest_ancestor_does_not_memoize_transient_error(monkeypatch):
    monkeypatch.setattr(hermes_state_guard, "_PYTEST_ANCESTOR", None)

    # 1. Simulate an AccessDenied exception while walking parents
    class MockProcessDenied:
        def parents(self):
            raise PermissionError("Access denied walking process tree")

    mock_psutil = MagicMock()
    mock_psutil.Process.return_value = MockProcessDenied()
    monkeypatch.setattr(hermes_state_guard, "psutil", mock_psutil)

    res1 = hermes_state_guard._has_pytest_ancestor()
    assert res1 is False
    # Crucial check: _PYTEST_ANCESTOR must NOT be memoized as False
    assert hermes_state_guard._PYTEST_ANCESTOR is None

    # 2. Subsequent call where walk succeeds and finds pytest
    mock_pytest_parent = MagicMock()
    mock_pytest_parent.cmdline.return_value = ["/usr/bin/python", "-m", "pytest"]
    mock_pytest_parent.name.return_value = "pytest"

    class MockProcessSuccess:
        def parents(self):
            return [mock_pytest_parent]

    mock_psutil.Process.return_value = MockProcessSuccess()

    res2 = hermes_state_guard._has_pytest_ancestor()
    assert res2 is True
    # Now it must be memoized as True
    assert hermes_state_guard._PYTEST_ANCESTOR is True

    # 3. Further calls return True directly from cache
    mock_psutil.Process.side_effect = AssertionError("Should not be called when memoized")
    assert hermes_state_guard._has_pytest_ancestor() is True
