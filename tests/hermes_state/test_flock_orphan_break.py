"""Regression test for #126784:
_acquire_db_flock allows multiple orphan-break attempts when concurrent breakers
replace the lock file, and _rewrite_lock_file fsyncs before return.
"""

from __future__ import annotations

import errno
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

import hermes_state_common as hsc


def test_rewrite_lock_file_calls_fsync():
    mock_handle = MagicMock()
    mock_handle.fileno.return_value = 10
    with patch("os.fsync") as mock_fsync:
        hsc._rewrite_lock_file(mock_handle, b"test")
        mock_fsync.assert_called_once_with(10)


def test_acquire_db_flock_multiple_breaks_allowed(tmp_path):
    lock_file = tmp_path / "test.lock"
    lock_file.write_bytes(b"")
    handle = open(lock_file, "a+b")

    mock_fcntl = MagicMock()
    mock_fcntl.LOCK_EX = 2
    mock_fcntl.LOCK_NB = 4

    flock_calls = 0
    def mock_flock(fd, flags):
        nonlocal flock_calls
        flock_calls += 1
        if flock_calls <= 2:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return None

    mock_fcntl.flock = mock_flock

    modules = dict(sys.modules)
    modules["fcntl"] = mock_fcntl

    with patch.dict(sys.modules, modules):
        with patch.object(hsc, "_read_lock_holder_record", return_value={"pid": 999999, "start_ticks": 1}):
            with patch.object(hsc, "_lock_holder_provably_dead", return_value=True):
                with patch("os.unlink"):
                    with patch("os.fstat", return_value=SimpleNamespace(st_dev=1, st_ino=100)):
                        with patch("os.stat", return_value=SimpleNamespace(st_dev=1, st_ino=100)):
                            acquired, final_handle = hsc._acquire_db_flock(
                                str(lock_file), handle, timeout_seconds=0.001, poll_seconds=0.001, description="test"
                            )
                            assert acquired is True
                            final_handle.close()
