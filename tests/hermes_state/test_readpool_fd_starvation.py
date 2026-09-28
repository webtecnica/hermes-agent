"""Regression test for #126782:
_open_fd_count uses a distinct sentinel for starved descriptor probe,
stamps cache after probe, and _fd_headroom_ok logs warning and recovers.
"""

from __future__ import annotations

import errno
import logging
import os
import time
from unittest.mock import patch
import pytest

import hermes_state_readpool as rp


def test_open_fd_count_starved_sentinel():
    with patch("os.listdir", side_effect=OSError(errno.EMFILE, "Too many open files")):
        assert rp._open_fd_count() is rp._FD_COUNT_STARVED


def test_fd_headroom_starvation_and_post_probe_stamp(caplog):
    # Reset cache
    with rp._fd_usage_lock:
        rp._fd_usage_cache = (0.0, None)

    # Mock soft limit to 100
    with patch.object(rp, "_fd_soft_limit", return_value=100):
        # Force EMFILE
        with patch.object(rp, "_open_fd_count", return_value=rp._FD_COUNT_STARVED):
            start = time.monotonic()
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                assert not rp._fd_headroom_ok()
            end = time.monotonic()

            with rp._fd_usage_lock:
                stamp, cached = rp._fd_usage_cache
                assert cached is rp._FD_COUNT_STARVED
                assert start <= stamp <= end + 0.1

            assert any("fd probe starved" in rec.message for rec in caplog.records)

        # After expiry, simulate recovery
        with rp._fd_usage_lock:
            rp._fd_usage_cache = (0.0, None)

        with patch.object(rp, "_open_fd_count", return_value=20):
            assert rp._fd_headroom_ok()
            with rp._fd_usage_lock:
                _, cached = rp._fd_usage_cache
                assert cached == 20
