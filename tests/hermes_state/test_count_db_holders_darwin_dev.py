"""Regression test for #126785:
count_db_holders on macOS matches full 64-bit vst_dev (e.g. on APFS synthetic devices > 2^32)
without truncation.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

import hermes_state_dbfile as dbfile


def test_darwin_fd_targets_unpack_64bit_dev():
    """Verify that _iter_darwin_fd_targets unpacks full 64-bit device ID."""
    dev_64bit = 0x1_8000_1234  # Exceeds 2^32
    ino_64bit = 987654321

    # Build synthetic 1200-byte record
    raw = bytearray(dbfile._DARWIN_FD_RECORD_SIZE)
    struct.pack_into("<Q", raw, dbfile._DARWIN_FD_DEV_OFFSET, dev_64bit)
    struct.pack_into("<Q", raw, dbfile._DARWIN_FD_INO_OFFSET, ino_64bit)
    path_bytes = b"/tmp/state.db\x00"
    raw[dbfile._DARWIN_FD_PATH_OFFSET:dbfile._DARWIN_FD_PATH_OFFSET + len(path_bytes)] = path_bytes

    record_mock = MagicMock()
    record_mock.raw = bytes(raw)

    mock_lib = MagicMock()
    mock_lib.proc_pidinfo.return_value = dbfile._DARWIN_PROC_FD_INFO_SIZE
    mock_lib.proc_pidfdinfo.return_value = dbfile._DARWIN_FD_RECORD_SIZE

    with patch.object(dbfile, "_darwin_libproc", return_value=mock_lib):
        with patch.object(dbfile, "_darwin_all_pids", return_value=[42]):
            with patch("ctypes.create_string_buffer") as mock_buf:
                # First call: proc_pidinfo buffer containing 1 fd (fd 3)
                fd_listing = bytearray(4096)
                struct.pack_into("<i", fd_listing, 0, 3)
                fd_buf = MagicMock()
                fd_buf.raw = bytes(fd_listing)

                # Configure mock_buf return values
                mock_buf.side_effect = [fd_buf, record_mock]

                results = list(dbfile._iter_darwin_fd_targets())
                assert len(results) == 1
                pid, fd, target, (st_dev, st_ino) = results[0]
                assert pid == 42
                assert fd == 3
                assert target == "/tmp/state.db"
                assert st_dev == dev_64bit
                assert st_ino == ino_64bit


def test_count_db_holders_matches_64bit_dev_on_darwin():
    """count_db_holders must return non-zero when holder has 64-bit device number > 2^32."""
    dev_64bit = 0x2_0000_0001
    ino = 123456

    mock_stat = SimpleNamespace(st_dev=dev_64bit, st_ino=ino)
    mock_targets = [(101, 4, "/path/state.db", (dev_64bit, ino))]

    with patch("sys.platform", "darwin"):
        with patch("os.path.realpath", return_value="/path/state.db"):
            with patch("os.stat", return_value=mock_stat):
                with patch.object(dbfile, "_iter_darwin_fd_targets", return_value=mock_targets):
                    holders = dbfile.count_db_holders(Path("/path/state.db"))
                    assert holders == 1
