"""Regression test for #126783:
_read_budgets maintains accounting across GC of abandoned handles with checked-out permits,
preventing silent resets of the per-file read ceiling.
"""

from __future__ import annotations

import gc
from pathlib import Path
import pytest

from hermes_state import SessionDB
import hermes_state_readpool as rp


def test_read_budget_retained_across_unclosed_handle_gc(tmp_path):
    db_path = tmp_path / "state.db"

    # Open first handle
    h1 = SessionDB(db_path=db_path)
    budget1 = h1._read_budget

    assert budget1.permits_available() == rp._READ_POOL_MAX

    # Check out a read permit on h1
    assert budget1.acquire(h1)
    assert budget1.permits_available() == rp._READ_POOL_MAX - 1

    # Deliberately abandon h1 without close() and force GC
    del h1
    gc.collect()

    # Open a second handle on the same path
    h2 = SessionDB(db_path=db_path)
    budget2 = h2._read_budget

    try:
        # It must be the SAME budget preserving the checked-out permit, NOT a reset ceiling
        assert budget2 is budget1
        assert budget2.permits_available() == rp._READ_POOL_MAX - 1
    finally:
        # Release the checked-out permit and close h2
        budget2.release()
        h2.close()
        gc.collect()

    # After releasing permit and closing h2, reaping should prune the idle budget
    with rp._read_budgets_lock:
        rp._reap_idle_budgets_locked()
        key = rp._read_budget_key(db_path)
        assert key not in rp._read_budgets
