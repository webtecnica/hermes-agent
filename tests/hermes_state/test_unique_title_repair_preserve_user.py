"""Regression test for #126764:
Unique title repair must preserve user-typed session names via disambiguation,
and scope NULL clearing to auto-generated/derived titles while logging both.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


def test_unique_title_repair_preserves_user_titles_and_nulls_derived(tmp_path, caplog):
    db_path = tmp_path / "state.db"
    
    # Simulate a pre-index database containing duplicates
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    # Drop the unique index if SCHEMA_SQL created it
    conn.execute("DROP INDEX IF EXISTS idx_sessions_title_unique")
    
    # 2 user sessions with identical title
    conn.execute(
        "INSERT INTO sessions (id, source, title, title_source, started_at) VALUES (?, ?, ?, ?, ?)",
        ("user_older", "cli", "User Session", "user", 100.0),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, title, title_source, started_at) VALUES (?, ?, ?, ?, ?)",
        ("user_newer", "cli", "User Session", "user", 200.0),
    )
    
    # 2 auto-derived sessions with identical title
    conn.execute(
        "INSERT INTO sessions (id, source, title, title_source, started_at) VALUES (?, ?, ?, ?, ?)",
        ("derived_older", "cli", "Derived Session", "derived", 100.0),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, title, title_source, started_at) VALUES (?, ?, ?, ?, ?)",
        ("derived_newer", "cli", "Derived Session", "derived", 200.0),
    )
    conn.commit()
    conn.close()

    # Now open with SessionDB, which runs _ensure_unique_title_index
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        sdb = SessionDB(db_path=db_path)
        try:
            # Check user titles
            user_newer_title = sdb._conn.execute(
                "SELECT title FROM sessions WHERE id = 'user_newer'"
            ).fetchone()[0]
            user_older_title = sdb._conn.execute(
                "SELECT title FROM sessions WHERE id = 'user_older'"
            ).fetchone()[0]
            assert user_newer_title == "User Session"
            assert user_older_title == "User Session (2)"

            # Check derived titles
            derived_newer_title = sdb._conn.execute(
                "SELECT title FROM sessions WHERE id = 'derived_newer'"
            ).fetchone()[0]
            derived_older_title = sdb._conn.execute(
                "SELECT title FROM sessions WHERE id = 'derived_older'"
            ).fetchone()[0]
            assert derived_newer_title == "Derived Session"
            assert derived_older_title is None

            # Check warnings logged
            assert any("Disambiguating duplicate user-typed session title" in rec.message for rec in caplog.records)
            assert any("Clearing duplicate auto-derived session title" in rec.message for rec in caplog.records)

            # Check unique index is present and enforces uniqueness
            with pytest.raises(sqlite3.IntegrityError):
                sdb._conn.execute(
                    "INSERT INTO sessions (id, title, started_at) VALUES ('dup', 'User Session', 300.0)"
                )
        finally:
            sdb.close()
