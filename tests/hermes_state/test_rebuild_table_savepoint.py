"""Regression test for #126763:
_rebuild_table in-transaction path must use SAVEPOINT so a failed rebuild
rolls back cleanly, leaving the live table intact and no stranded *_legacy_pk table.
"""

from __future__ import annotations

import sqlite3
import pytest

from hermes_state_schema import SessionSchemaMixin


def test_rebuild_table_in_transaction_rollback_on_failure(tmp_path):
    db_path = tmp_path / "test_rebuild.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    cursor = conn.cursor()

    # Create a table with initial rows
    cursor.execute("CREATE TABLE items (id TEXT PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO items VALUES ('1', 'apple')")
    cursor.execute("INSERT INTO items VALUES ('2', 'banana')")
    conn.commit()

    # Open an explicit transaction
    cursor.execute("BEGIN IMMEDIATE")
    assert conn.in_transaction

    # Proposed new DDL adds NOT NULL constraint to `value`
    new_ddl = "CREATE TABLE items (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
    # Bad copy sql that violates NOT NULL constraint on row 3
    bad_copy_sql = "INSERT INTO items SELECT id, NULL FROM items_legacy_pk"

    with pytest.raises(sqlite3.IntegrityError):
        SessionSchemaMixin._rebuild_table(
            cursor,
            table="items",
            legacy_name="items_legacy_pk",
            ddl=new_ddl,
            copy_sql=bad_copy_sql,
        )

    # Rebuild failed, but transaction should still be open and savepoint rolled back
    assert conn.in_transaction

    # Commit caller transaction
    conn.commit()

    # Verify table 'items' still exists with original rows, and 'items_legacy_pk' does NOT exist
    rows = conn.execute("SELECT id, value FROM items ORDER BY id").fetchall()
    assert rows == [("1", "apple"), ("2", "banana")]

    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "items" in tables
    assert "items_legacy_pk" not in tables

    conn.close()


def test_rebuild_table_in_transaction_success(tmp_path):
    db_path = tmp_path / "test_rebuild_ok.db"
    conn = sqlite3.connect(db_path, isolation_level=None)
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE items (id TEXT PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO items VALUES ('1', 'apple')")
    conn.commit()

    cursor.execute("BEGIN IMMEDIATE")
    new_ddl = "CREATE TABLE items (id TEXT PRIMARY KEY, value TEXT NOT NULL, extra TEXT DEFAULT '')"
    copy_sql = "INSERT INTO items SELECT id, value, 'new' FROM items_legacy_pk"

    SessionSchemaMixin._rebuild_table(
        cursor,
        table="items",
        legacy_name="items_legacy_pk",
        ddl=new_ddl,
        copy_sql=copy_sql,
    )

    conn.commit()
    rows = conn.execute("SELECT id, value, extra FROM items").fetchall()
    assert rows == [("1", "apple", "new")]
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "items_legacy_pk" not in tables
    conn.close()
