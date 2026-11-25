"""Regression tests for #159863: BLOB-affinity cells written by legacy ``state.db`` files.

A value bound as ``bytes`` is stored as a BLOB even when the column declares TEXT
(SQLite never coerces a BLOB by column affinity), so a legacy row can hand a read
path ``bytes`` where it expects ``str``. Two reads crashed or silently degraded:

* ``_store_system_prompt`` hashed the prompt with ``.encode("utf-8")`` ->
  ``AttributeError: 'bytes' object has no attribute 'encode'``. The v25 migration
  (``_dedupe_legacy_system_prompts``) is the caller that REUSES a stored prompt: it
  reads ``sessions.system_prompt`` back off disk and hands it to the hasher, so a
  legacy BLOB row aborted schema init.
* ``_decode_content`` only inspected ``str`` cells, so a prefixed JSON cell held as
  ``bytes`` came back as raw ``bytes`` instead of the decoded structure.

Both are read-side only: no schema change, no data rewrite. These tests use a real
``StoreHolder`` against a real sqlite file in ``tmp_path`` and mock nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.core import _CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session

LEGACY_PROMPT = "You are Hermes, a legacy inline system prompt."
DECODE_WARNING = "Failed to decode JSON-encoded message content; returning raw string"
# The v25 prompt-dedupe migration boundary: anything below runs
# ``_dedupe_legacy_system_prompts`` on schema init.
V25_BOUNDARY = 24


def _downgrade_so_the_prompt_dedupe_reruns(db_path):
    """Mark the store as pre-v25 so opening it re-runs the prompt dedupe migration."""
    raw = sqlite3.connect(db_path)
    raw.execute("UPDATE schema_version SET version = ?", (V25_BOUNDARY,))
    raw.commit()
    raw.close()


def _store_legacy_blob_cell(holder, sql, params):
    """Write *params* verbatim (bytes stay BLOB) through the holder's own writer."""
    holder._conn.execute(sql, params)
    holder._conn.commit()


class TestLegacyBlobSystemPrompt:
    def test_v25_prompt_dedupe_rehashes_a_blob_affinity_prompt(self, tmp_path):
        """The migration reuses the stored prompt: it must hash the decoded text."""
        db_path = tmp_path / "state.db"
        db = StoreHolder(db_path=db_path)
        create_session(db, "s1", "cli")
        _store_legacy_blob_cell(
            db, "UPDATE sessions SET system_prompt = ? WHERE id = 's1'",
            (sqlite3.Binary(LEGACY_PROMPT.encode("utf-8")),),
        )
        stored, affinity = db._conn.execute(
            "SELECT system_prompt, typeof(system_prompt) FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert (stored, affinity) == (LEGACY_PROMPT.encode("utf-8"), "blob")
        db.close()

        _downgrade_so_the_prompt_dedupe_reruns(db_path)

        # Today this raises AttributeError from _store_system_prompt.
        reopened = StoreHolder(db_path=db_path)
        try:
            expected = hashlib.sha256(LEGACY_PROMPT.encode("utf-8")).hexdigest()
            prompt_hash, inline = reopened._conn.execute(
                "SELECT system_prompt_hash, system_prompt FROM sessions WHERE id = 's1'"
            ).fetchone()
            assert prompt_hash == expected
            assert inline is None  # deduped: the inline snapshot is cleared
            row_hash, prompt, prompt_affinity = reopened._conn.execute(
                "SELECT hash, prompt, typeof(prompt) FROM system_prompts"
            ).fetchone()
            assert (row_hash, prompt, prompt_affinity) == (expected, LEGACY_PROMPT, "text")
        finally:
            reopened.close()

    def test_plain_str_prompt_still_hashes_the_same(self, tmp_path):
        """Control: a normal ``str`` prompt is byte-identical to the pre-fix behavior."""
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            expected = hashlib.sha256(LEGACY_PROMPT.encode("utf-8")).hexdigest()
            assert db._store_system_prompt(db._conn, LEGACY_PROMPT) == expected
            assert db._conn.execute(
                "SELECT prompt FROM system_prompts WHERE hash = ?", (expected,)
            ).fetchone()[0] == LEGACY_PROMPT
        finally:
            db.close()

    def test_none_prompt_stays_none(self, tmp_path):
        """Control: ``None`` (no prompt to store) is still ``None``, not an empty hash."""
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            assert db._store_system_prompt(db._conn, None) is None
        finally:
            db.close()


class TestLegacyBlobMessageContent:
    def test_get_messages_decodes_a_blob_json_cell(self, tmp_path):
        """A prefixed JSON cell held as BLOB is decoded, not returned as ``bytes``."""
        payload = [{"type": "text", "text": "hi"}]
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            create_session(db, "s1", "cli")
            msg_id = append_message(db, "s1", "user", content="placeholder")
            _store_legacy_blob_cell(
                db, "UPDATE messages SET content = ? WHERE id = ?",
                (sqlite3.Binary((_CONTENT_JSON_PREFIX + json.dumps(payload)).encode("utf-8")), msg_id),
            )
            _content, affinity = db._conn.execute(
                "SELECT content, typeof(content) FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
            assert affinity == "blob"

            # Today this comes back as raw ``bytes``.
            rows = get_messages(db, "s1")
            assert rows[-1]["content"] == payload
        finally:
            db.close()

    def test_plain_str_content_is_unchanged(self, tmp_path):
        """Control: an ordinary text cell round-trips as the same ``str``."""
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            create_session(db, "s1", "cli")
            append_message(db, "s1", "user", content="plain text")
            rows = get_messages(db, "s1")
            assert rows[-1]["content"] == "plain text"
            assert isinstance(rows[-1]["content"], str)
        finally:
            db.close()

    def test_blob_cell_without_a_prefix_stays_bytes(self, tmp_path):
        """Boundary: a bytes cell that is NOT structured content keeps its raw value — a
        schema column must never vanish or silently change type (#116510)."""
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            create_session(db, "s1", "cli")
            msg_id = append_message(db, "s1", "user", content="placeholder")
            _store_legacy_blob_cell(
                db, "UPDATE messages SET content = ? WHERE id = ?",
                (sqlite3.Binary(b"\xff\xfe not utf-8"), msg_id),
            )
            rows = get_messages(db, "s1")
            assert rows[-1]["content"] == b"\xff\xfe not utf-8"
        finally:
            db.close()

    def test_invalid_prefixed_str_cell_still_warns_once(self, tmp_path, caplog):
        """Control: the prefix keeps ONE owner — a bad ``str`` cell still falls into
        ``_json_or`` with its single warning and comes back as the same string."""
        cell = _CONTENT_JSON_PREFIX + "{bad"
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            create_session(db, "s1", "cli")
            msg_id = append_message(db, "s1", "user", content="placeholder")
            _store_legacy_blob_cell(db, "UPDATE messages SET content = ? WHERE id = ?", (cell, msg_id))

            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                rows = get_messages(db, "s1")
            assert rows[-1]["content"] == cell
            assert isinstance(rows[-1]["content"], str)
            assert [(r.name, r.levelname, r.getMessage()) for r in caplog.records] == [
                ("hermes_state", "WARNING", DECODE_WARNING)
            ]
        finally:
            db.close()

    def test_invalid_prefixed_blob_cell_warns_once_after_decoding(self, tmp_path, caplog):
        """A corrupted ``bytes`` cell reaches the same ``_json_or`` fallback: the bytes
        are decoded (replacement-safe) and the single existing warning still fires."""
        db = StoreHolder(db_path=tmp_path / "state.db")
        try:
            create_session(db, "s1", "cli")
            msg_id = append_message(db, "s1", "user", content="placeholder")
            _store_legacy_blob_cell(
                db, "UPDATE messages SET content = ? WHERE id = ?",
                (sqlite3.Binary((_CONTENT_JSON_PREFIX + "{bad").encode("utf-8")), msg_id),
            )

            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                rows = get_messages(db, "s1")
            assert rows[-1]["content"] == _CONTENT_JSON_PREFIX + "{bad"
            assert [(r.name, r.levelname, r.getMessage()) for r in caplog.records] == [
                ("hermes_state", "WARNING", DECODE_WARNING)
            ]
        finally:
            db.close()
