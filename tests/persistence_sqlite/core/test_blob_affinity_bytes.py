"""A ``state.db`` column without TEXT affinity must not break the prompt path.

Legacy stores can hold rows whose columns lost their TEXT affinity, so SQLite
hands the values back as ``bytes`` (a snapshot written with ``sqlite3.Binary``).
Two read paths then failed:

* v25 system-prompt dedupe called ``system_prompt.encode("utf-8")`` on the
  value and raised ``AttributeError`` — aborting schema init, not just the
  migration, because ``_dedupe_legacy_system_prompts`` only catches
  ``sqlite3.OperationalError``;
* ``_decode_content`` only recognized ``str``, so a stored ``\\x00json:``
  envelope came back as the raw BLOB instead of decoded message content.

Contract: bytes are decoded (UTF-8, replacement for bad bytes) before the hash
or the prefix check, so the row migrates and readers never get raw bytes.
"""

from __future__ import annotations

import sqlite3

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema import _dedupe_legacy_system_prompts
from persistence.sqlite.conversation._store.core import _CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.messages import _decode_content


def test_dedupe_migrates_a_blob_system_prompt(tmp_path):
    db_path = tmp_path / "state.db"
    StoreHolder(db_path=db_path).close()

    seed = sqlite3.connect(db_path)
    try:
        seed.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, 'test', 1.0)",
            ("legacy-blob",),
        )
        seed.execute(
            "UPDATE sessions SET system_prompt = ? WHERE id = ?",
            (sqlite3.Binary(b"prompt legado"), "legacy-blob"),
        )
        seed.commit()
    finally:
        seed.close()

    holder = StoreHolder(db_path=db_path)
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Must not raise AttributeError: BLOB rows migrate like TEXT ones.
            _dedupe_legacy_system_prompts(holder, conn.cursor())
            conn.commit()
            row = conn.execute(
                "SELECT system_prompt, system_prompt_hash FROM sessions WHERE id = ?",
                ("legacy-blob",),
            ).fetchone()
            assert row["system_prompt"] is None, "legacy inline prompt was not deduped"
            assert row["system_prompt_hash"], "deduped row carries no prompt hash"
            stored = conn.execute(
                "SELECT prompt FROM system_prompts WHERE hash = ?",
                (row["system_prompt_hash"],),
            ).fetchone()
            assert stored is not None, "prompt snapshot was not stored"
            assert stored["prompt"] == "prompt legado", "snapshot was not decoded to text"
        finally:
            conn.close()
    finally:
        holder.close()


def test_decode_content_decodes_a_blob_json_envelope():
    stored = (_CONTENT_JSON_PREFIX + '{"a": 1}').encode("utf-8")

    decoded = _decode_content(stored)

    assert decoded == {"a": 1}
    assert not isinstance(decoded, bytes)
    # Plain (non-envelope) legacy content must also come back as text.
    plain = _decode_content(b"plain bytes")
    assert plain == "plain bytes"
    assert not isinstance(plain, bytes)
