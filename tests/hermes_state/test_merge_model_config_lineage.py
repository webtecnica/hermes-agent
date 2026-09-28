import json
import logging
from hermes_state import SessionDB
from hermes_state_sessions import _parse_model_config


def test_parse_model_config_valid():
    assert _parse_model_config(None) == {}
    assert _parse_model_config("") == {}
    assert _parse_model_config("   ") == {}
    assert _parse_model_config({"a": 1}) == {"a": 1}
    assert _parse_model_config('{"_delegate_from": "abc"}') == {"_delegate_from": "abc"}


def test_parse_model_config_unparseable_or_non_dict():
    # Malformed / truncated JSON must return None
    assert _parse_model_config('{"_delegate_from": "abc"') is None
    # JSON scalars and arrays must return None
    assert _parse_model_config("[]") is None
    assert _parse_model_config("[1, 2]") is None
    assert _parse_model_config("123") is None
    assert _parse_model_config('"some string"') is None
    assert _parse_model_config("true") is None
    # Python non-dict objects
    assert _parse_model_config([1, 2]) is None
    assert _parse_model_config(123) is None


def test_merge_model_config_preserves_unparseable_row(tmp_path, caplog):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)

    session_id = "test-session-126761"
    db.ensure_session(session_id, cwd="/tmp")

    # Manually inject a malformed / truncated model_config containing a lineage marker
    corrupt_raw = '{"_delegate_from": "parent-123", "extra": "data"'
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (corrupt_raw, session_id)))

    # Call set_session_yolo which routes through _merge_model_config_json
    with caplog.at_level(logging.ERROR, logger="hermes_state"):
        db.set_session_yolo(session_id, True)

    # Verify model_config was NOT overwritten with an empty-derived {"yolo_mode": True}
    row = db._read_one("SELECT model_config FROM sessions WHERE id = ?", (session_id,))
    assert row[0] == corrupt_raw

    # An error must be logged indicating refusal to merge into unparseable config
    assert any("has unparseable model_config; refusing to merge" in record.message for record in caplog.records)


def test_merge_model_config_merges_cleanly_on_valid_row(tmp_path):
    db_path = tmp_path / "test_state.db"
    db = SessionDB(db_path)

    session_id = "test-session-valid"
    db.ensure_session(session_id, cwd="/tmp")

    valid_raw = json.dumps({"_delegate_from": "parent-456", "foo": "bar"})
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET model_config = ? WHERE id = ?", (valid_raw, session_id)))

    db.set_session_yolo(session_id, True)

    row = db._read_one("SELECT model_config FROM sessions WHERE id = ?", (session_id,))
    merged = json.loads(row[0])
    assert merged["_delegate_from"] == "parent-456"
    assert merged["foo"] == "bar"
    assert merged["yolo_mode"] is True
