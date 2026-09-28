import pytest
from hermes_state import SessionDB


def test_search_context_window_orders_by_id_despite_regressed_clock(tmp_path):
    db_path = tmp_path / "test_search_state.db"
    db = SessionDB(db_path)

    session_id = "test-session-clock-regress"
    db.ensure_session(session_id, cwd="/tmp")

    # Insert messages where IDs are monotonic (1, 2, 3) but timestamps are inverted
    # Message 1 (preceding): timestamp 1000.0
    # Message 2 (search hit): timestamp 500.0 (simulated NTP step back)
    # Message 3 (following): timestamp 700.0
    m1_id = db.append_message(session_id, "user", "Preceding question about elephants", timestamp=1000.0)
    m2_id = db.append_message(session_id, "assistant", "Target response keyword elefante", timestamp=500.0)
    m3_id = db.append_message(session_id, "user", "Following user thank you note", timestamp=700.0)

    # Search for "elefante" with context requested
    results = db.search_messages("elefante", fields=["id", "snippet", "context"])
    assert len(results) == 1
    hit = results[0]
    assert hit["id"] == m2_id

    # The context window must contain the preceding neighbour (m1), the hit (m2), and following (m3) in ID order
    context = hit.get("context", [])
    assert len(context) == 3

    assert "Preceding question" in context[0]["content"]
    assert "Target response" in context[1]["content"]
    assert "Following user" in context[2]["content"]
