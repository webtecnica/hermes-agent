"""A delegate/branch child must never inherit its parent's gateway routing columns.

``_inherit_parent_session_metadata`` states the rule in its own docstring — routing columns are
inherited ONLY across a compression fork, because "peer recovery could repoint traffic into a
subagent's session" — but the SQL gated on the PARENT's ``end_reason`` alone.  A delegate child
whose gateway parent had already rotated on compression (long batch, queued child, detached unit)
therefore took the chat's ``session_key``/``chat_id``/``user_id``, leaving two live rows holding one
routing key (NousResearch/hermes-agent#116322, first reported as #92859).

Real ``SessionDB`` on a temp path, no mocks: the contract asserted here is the RELATIONSHIP between
the two child kinds — a compression continuation keeps inheriting, a delegate/branch fork does not.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB

ROUTING_COLUMNS = ("session_key", "chat_id", "chat_type", "thread_id", "user_id")


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _compressed_gateway_parent(db: SessionDB) -> dict:
    """A gateway conversation that rotated on compression, i.e. the one case that inherits."""
    db.create_session(
        "parent", source="telegram", session_key="agent:main:telegram:dm:42",
        chat_id="42", chat_type="dm", thread_id="7", user_id="u1",
    )
    db.end_session("parent", "compression")
    return db.get_session("parent")


@pytest.mark.parametrize("marker", ["_delegate_from", "_branched_from"])
def test_delegate_and_branch_children_do_not_take_over_the_parent_route(db: SessionDB, marker: str) -> None:
    parent = _compressed_gateway_parent(db)

    db.create_session(
        "worker", source="subagent", parent_session_id="parent", model_config={marker: "parent"},
    )
    db.create_session("continuation", source="telegram", parent_session_id="parent")

    worker, continuation = db.get_session("worker"), db.get_session("continuation")
    for column in ROUTING_COLUMNS:
        assert parent[column], f"fixture must seed {column}"
        # The continuation IS the conversation; the worker is an internal transcript.
        assert continuation[column] == parent[column], column
        assert worker[column] is None, column


def test_compression_continuation_of_a_branch_child_keeps_inheriting(db: SessionDB) -> None:
    """A continuation copies model_config verbatim — the branch marker comes along — so the
    exclusion must match the marker's VALUE against the parent id, not its mere presence."""
    db.create_session("root", source="telegram")
    db.create_session(
        "branch", source="telegram", parent_session_id="root", model_config={"_branched_from": "root"},
        session_key="agent:main:telegram:dm:42", chat_id="42", chat_type="dm", user_id="u1",
    )
    db.end_session("branch", "compression")

    db.create_session(
        "continuation", source="telegram", parent_session_id="branch", model_config={"_branched_from": "root"},
    )

    continuation = db.get_session("continuation")
    assert continuation["session_key"] == "agent:main:telegram:dm:42"
    assert continuation["chat_id"] == "42" and continuation["user_id"] == "u1"


_KEY = "agent:main:telegram:dm:42"


def _hijacked_chat(db: SessionDB) -> None:
    """A conversation plus the delegate child that took its routing identity over — the row a
    restarted gateway used to adopt, which is why the chat kept answering as the subagent."""
    db.create_session(
        "parent", source="telegram", session_key=_KEY, chat_id="42", chat_type="dm", user_id="u1",
    )
    db.end_session("parent", "agent_close")  # recoverable: not a deliberate boundary
    db.create_session(
        "worker", source="telegram", session_key=_KEY, chat_id="42", chat_type="dm", user_id="u1",
        parent_session_id="parent", model_config={"_delegate_from": "parent"},
    )


def test_route_recovery_skips_the_delegate_child_row(db: SessionDB) -> None:
    """Restart symptom: the child is the NEWEST keyed row, so ranking alone would pick it and the
    chat would resume inside the subagent transcript instead of the parent conversation."""
    _hijacked_chat(db)

    recovered = db.find_latest_gateway_session_for_peer(source="telegram", session_key=_KEY)

    assert recovered is not None
    assert recovered["id"] == "parent"


def test_route_recovery_tuple_fallback_skips_the_delegate_child_row(db: SessionDB) -> None:
    """The keyed lookup misses (the only keyed row IS the delegate child), so recovery falls back
    to the peer tuple: there the child must be disqualified too, letting the clean conversation
    row — keyless legacy data — win instead of resuming inside the subagent."""
    db.create_session(
        "parent", source="telegram", chat_id="42", chat_type="dm", user_id="u1",
    )
    db.append_message("parent", role="user", content="oi")
    db.end_session("parent", "agent_close")  # recoverable: not a deliberate boundary
    db.create_session(
        "worker", source="telegram", session_key=_KEY, chat_id="42", chat_type="dm", user_id="u1",
        parent_session_id="parent", model_config={"_delegate_from": "parent"},
    )

    recovered = db.find_latest_gateway_session_for_peer(
        source="telegram", session_key=_KEY, chat_id="42", chat_type="dm", user_id="u1",
    )

    assert recovered is not None
    assert recovered["id"] == "parent"
