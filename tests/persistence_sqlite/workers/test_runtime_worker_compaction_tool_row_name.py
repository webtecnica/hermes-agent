"""A tool row's wire ``name`` must not refuse a worker write.

Compaction copies a kept row and drops only the persistence marker
(``agent.context_compressor._fresh_compaction_message_copy``), so an alternation-repair stub
(``agent/agent_runtime_helpers.py``: ``{"role": "tool", "name": ..., ...}``) reaches the worker
writer with its wire ``name``. The in-process session writes by column and ignores the key; the
worker's closed field list refused it (#159238).

``compression.archive`` through ``RuntimeSessionStore`` projects its rows, so that path already
survived; the receipt itself and the unprojected ``transcript.append`` of the same row did not.
"""
import pytest

from agent.context_compressor import _fresh_compaction_message_copy
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.workers._receipt import mutate_worker_execution
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import

TOOL_ROW = {"role": "tool", "name": "terminal", "tool_name": "terminal", "tool_call_id": "c1", "content": "ok"}


def _archive(holder, store, messages):
    """One ``compression.archive`` receipt on the worker's session, sequence from its journal.

    The receipt consumes the journal's next sequence, so the journal advances with it: the reads
    after it go through the same store.
    """
    result = mutate_worker_execution(
        holder, **store.scope, sequence=store.journal["next_sequence"], operation="compression.archive",
        payload={"messages": messages, "model_config_patch": None, "watermark": None,
                 "lock_holder": None, "tail_count": 0})
    store.journal["next_sequence"] += 1
    return result


def test_compression_archive_accepts_a_tool_row_carrying_its_wire_name(worker):
    persistence, store = worker
    holder = persistence.holder()
    copied = _fresh_compaction_message_copy(TOOL_ROW)
    assert "name" in copied

    result = _archive(holder, store, [{"role": "user", "content": "summary"}, copied])

    assert result["value"] == 2
    rows = store.get_messages_as_conversation("owned")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "summary"), ("tool", "ok")]
    assert rows[1]["tool_call_id"] == "c1" and rows[1]["tool_name"] == "terminal"


def test_the_worker_commit_path_sends_the_copied_tool_row(worker):
    """The producer's own entry point: ``RuntimeSessionStore.archive_and_compact``."""
    _persistence, store = worker

    assert store.archive_and_compact("owned", [{"role": "user", "content": "summary"},
                                               _fresh_compaction_message_copy(TOOL_ROW)]) == 2
    assert store.failure is None and store.journal["pending"] == []
    rows = store.get_messages_as_conversation("owned")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "summary"), ("tool", "ok")]
    assert rows[1]["tool_call_id"] == "c1" and rows[1]["tool_name"] == "terminal"


def test_the_next_transcript_append_of_that_same_row_is_accepted(worker):
    """The symptom after the rollback: the worker's ordinary append of the same row.

    ``RuntimeSessionStore.append_messages_batch`` sends the live dicts unprojected, so the closed
    field list refused the tool row here too and the turn ended ``session_persistence_failed``.
    """
    _persistence, store = worker

    store.append_messages_batch("owned", [{"role": "user", "content": "hi"},
                                          _fresh_compaction_message_copy(TOOL_ROW)])
    assert store.failure is None and store.journal["pending"] == []
    rows = store.get_messages_as_conversation("owned")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "hi"), ("tool", "ok")]
    assert rows[1]["tool_call_id"] == "c1" and rows[1]["tool_name"] == "terminal"


def test_an_unknown_key_is_still_refused(worker):
    """Control: the vocabulary is not open. Only the tool row's wire key was admitted."""
    persistence, store = worker
    holder = persistence.holder()

    with pytest.raises(AuthorityRefusal, match="invalid_params"):
        _archive(holder, store, [{"role": "user", "content": "summary"},
                                 {"role": "user", "content": "poisoned", "bogus": 1}])
    assert store.journal["pending"] == []


def test_a_tool_row_without_its_wire_name_still_commits(worker):
    """Control: the plain tool row the store itself reads back keeps committing."""
    persistence, store = worker
    holder = persistence.holder()

    assert _archive(holder, store, [{"role": "user", "content": "summary"},
                                    {"role": "tool", "tool_name": "terminal", "tool_call_id": "c1",
                                     "content": "ok"}])["value"] == 2
    rows = store.get_messages_as_conversation("owned")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "summary"), ("tool", "ok")]
    assert rows[1]["tool_name"] == "terminal"


def test_a_lone_wire_name_commits_without_inventing_a_tool_name(worker):
    """``name`` is wire-only: the row commits, and the durable name stays ``tool_name``.

    The repaired stub carries both keys (the wire ``name`` and the column's ``tool_name``). A row
    with only the wire key keeps its content and call id; no writer turns ``name`` into a column,
    exactly as an in-process session write does not.
    """
    persistence, store = worker
    holder = persistence.holder()

    assert _archive(holder, store, [{"role": "user", "content": "summary"},
                                    {"role": "tool", "name": "terminal", "tool_call_id": "c1",
                                     "content": "ok"}])["value"] == 2
    rows = store.get_messages_as_conversation("owned")
    assert (rows[1]["content"], rows[1]["tool_call_id"]) == ("ok", "c1")
    assert rows[1].get("tool_name") is None


def test_the_main_session_commit_branch_is_unchanged(worker):
    """Control: the ``persistence`` branch projects by column and never reads ``name``."""
    persistence, store = worker

    from agent.conversation_compression_archive import commit_compaction

    commit_compaction(store, persistence, "owned", [{"role": "user", "content": "summary"},
                                                    _fresh_compaction_message_copy(TOOL_ROW)])
    rows = store.get_messages_as_conversation("owned")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "summary"), ("tool", "ok")]
    assert rows[1]["tool_name"] == "terminal" and rows[1]["tool_call_id"] == "c1"
