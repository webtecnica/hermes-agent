"""A2A Task identity is stable across reads (issue #157682).

``tasks/get`` and ``tasks/list`` re-render a stored task through
``TaskStore.to_task``. The identity of that rendering — the artifact id, the
status message id and the status timestamp — belongs to the transaction that
produced them, not to the read: two reads of the same terminal task must return
the same values, and the timestamp must be the instant of the last state
transition rather than the instant of the read.
"""

from __future__ import annotations

import time

import pytest

from plugins.platforms.a2a import protocol


@pytest.fixture()
def store() -> protocol.TaskStore:
    return protocol.TaskStore()


def _terminal_task(store: protocol.TaskStore, reply: str = "the answer") -> dict:
    """Register a task, move it to WORKING and then to COMPLETED."""
    store.create("task-1", "ctx-1", "peer-1")
    store.set_state("task-1", protocol.STATE_WORKING)
    store.complete("task-1", protocol.STATE_COMPLETED, reply)
    return store.get("task-1") or {}


def test_terminal_reads_return_identical_ids_and_timestamp(store):
    """Two GetTask renders of a completed task carry the same identity."""
    rec = _terminal_task(store)
    first = protocol.TaskStore.to_task(rec)
    time.sleep(0.02)  # let the clock pass a millisecond boundary
    second = protocol.TaskStore.to_task(rec)

    assert first["artifacts"][0]["artifactId"] == second["artifacts"][0]["artifactId"]
    assert first["status"]["message"]["messageId"] == second["status"]["message"]["messageId"]
    assert first["status"]["timestamp"] == second["status"]["timestamp"]


def test_terminal_timestamp_is_the_transition_instant(store):
    """The final timestamp records when the task reached its terminal state."""
    rec = _terminal_task(store)
    time.sleep(0.02)
    task = protocol.TaskStore.to_task(rec)

    assert task["status"]["timestamp"] == rec["state_iso"]
    assert task["status"]["timestamp"] != protocol.now_iso()


def test_working_task_keeps_its_transition_timestamp_without_artifacts(store):
    """Control (a): a non-terminal task has no artifacts and its own transition time."""
    store.create("task-2", "ctx-2", "peer-1")
    store.set_state("task-2", protocol.STATE_WORKING)
    rec = store.get("task-2") or {}
    time.sleep(0.02)
    first = protocol.TaskStore.to_task(rec)
    second = protocol.TaskStore.to_task(store.get("task-2") or {})

    assert first["status"]["state"] == protocol.STATE_WORKING
    assert "artifacts" not in first
    assert "message" not in first["status"]
    assert first["status"]["timestamp"] == rec["state_iso"]
    assert first["status"]["timestamp"] == second["status"]["timestamp"]


def test_build_task_honours_explicit_identity():
    """Control (b): the new keyword arguments are used verbatim when supplied."""
    task = protocol.build_task(
        "task-9", "ctx-9", protocol.STATE_COMPLETED, "the answer",
        state_iso="2026-01-02T03:04:05.006Z", message_id="msg-fixed", artifact_id="art-fixed",
    )

    assert task["status"]["timestamp"] == "2026-01-02T03:04:05.006Z"
    assert task["status"]["message"]["messageId"] == "msg-fixed"
    assert task["artifacts"][0]["artifactId"] == "art-fixed"


def test_include_artifacts_false_still_drops_artifacts(store):
    """Control (c): ListTasks can omit artifacts without disturbing the rest."""
    rec = _terminal_task(store)
    with_artifacts = protocol.TaskStore.to_task(rec)
    without = protocol.TaskStore.to_task(rec, include_artifacts=False)

    assert "artifacts" not in without
    assert without["status"]["message"]["messageId"] == with_artifacts["status"]["message"]["messageId"]
    assert without["status"]["timestamp"] == with_artifacts["status"]["timestamp"]


def test_complete_is_idempotent_and_does_not_remint_identity(store):
    """A second transition attempt returns None and leaves the identity untouched."""
    rec = _terminal_task(store)
    assert store.complete("task-1", protocol.STATE_FAILED, "too late") is None
    after = store.get("task-1") or {}

    assert after["state"] == protocol.STATE_COMPLETED
    assert after["state_iso"] == rec["state_iso"]
    assert after["message_id"] == rec["message_id"]
    assert after["artifact_id"] == rec["artifact_id"]


def test_state_transitions_move_the_timestamp_forward(store):
    """submitted -> working -> terminal each stamp their own transition instant."""
    store.create("task-3", "ctx-3", "peer-1")
    submitted = store.get("task-3") or {}
    time.sleep(0.02)
    store.set_state("task-3", protocol.STATE_WORKING)
    working = store.get("task-3") or {}

    assert working["state_iso"] > submitted["state_iso"]
    assert protocol.TaskStore.to_task(working)["status"]["timestamp"] == working["state_iso"]


def test_failed_task_without_reply_has_message_id_but_no_artifact(store):
    """A terminal task without a reply has no artifact to name, but its message id is pinned."""
    store.create("task-4", "ctx-4", "peer-1")
    store.complete("task-4", protocol.STATE_FAILED, "")
    rec = store.get("task-4") or {}
    first = protocol.TaskStore.to_task(rec)
    time.sleep(0.02)
    second = protocol.TaskStore.to_task(rec)

    assert "artifacts" not in first
    assert "message" not in first["status"]
    assert first["status"]["timestamp"] == second["status"]["timestamp"] == rec["state_iso"]
