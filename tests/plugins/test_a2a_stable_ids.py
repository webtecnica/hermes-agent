"""A2A task identity is minted once, at the state transition, and reused by every read.

Regression tests for issue #157682: ``build_task`` used to mint a fresh ``artifactId``,
a fresh ``status.message.messageId`` and a fresh ``status.timestamp`` on every render, so
two ``tasks/get`` reads of the same terminal task disagreed on all three fields.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def _completed_store() -> tuple[protocol.TaskStore, dict]:
    store = protocol.TaskStore()
    store.create("task-1", "ctx-1", "peer")
    store.set_state("task-1", protocol.STATE_WORKING)
    store.complete("task-1", protocol.STATE_COMPLETED, "the answer")
    return store, store.get("task-1")


class TestStableTerminalIdentity:
    def test_ids_are_identical_across_two_reads(self):
        _store, rec = _completed_store()
        first = protocol.TaskStore.to_task(rec)
        time.sleep(0.01)
        second = protocol.TaskStore.to_task(rec)
        assert first["artifacts"][0]["artifactId"] == second["artifacts"][0]["artifactId"]
        assert first["status"]["message"]["messageId"] == second["status"]["message"]["messageId"]
        assert first["status"]["timestamp"] == second["status"]["timestamp"]

    def test_timestamp_is_the_transition_instant_not_the_read(self):
        store, rec = _completed_store()
        transition = rec["state_iso"]
        time.sleep(0.02)
        task = protocol.TaskStore.to_task(store.get("task-1"))
        assert task["status"]["timestamp"] == transition
        assert task["status"]["timestamp"] < protocol.now_iso()

    def test_complete_is_idempotent_and_does_not_remint(self):
        store, rec = _completed_store()
        assert store.complete("task-1", protocol.STATE_FAILED, "late reply") is None
        again = store.get("task-1")
        assert again["state_iso"] == rec["state_iso"]
        assert again["message_id"] == rec["message_id"]
        assert again["artifact_id"] == rec["artifact_id"]
        assert protocol.TaskStore.to_task(again)["status"]["state"] == protocol.STATE_COMPLETED


class TestNonTerminalTask:
    def test_working_task_keeps_its_transition_timestamp_and_has_no_artifacts(self):
        store = protocol.TaskStore()
        store.create("task-2", "ctx-2", "peer")
        store.set_state("task-2", protocol.STATE_WORKING)
        rec = store.get("task-2")
        first = protocol.TaskStore.to_task(rec)
        time.sleep(0.02)
        second = protocol.TaskStore.to_task(store.get("task-2"))
        assert first["status"]["state"] == protocol.STATE_WORKING
        assert "artifacts" not in first
        assert first["status"]["timestamp"] == second["status"]["timestamp"] == rec["state_iso"]
        assert first["status"]["timestamp"] < protocol.now_iso()

    def test_state_iso_tracks_each_transition(self):
        store = protocol.TaskStore()
        submitted = store.create("task-3", "ctx-3", "peer")
        assert submitted["state_iso"] == submitted["created_iso"]
        time.sleep(0.01)
        store.set_state("task-3", protocol.STATE_WORKING)
        working_iso = store.get("task-3")["state_iso"]
        assert working_iso > submitted["state_iso"]
        time.sleep(0.01)
        store.complete("task-3", protocol.STATE_COMPLETED, "done")
        assert store.get("task-3")["state_iso"] > working_iso

    def test_failed_task_status_message_id_is_stable(self):
        store = protocol.TaskStore()
        store.create("task-4", "ctx-4", "peer")
        store.complete("task-4", protocol.STATE_FAILED, "went wrong")
        rec = store.get("task-4")
        first = protocol.TaskStore.to_task(rec)
        time.sleep(0.01)
        second = protocol.TaskStore.to_task(rec)
        assert "artifacts" not in first
        assert first["status"]["message"]["messageId"] == second["status"]["message"]["messageId"]


class TestBuildTaskIdentityKwargs:
    def test_explicit_identity_is_used(self):
        task = protocol.build_task("t", "c", protocol.STATE_COMPLETED, "x",
                                   state_iso="2026-01-02T03:04:05.678Z",
                                   message_id="mid", artifact_id="aid")
        assert task["status"]["timestamp"] == "2026-01-02T03:04:05.678Z"
        assert task["status"]["message"]["messageId"] == "mid"
        assert task["artifacts"][0]["artifactId"] == "aid"

    def test_without_identity_values_stay_fresh(self):
        """A task rendered before its record exists (first reply) keeps minting fresh values."""
        first = protocol.build_task("t", "c", protocol.STATE_COMPLETED, "x")
        second = protocol.build_task("t", "c", protocol.STATE_COMPLETED, "x")
        assert first["status"]["message"]["messageId"] != second["status"]["message"]["messageId"]
        assert first["artifacts"][0]["artifactId"] != second["artifacts"][0]["artifactId"]

    def test_include_artifacts_false_keeps_identity(self):
        _store, rec = _completed_store()
        task = protocol.TaskStore.to_task(rec, include_artifacts=False)
        assert "artifacts" not in task
        assert task["status"]["message"]["messageId"] == rec["message_id"]
        assert task["status"]["timestamp"] == rec["state_iso"]


class TestAdapterUsesRecordedIdentity:
    def test_immediate_terminal_task_matches_the_stored_task(self):
        """The SendMessage reply and a later tasks/get must agree on id fields."""
        store = protocol.TaskStore()
        rec = store.create("task-5", "ctx-5", "peer")
        fake = SimpleNamespace(tasks=store)
        fake._task_identity = A2AAdapter._task_identity.__get__(fake)

        task, pending = A2AAdapter._end_task(fake, rec, protocol.STATE_FAILED, "gateway not ready",
                                             stored_reply="gateway not ready")

        assert pending is None
        stored = protocol.TaskStore.to_task(store.get("task-5"))
        assert task["status"]["timestamp"] == stored["status"]["timestamp"]
        assert task["status"]["message"]["messageId"] == stored["status"]["message"]["messageId"]
        assert task == stored

    def test_stream_artifact_update_reuses_the_recorded_artifact_id(self):
        """The SSE artifactUpdate and the task's artifact must carry the same id."""
        store, rec = _completed_store()
        writes: list[str] = []
        fake = SimpleNamespace(tasks=store, _sse_write=lambda _handler, chunk: writes.append(chunk))

        A2AAdapter._emit_terminal(fake, object(), "task-1", "ctx-1", protocol.STATE_COMPLETED, "the answer")

        frames = [json.loads(chunk[len("data: "):].strip()) for chunk in writes if chunk.startswith("data: ")]
        artifact_ids = [f["artifactUpdate"]["artifact"]["artifactId"] for f in frames if "artifactUpdate" in f]
        assert artifact_ids == [rec["artifact_id"]]
        assert artifact_ids[0] == protocol.TaskStore.to_task(rec)["artifacts"][0]["artifactId"]
        assert any("statusUpdate" in f for f in frames)
