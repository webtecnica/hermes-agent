"""Due-scan dispatch-limit guard: removal of a re-armed consumed one-shot is
operator-visible (#93524 follow-up to #93615).

Pre-#93615 stores (or hand edits) can hold a record that already completed a
run (last_run_at set) but was re-armed without a budget reset. The due-scan
guard removes it without firing — correct, since #93615's policy routes
re-runs through `cron resume` — but the removal must log at WARNING with a
remediation hint, never a silent INFO delete.
"""

import logging
from datetime import timedelta

import pytest

from cron.jobs import (
    claim_dispatch,
    create_job,
    get_due_jobs,
    load_jobs,
    mark_job_run,
    save_jobs,
    _hermes_now,
)

@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir (same pattern as the sibling
    grace-gate tests)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path

def test_guard_warns_on_rearmed_consumed_record(temp_home, caplog):
    job = create_job(
        prompt="x",
        schedule=(_hermes_now() + timedelta(hours=1)).isoformat(),
        name="warn-guard",
        deliver="local",
    )
    jid = job["id"]
    assert claim_dispatch(jid)
    mark_job_run(jid, True)

    # Simulate a pre-#93615 re-arm: enabled+scheduled+due, budget still spent.
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == jid:
            j["enabled"] = True
            j["state"] = "scheduled"
            j["next_run_at"] = (_hermes_now() - timedelta(seconds=5)).isoformat()
    save_jobs(jobs)

    with caplog.at_level(logging.INFO, logger="cron"):
        due = get_due_jobs()

    assert jid not in [d["id"] for d in due]
    assert jid not in [j["id"] for j in load_jobs()]
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "WITHOUT firing" in r.getMessage()
    ]
    assert warnings, "expected WARNING on removal of a re-armed consumed one-shot"
    assert "cron resume" in warnings[0].getMessage()


def test_completed_occurrence_oneshot_retires_spent_record(temp_home, monkeypatch):
    """A once job whose occurrence is already completed in the ledger must be retired

    instead of remaining as a zombie scheduled record (#126787).
    """
    job = create_job(
        prompt="zombie test",
        schedule=(_hermes_now() - timedelta(seconds=10)).isoformat(),
        name="zombie-record",
        deliver="local",
    )
    jid = job["id"]

    # Mock completed_occurrence to simulate an already completed execution for this slot
    monkeypatch.setattr("cron.occurrences.completed_occurrence", lambda j, inst: True)

    due = get_due_jobs()
    assert jid not in [d["id"] for d in due]
    # The zombie record should have been retired from jobs store
    remaining_jobs = [j["id"] for j in load_jobs()]
    assert jid not in remaining_jobs, "expected completed one-shot to be retired from jobs store"


def test_oneshot_dispatch_limit_reached_before_completed_occurrence(temp_home, monkeypatch):
    """The dispatch limit check must run before completed_occurrence and retire the spent record."""
    job = create_job(
        prompt="limit test",
        schedule=(_hermes_now() - timedelta(seconds=10)).isoformat(),
        name="limit-record",
        deliver="local",
    )
    jid = job["id"]
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == jid:
            j["repeat"] = {"times": 1, "completed": 1}
    save_jobs(jobs)

    due = get_due_jobs()
    assert jid not in [d["id"] for d in due]
    assert jid not in [j["id"] for j in load_jobs()]
