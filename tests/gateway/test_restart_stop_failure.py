"""A restart whose ``stop()`` raises must not wedge the gateway in ``draining`` (#159067).

Before this fix the restart task's exception was never retrieved (no done-callback) and
``_stop_impl`` disarmed the shutdown watchdog in a ``finally``, so a failing ``stop()`` produced no
log line and no hard exit: the gateway stayed alive in ``draining`` with ``active_agents: 0`` until
the supervisor SIGKILLed it.

These tests drive the real ``stop()`` / ``_stop_impl`` watchdog wiring with only the teardown
phases replaced, and arm the real shutdown watchdog with a short leash so its hard exit is
observed instead of waited for.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock

import pytest

import gateway.shutdown_watchdog as shutdown_watchdog_module
import gateway.run_shutdown as run_shutdown_module
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner

_STOP_PHASES = (
    "_stop_begin_teardown",
    "_stop_drain_active_work",
    "_stop_interrupt_remaining_work",
    "_stop_finalize_agents_and_adapters",
    "_stop_release_runtime_state",
    "_stop_quiesce_and_close_session_dbs",
    "_stop_persist_exit_state",
)

# Module-wide so a watchdog thread that outlives its test can never os._exit the pytest worker.
_HARD_EXITS: list[int] = []
_HARD_EXIT_WAITERS: list[threading.Event] = []


def _record_hard_exit(exit_code: int) -> None:
    _HARD_EXITS.append(exit_code)
    for waiter in _HARD_EXIT_WAITERS:
        waiter.set()


@pytest.fixture(scope="module", autouse=True)
def _no_real_hard_exit():
    patcher = pytest.MonkeyPatch()
    patcher.setattr(shutdown_watchdog_module, "_hard_exit", _record_hard_exit)
    patcher.setattr(shutdown_watchdog_module, "_write_watchdog_dump", lambda *a, **k: None)
    patcher.setattr(shutdown_watchdog_module, "_mark_exited_quietly", lambda *a, **k: None)
    yield
    patcher.undo()


@pytest.fixture
def hard_exits():
    _HARD_EXITS.clear()
    fired = threading.Event()
    _HARD_EXIT_WAITERS.append(fired)
    try:
        yield _HARD_EXITS, fired
    finally:
        _HARD_EXIT_WAITERS.remove(fired)


def _bare_stop_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._restart_requested = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._draining = False
    runner._running = True
    runner._stop_task = None
    runner._restart_drain_timeout = 5.0
    runner._active_deferred_agent_worker_count = lambda: 0
    return runner


def _patch_stop_phases(monkeypatch, *, failing: str | None = None) -> None:
    """Replace every ``_stop_*`` phase (and the settle step) with a no-op."""

    async def _begin_teardown(runner, ctx):
        ctx.started_at = time.monotonic()  # the real phase stamps the stop clock

    monkeypatch.setattr(GatewayRunner, "_stop_begin_teardown", _begin_teardown)
    for name in _STOP_PHASES[1:]:
        monkeypatch.setattr(GatewayRunner, name, AsyncMock())
    # Not awaited by ``_stop_impl``: a plain callable, not an AsyncMock coroutine.
    monkeypatch.setattr(GatewayRunner, "_stop_quiesce_and_close_session_dbs", lambda *a, **k: None)
    monkeypatch.setattr("gateway.run_runtime.settle_gateway_runtime", AsyncMock())
    if failing is not None:
        monkeypatch.setattr(
            GatewayRunner, failing, AsyncMock(side_effect=RuntimeError("drain exploded"))
        )


class _WatchdogSpy:
    """Records the armed watchdog and arms the real one with a short leash."""

    def __init__(self, dump_path):
        self.dump_path = dump_path
        self.delay_s = None
        self.done = None
        self.exit_code = None

    def __call__(self, delay_s, *, done_event=None, snapshot_fn=None, exit_code=1, **_kwargs):
        self.delay_s, self.done, self.exit_code = delay_s, done_event, exit_code
        return shutdown_watchdog_module.arm_shutdown_watchdog(
            0.1, done_event=done_event, exit_code=exit_code,
            dump_path=self.dump_path, name="test-159067-watchdog",
        )


def _shutdown_failure_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "Gateway shutdown failed" in r.getMessage()]


@pytest.mark.asyncio
async def test_stop_failure_logs_traceback_and_keeps_watchdog_armed(
    monkeypatch, caplog, tmp_path, hard_exits
):
    """A raising ``stop()`` is logged and its watchdog still hard-exits non-zero."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # real watchdog wiring, not the skip
    exit_calls, exit_fired = hard_exits
    runner = _bare_stop_runner()
    _patch_stop_phases(monkeypatch, failing="_stop_drain_active_work")
    spy = _WatchdogSpy(tmp_path / "watchdog.log")
    monkeypatch.setattr(run_shutdown_module, "arm_shutdown_watchdog", spy)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        with pytest.raises(RuntimeError, match="drain exploded"):
            await runner.stop()

    records = _shutdown_failure_records(caplog)
    assert records, "the failing stop() left no log record"
    assert records[0].exc_info is not None, "the log record must carry the traceback"
    assert records[0].exc_info[0] is RuntimeError

    assert spy.done is not None, "stop() never armed the shutdown watchdog"
    assert spy.delay_s > 0
    assert not spy.done.is_set(), "a failed stop() must not disarm the watchdog"
    assert exit_calls == [], "the watchdog fired before the test observed its armed state"

    assert exit_fired.wait(timeout=5.0), "the still-armed watchdog never reached its hard exit"
    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_successful_stop_disarms_watchdog(monkeypatch, caplog, tmp_path, hard_exits):
    """The happy path still disarms the watchdog and never logs a shutdown failure."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    exit_calls, exit_fired = hard_exits
    runner = _bare_stop_runner()
    _patch_stop_phases(monkeypatch)
    spy = _WatchdogSpy(tmp_path / "watchdog.log")
    monkeypatch.setattr(run_shutdown_module, "arm_shutdown_watchdog", spy)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        await runner.stop()

    assert spy.done.is_set(), "a completed stop() must disarm the watchdog"
    assert not _shutdown_failure_records(caplog)
    assert not exit_fired.wait(timeout=0.3), "the disarmed watchdog must not hard-exit"
    assert exit_calls == []


async def _run_failing_restart(runner):
    from gateway import restart_when_idle

    assert restart_when_idle.request_restart(runner, detached=False, via_service=False)
    task = runner._restart_task
    with pytest.raises(RuntimeError, match="stop exploded"):
        await task
    # Done-callbacks are scheduled with call_soon: let them run before asserting.
    await asyncio.sleep(0.05)


def _restart_failure_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "Gateway restart task failed" in r.getMessage()]


@pytest.mark.asyncio
async def test_restart_task_failure_is_logged_and_arms_a_watchdog(monkeypatch, caplog):
    """A restart that fails before ``stop()`` armed anything still exits non-zero."""
    from gateway import restart_when_idle

    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(restart_when_idle, "await_idle", AsyncMock(return_value=True))
    runner.stop = AsyncMock(side_effect=RuntimeError("stop exploded"))
    armed: list[tuple[float, dict]] = []

    def fake_arm(delay_s, **kwargs):
        armed.append((delay_s, kwargs))
        return threading.Event()

    monkeypatch.setattr(shutdown_watchdog_module, "arm_shutdown_watchdog", fake_arm)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        await _run_failing_restart(runner)

    records = _restart_failure_records(caplog)
    assert records, "the failed restart task left no log record"
    assert records[0].exc_info is not None, "the log record must carry the traceback"
    assert records[0].exc_info[0] is RuntimeError

    assert len(armed) == 1, "a restart failure with no watchdog pending must arm one"
    delay_s, kwargs = armed[0]
    assert delay_s > 0
    assert kwargs["exit_code"] == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.asyncio
async def test_restart_task_failure_does_not_double_arm_stop_watchdog(monkeypatch, caplog):
    """When ``stop()`` already armed the watchdog, the callback only logs it."""
    from gateway import restart_when_idle

    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(restart_when_idle, "await_idle", AsyncMock(return_value=True))
    runner.stop = AsyncMock(side_effect=RuntimeError("stop exploded"))
    runner._shutdown_watchdog_done = threading.Event()  # armed by _stop_impl, never disarmed
    armed: list[tuple[float, dict]] = []
    monkeypatch.setattr(
        shutdown_watchdog_module, "arm_shutdown_watchdog",
        lambda delay_s, **kwargs: armed.append((delay_s, kwargs)),
    )

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        await _run_failing_restart(runner)

    assert _restart_failure_records(caplog), "the failed restart task left no log record"
    assert armed == [], "the shutdown watchdog already counting down must not be armed twice"
