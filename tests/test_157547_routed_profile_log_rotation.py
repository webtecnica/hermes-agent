"""#157547: a routed profile's agent.log follows ITS OWN ``logging.*`` rotation limits.

A multiplexed gateway (or the Desktop/``serve`` backend) writes every served profile's
``agent.log`` through ``_ProfileRoutingFileHandler``. The router used to copy the launch
profile's handler limits and reuse them for every routed home, so a served profile's
``logging.max_size_mb`` / ``logging.backup_count`` were silently ignored. These tests pin
the per-home limits and the two controls: the launch profile's own file must not change,
and a standalone (unrouted) profile must keep resolving the same values.
"""
import logging
from pathlib import Path

import pytest

import hermes_logging
from hermes_logging import _ProfileRoutingFileHandler


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level sentinel and drop queued file handlers between tests."""
    hermes_logging._logging_initialized = False
    hermes_logging._reset_queued_handlers()
    root = logging.getLogger()
    prev_root_level = root.level
    root.setLevel(logging.NOTSET)
    pre_existing = list(root.handlers)
    hermes_logging._install_session_record_factory()
    yield
    hermes_logging._reset_queued_handlers()
    for handler in list(root.handlers):
        if handler not in pre_existing:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(prev_root_level)
    hermes_logging._logging_initialized = False
    hermes_logging.clear_session_context()


def _make_home(path: Path, config_text: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    (path / "config.yaml").write_text(config_text, encoding="utf-8")
    return path


def _emit_for_home(home: Path, message: str, level: int = logging.INFO) -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        logging.getLogger("repro.profile-rotation").log(level, message)
    finally:
        reset_hermes_home_override(token)
    hermes_logging.flush_log_queue()


def _router(filename: str) -> _ProfileRoutingFileHandler:
    routers = [
        handler for handler in hermes_logging._queued_file_handlers
        if isinstance(handler, _ProfileRoutingFileHandler) and handler._filename == filename
    ]
    assert len(routers) == 1, routers
    return routers[0]


def _limits(handler) -> tuple[int, int]:
    return handler.maxBytes, handler.backupCount


def _setup_logging_for(home: Path) -> None:
    """``setup_logging()`` as that profile runs it alone: its home IS ``HERMES_HOME``.

    ``setup_logging`` resolves ``logging.*`` from ``get_config_path()`` (the process HERMES_HOME),
    so the launch profile's own limits must be read with the override pointed at its home.
    """
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        hermes_logging.setup_logging(hermes_home=home, mode="gateway")
    finally:
        reset_hermes_home_override(token)


MIB = 1024 * 1024


def test_routed_profile_uses_its_own_rotation_limits(tmp_path):
    """The served profile rotates at ITS 20 MiB x 10, not the launch profile's 5 MiB x 3."""
    launch = _make_home(tmp_path / "launch", "model: {}\n")
    served = _make_home(
        tmp_path / "served", "logging:\n  max_size_mb: 20\n  backup_count: 10\n"
    )
    _setup_logging_for(launch)
    assert hermes_logging.enable_profile_log_routing([launch, served]) is True

    _emit_for_home(served, "hello from served profile", level=logging.WARNING)

    router = _router("agent.log")
    # Control: the launch profile's own file keeps its resolved limits (5 MiB x 3).
    assert (router._max_bytes, router._backup_count) == (5 * MIB, 3)
    child = router._profile_handlers[served.resolve()]
    assert _limits(child) == (20 * MIB, 10)
    assert "hello from served profile" in (served / "logs" / "agent.log").read_text(
        encoding="utf-8-sig"
    )
    # ``logging.*`` only drives agent.log in a standalone profile; component logs keep
    # their fixed spec, so the routed errors.log must not inherit the served limits either.
    assert _limits(_router("errors.log")._profile_handlers[served.resolve()]) == (2 * MIB, 2)


def test_routed_profile_without_logging_block_falls_back_to_defaults(tmp_path):
    """A routed profile with no ``logging`` block gets the 5 MiB x 3 default, not the launch's."""
    launch = _make_home(
        tmp_path / "launch", "logging:\n  max_size_mb: 50\n  backup_count: 7\n"
    )
    served = _make_home(tmp_path / "served", "model: {}\n")
    _setup_logging_for(launch)
    assert hermes_logging.enable_profile_log_routing([launch, served]) is True

    _emit_for_home(served, "hello from a profile with no logging block")

    router = _router("agent.log")
    assert (router._max_bytes, router._backup_count) == (50 * MIB, 7)
    child = router._profile_handlers[served.resolve()]
    assert _limits(child) == (5 * MIB, 3)


def test_standalone_profile_keeps_its_own_rotation_limits(tmp_path):
    """Control: an unrouted profile already resolves 20 MiB x 10 and must not regress."""
    served = _make_home(
        tmp_path / "served", "logging:\n  max_size_mb: 20\n  backup_count: 10\n"
    )
    _setup_logging_for(served)
    logging.getLogger("repro.standalone").info("hello standalone")
    hermes_logging.flush_log_queue()

    direct = [
        handler for handler in hermes_logging._queued_file_handlers
        if isinstance(handler, hermes_logging.RotatingFileHandler)
        and Path(handler.baseFilename) == (served / "logs" / "agent.log").resolve()
    ]
    assert len(direct) == 1, direct
    assert _limits(direct[0]) == (20 * MIB, 10)
