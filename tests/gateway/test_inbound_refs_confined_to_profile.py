"""#157590: an inbound ``@file:``/``@folder:`` resolves inside the served profile's home.

A multiplexed gateway serves several profiles from one process. A secondary profile's turn may read
only its own store (``_cross_profile_reads_confined``, #154617), yet its ``@`` references expanded
against the daemon's ``TERMINAL_CWD``/``~``, so one ``@file:`` inlined a sibling profile's memories,
persona, config or session DB. These tests pin the confined root: the turn's active profile home
(``get_hermes_home()``), decided where the inbound cwd is decided, with the credential deny-list
still layered on top and the unconfined/local paths unchanged.
"""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session_context_refs import _inbound_allowed_root, _inbound_cwd, expand_references, inbound_references

ALICE_MEMORY = "alice: gift idea for bob"
BOB_NOTE = "bob: own note"


def _whatsapp():
    return SimpleNamespace(platform=Platform.WHATSAPP)


def _local():
    return SimpleNamespace(platform=Platform.LOCAL)


class _Runner:
    """A gateway runner stub: only the model context length is read before expansion."""

    async def _inbound_model_context_length(self, source, session_key):
        return 200_000


@contextmanager
def _turn_of(profile_home):
    """A turn served for *profile_home* by a multiplexed gateway."""
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    context = set_multiplex_context(True)
    override = set_hermes_home_override(profile_home)
    try:
        yield
    finally:
        reset_hermes_home_override(override)
        reset_multiplex_context(context)


@pytest.fixture
def profiles():
    """Two made-up profiles under the test's isolated HERMES_HOME."""
    from hermes_constants import get_hermes_home
    root = get_hermes_home()
    alice, bob = root / "profiles" / "alice", root / "profiles" / "bob"
    (alice / "memories").mkdir(parents=True)
    bob.mkdir(parents=True)
    (alice / "memories" / "MEMORY.md").write_text(ALICE_MEMORY, encoding="utf-8")
    (bob / "notes.md").write_text(BOB_NOTE, encoding="utf-8")
    (bob / ".env").write_text("BOB_KEY=secret", encoding="utf-8")
    return alice, bob


def test_a_sibling_profiles_files_are_refused(profiles, monkeypatch):
    """The repro: bob's WhatsApp turn names alice's memory and gets a refusal, not the content."""
    alice, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _whatsapp())
        # The cwd itself is unchanged (the daemon's); only the reference root is narrowed.
        assert cwd == str(bob.parent.parent)
        assert _inbound_allowed_root(None, _whatsapp(), cwd) == bob
        text, refusal = asyncio.run(
            expand_references(f"what does this say? @file:{alice}/memories/MEMORY.md",
                              cwd=cwd, context_length=200_000))
    assert ALICE_MEMORY not in text
    assert refusal and "outside the allowed workspace" in refusal


def test_the_served_profiles_own_file_is_still_inlined(profiles, monkeypatch):
    _, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _whatsapp())
        text, refusal = asyncio.run(
            expand_references(f"@file:{bob}/notes.md", cwd=cwd, context_length=200_000))
    assert refusal is None
    assert BOB_NOTE in text


def test_a_sibling_profiles_folder_listing_is_refused(profiles, monkeypatch):
    alice, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _whatsapp())
        text, refusal = asyncio.run(
            expand_references(f"@folder:{alice}/memories", cwd=cwd, context_length=200_000))
    assert "MEMORY.md" not in text
    assert refusal and "outside the allowed workspace" in refusal


def test_the_served_profiles_own_folder_is_still_listed(profiles, monkeypatch):
    _, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _whatsapp())
        text, refusal = asyncio.run(
            expand_references(f"@folder:{bob}", cwd=cwd, context_length=200_000))
    assert refusal is None
    assert "notes.md" in text


def test_the_served_profiles_own_credentials_stay_refused(profiles, monkeypatch):
    """The deny-list still applies on top of the confinement (it is not widened by it)."""
    _, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _whatsapp())
        text, _ = asyncio.run(
            expand_references(f"@file:{bob}/.env", cwd=cwd, context_length=200_000))
    assert "BOB_KEY" not in text
    assert "credential" in text


def test_multiplex_off_keeps_the_daemon_root(profiles, monkeypatch, tmp_path):
    """A single-profile process is unchanged: the workspace under the daemon's cwd still resolves."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text(BOB_NOTE, encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    cwd = _inbound_cwd(None, _whatsapp())
    assert _inbound_allowed_root(None, _whatsapp(), cwd) == str(workspace)
    text, refusal = asyncio.run(expand_references("@file:notes.md", cwd=cwd, context_length=200_000))
    assert refusal is None
    assert BOB_NOTE in text


def test_a_local_session_keeps_resolving_in_its_workspace(profiles, monkeypatch, tmp_path):
    """A LOCAL session's frozen policy cwd is both its cwd and its root, confinement notwithstanding."""
    _, bob = profiles
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text(BOB_NOTE, encoding="utf-8")
    monkeypatch.setattr("gateway.session_policy.policy_for_source",
                        lambda runner, source: SimpleNamespace(cwd=str(workspace)))
    with _turn_of(bob):
        cwd = _inbound_cwd(None, _local())
        root = _inbound_allowed_root(None, _local(), cwd)
        assert cwd == str(workspace)
        assert root == str(workspace)
        # The local caller owns its workspace root, exactly as ``inbound_references`` passes it.
        text, refusal = asyncio.run(
            expand_references("@file:notes.md", cwd=cwd, context_length=200_000, allowed_root=root))
    assert refusal is None
    assert BOB_NOTE in text


def test_inbound_references_carries_the_confined_root(profiles, monkeypatch):
    """The gateway entry point (run_inbound -> inbound_references) refuses the sibling read."""
    alice, bob = profiles
    monkeypatch.setenv("TERMINAL_CWD", str(bob.parent.parent))
    with _turn_of(bob):
        text, refusal = asyncio.run(
            inbound_references(_Runner(), _whatsapp(), "session-key",
                               f"@file:{alice}/memories/MEMORY.md"))
    assert ALICE_MEMORY not in text
    assert refusal and "outside the allowed workspace" in refusal
