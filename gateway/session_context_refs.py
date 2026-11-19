"""``@`` references of one turn: expanded once, in the session's workspace, on the text the model reads.

The inbound prepare step, the classic CLI's API-only text (``cli_turn_v1.api_text``: voice prefix,
model-switch and skills notes) and managed workers (``--safe-mode`` / ``--ignore-user-config``,
which never pass the inbound step) all expand here. A local session's refusal is the turn's reply:
its adapter delivers through settlement, so a side ``send`` would never reach the viewer.
"""
import logging
import os
from contextlib import suppress
from functools import partial
from pathlib import Path

from gateway.config import Platform

logger = logging.getLogger(__name__)


def _confined_reference_root(cwd):
    """The root a path reference of THIS task may resolve under: ``cwd``, or the served profile's home
    when the task is confined to its own profile.

    A multiplexed gateway serves several profiles from one process, and a secondary profile's turn may
    read only its own store (``_cross_profile_reads_confined``, #154617). The same confinement holds
    for ``@file:``/``@folder:``: their root would otherwise be the daemon's ``TERMINAL_CWD`` (or
    ``~``), which reaches a sibling profile's memories, persona, config and session DB with a single
    reference (#157590). The profile root is the task's active home — ``get_hermes_home()`` follows
    the per-turn override — never ``os.environ``, which still names the launch profile.
    """
    from tools.session_search_tool import _cross_profile_reads_confined
    if not _cross_profile_reads_confined():
        return cwd
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _refuse_outside_root(text, cwd, allowed_root, *, plugins=True):
    """The refusal for a reference resolving outside a NARROWED ``allowed_root``, else None.

    Only a narrowed root refuses the turn (#157590's profile confinement): with the default root a
    ``@file:`` outside the workspace keeps today's warning-and-continue. The check runs through the
    expansion's own guard (``_resolve_path``), so its refusal cannot drift from the one expansion
    records, and it runs before any content is attached.
    """
    root = Path(allowed_root).expanduser().resolve()
    start = Path(cwd).expanduser().resolve()
    if root == start:
        return None
    from agent.context_references import _resolve_path, parse_context_references
    refusals = []
    for ref in parse_context_references(text, plugins=plugins):
        if ref.kind not in ("file", "folder"):
            continue
        try:
            _resolve_path(start, ref.target, allowed_root=root)
        except ValueError as exc:
            refusals.append(f"{ref.raw}: {exc}")
    return "\n".join(refusals) or None


async def expand_references(text, *, cwd, context_length, plugins=True, allowed_root=None):
    """``(text, refusal)``: the expanded text, or the refusal when the injection was refused. A failed
    expansion leaves the text as typed. ``allowed_root`` narrows where a path reference may resolve:
    the caller that knows the turn's own workspace passes it, and the default is ``cwd`` — narrowed
    to the served profile's home when the turn is confined to its own profile. A reference outside a
    narrowed root refuses the turn; outside the default root it stays a warning."""
    from agent.i18n import t
    try:
        from agent.context_references import preprocess_context_references_async
        root = _confined_reference_root(cwd) if allowed_root is None else allowed_root
        if (refusal := _refuse_outside_root(text, cwd, root, plugins=plugins)) is not None:
            return text, refusal
        expand = partial(preprocess_context_references_async, cwd=cwd, context_length=context_length,
                         allowed_root=root)
        result = await (expand(text) if plugins else expand(text, plugins=False))
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    if result.blocked:
        return text, "\n".join(result.warnings) or t("gateway.notify.context_injection_refused")
    return (result.message if result.expanded else text), None


def _inbound_cwd(runner, source):
    from gateway.session_policy import policy_for_source
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        terminal_env = os.environ.get
    # A local session's references resolve in ITS workspace: this runs before the turn binds
    # the policy scope, so TERMINAL_CWD is still the daemon's own.
    if (policy := policy_for_source(runner, source)) is not None:
        return policy.cwd
    return terminal_env("TERMINAL_CWD", os.path.expanduser("~"))


def _inbound_allowed_root(runner, source, cwd):
    """The root an inbound reference may resolve under: a local session's own workspace, else the
    served profile's home when the turn is confined to it, else ``cwd`` (the daemon's terminal cwd).

    A local session's workspace is both its cwd and its root, so it is decided BEFORE the profile
    confinement: this runs before the turn binds the policy scope, where the profile signals alone
    cannot tell a local session from a messaging turn.
    """
    from gateway.session_policy import policy_for_source
    if policy_for_source(runner, source) is not None:
        return cwd
    return _confined_reference_root(cwd)


async def inbound_references(runner, source, session_key, text):
    """``expand_references`` for an inbound turn: a local session's policy cwd, else the terminal cwd;
    under profile confinement the reference root is the served profile's home."""
    try:
        cwd = _inbound_cwd(runner, source)
        allowed_root = _inbound_allowed_root(runner, source, cwd)
        context_length = await runner._inbound_model_context_length(source, session_key)
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    return await expand_references(text, cwd=cwd, context_length=context_length, allowed_root=allowed_root)


async def turn_references(runner, event, source, session_key, text):
    """An inbound turn's text with the references of the text the model reads expanded; None when
    refused. The classic CLI's committed ``api_text`` replaces ``text`` at the turn
    (``TurnRunner._prepare_turn_message``), so ITS references are expanded and rebound for this
    execution, and the returned text, the stored user row, is ``text`` with the same attached
    context, as a typed turn stores it: later turns still read the content. A refused local turn
    leaves its refusal on ``event`` for the prepare step to answer with."""
    from gateway.session_cli_turn import cli_turn, rebind_api_text
    api_text = cli_turn(event.message_id).get('api_text')
    target = text if api_text is None else api_text
    if "@" not in target:
        return text
    if source.platform == Platform.LOCAL:
        expanded, refusal = await inbound_references(runner, source, session_key, target)
        if refusal is not None:
            event._gateway_turn_refusal = refusal
            return None
    elif (expanded := await runner._expand_inbound_context_references(source, session_key, target)) is None:
        return None
    if api_text is None:
        return expanded
    rebind_api_text(event.message_id, expanded)
    # The expansion appends its context after the message, so that tail belongs to the stored row too.
    head = api_text.lstrip()
    tail = expanded[len(head):] if expanded.startswith(head) else ""
    return f"{text}{tail}".strip() if tail else text


async def managed_turn_text(authority, row, policy):
    """``(text, refusal)`` for a managed worker's admitted text, expanded against the session's
    workspace. Built-in kinds only for a bypass session, which loads no plugins; the context length
    is the frozen policy's, never the profile file a bypass session must not read. ``None`` text means
    the admitted text as committed."""
    text = (row.get('payload') or {}).get('text')
    if not isinstance(text, str) or "@" not in text:
        return None, None
    try:
        from agent.model_metadata import get_model_context_length_async
        model_config = policy.config(authority).get('model') or {}
        pinned = model_config.get('context_length') if isinstance(model_config, dict) else None
        with suppress(TypeError, ValueError):
            pinned = int(pinned) if pinned is not None else None
        context_length = await get_model_context_length_async(
            policy.current_model(authority) or '', base_url=policy.base_url or '', api_key='',
            provider=policy.provider or '', config_context_length=pinned if isinstance(pinned, int) else None)
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    return await expand_references(text, cwd=policy.cwd, context_length=context_length,
                                   plugins=not policy.ignore_user_config,
                                   # The admitted session's own workspace is this worker's root: the
                                   # #157590 confinement guards inbound messaging turns whose root would
                                   # otherwise be the daemon's home, not a frozen workspace.
                                   allowed_root=policy.cwd)


def refused_turn_result(row, refusal):
    """A turn that never ran because its references were refused: the refusal is its answer, and a
    finite (``chat -q`` / ``-z``) prompt fails with it, as ``execute_admission`` settles a notice."""
    if row['payload'].get('finite') is True:
        return {'final_response': '', 'messages': [], 'failed': True, 'completed': False, 'error': refusal}
    return {'final_response': refusal, 'messages': []}
