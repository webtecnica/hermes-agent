"""Markdown/QMD export helpers for Hermes sessions.

Filesystem-only: formats already-exported SessionDB dicts and writes them to user-selected export
directories. Must not mutate state.db or call delete/prune/archive APIs.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

from hermes_cli.timefmt import coerce_epoch

EXPORTER_VERSION = "hermes sessions export (md/qmd) v1"
_SHA_LINE_RE = re.compile(r"- SHA256 of exported body: `([0-9a-f]{64})`")
_SHA_PENDING = "pending"
_VERIFICATION_HEADING = "## Export verification"


def _new_sha_marker() -> str:
    """Return a per-export marker for the digest slot in the footer.

    A fixed sentinel is unsafe: a session may quote it (or a whole previous export
    footer) in its messages, and a whole-body replace would then stamp the digest over
    user text (#157707). A random marker cannot occur in content authored before it is
    drawn.
    """
    return f"__SHA256_{secrets.token_hex(16)}__"


def _iso_timestamp(value: Any) -> str:
    if value is None or value == "":
        return ""
    if (ts := coerce_epoch(value)) is None:
        return str(value)  # corrupt cell: odd-looking date, not an aborted export
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _frontmatter_line(key: str, value: Any) -> str:
    if value is None or isinstance(value, bool):
        shown = {None: "null", True: "true", False: "false"}[value]
    else:
        shown = json.dumps(value if isinstance(value, (int, float, list)) else str(value), ensure_ascii=False)
    return f"{key}: {shown}"


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _message_heading(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "message")
    name = message.get("name") or message.get("tool_name")
    label = f"Tool — {name}" if role == "tool" and name else role.capitalize()
    timestamp = _iso_timestamp(message.get("created_at") or message.get("timestamp"))
    return f"### {label}{' — ' + timestamp if timestamp else ''}"


def _session_id(session: dict[str, Any]) -> str:
    return str(session.get("id") or session.get("session_id") or "unknown-session")


def _segments(session: dict[str, Any]) -> list[dict[str, Any]]:
    segments = session.get("segments")
    return [s for s in segments if isinstance(s, dict)] if isinstance(segments, list) and segments else [session]


def _message_count(session: dict[str, Any]) -> int:
    return sum(len(seg.get("messages") or []) for seg in _segments(session))


def _render_messages(session: dict[str, Any]) -> str:
    segments = _segments(session)
    if _message_count(session) == 0:
        return "## Messages\n\n_No messages in this session._\n"
    parts: list[str] = ["## Messages\n"]
    for segment in segments:
        if len(segments) > 1:
            parts.append(f"## Compression segment: {_session_id(segment)}\n")
        for message in list(segment.get("messages") or []):
            parts.append(_message_heading(message) + "\n")
            content = message.get("content")
            rendered = "" if content is None else content.rstrip() if isinstance(content, str) else _json_block(content)
            if rendered:
                parts.append(rendered + "\n")
            if tool_calls := message.get("tool_calls"):
                parts.append("\n\n## Tool calls\n\n" + _json_block(tool_calls) + "\n")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _export_body_without_hash(
    session: dict[str, Any], *, fmt: str, exported_at: float, sha_marker: str
) -> str:
    session_id = _session_id(session)
    exported_iso = _iso_timestamp(exported_at)
    message_count = _message_count(session)
    fields = [
        ("session_id", session_id),
        ("title", session.get("title")),
        ("source", session.get("source")),
        ("created_at", _iso_timestamp(session.get("started_at") or session.get("created_at"))),
        ("updated_at", _iso_timestamp(session.get("last_active") or session.get("updated_at"))),
        ("ended_at", _iso_timestamp(session.get("ended_at"))),
        ("model", session.get("model")),
        ("provider", session.get("billing_provider") or session.get("provider")),
        ("cwd", session.get("cwd")),
        ("archived", bool(session.get("archived"))),
        ("message_count", message_count),
        ("tool_call_count", session.get("tool_call_count") or 0),
        *([("lineage_session_ids", session["lineage_session_ids"])] if session.get("lineage_session_ids") else []),
        ("format", fmt), ("exported_at", exported_iso), ("exporter", EXPORTER_VERSION),
    ]
    parts = [
        "\n".join(["---", *(_frontmatter_line(k, v) for k, v in fields), "---", ""]),
        f"# {session.get('title') or session_id}\n", f"Session ID: `{session_id}`\n",
        *([f"Source: `{session.get('source')}`\n"] if session.get("source") else []),
        *([f"Working directory: `{session.get('cwd')}`\n"] if session.get("cwd") else []),
        _render_messages(session),
        f"{_VERIFICATION_HEADING}\n",
        f"- Session id: `{session_id}`",
        f"- Exported messages: `{message_count}`",
        f"- Source DB message count at export: `{session.get('message_count', message_count)}`",
        f"- Exported at: `{exported_iso}`",
        f"- SHA256 of exported body: `{sha_marker}`",
    ]
    return "\n".join(parts).rstrip() + "\n"


def _check_fmt(fmt: str) -> None:
    if fmt not in {"md", "qmd"}:
        raise ValueError("fmt must be 'md' or 'qmd'")


def render_session_markdown(
    session: dict[str, Any], *, fmt: str = "md", include_verification: bool = True
) -> str:
    """Render a SessionDB export dictionary as Markdown/QMD text."""
    _check_fmt(fmt)
    sha_marker = _new_sha_marker()
    body = _export_body_without_hash(
        session, fmt=fmt, exported_at=time.time(), sha_marker=sha_marker
    )
    if not include_verification:
        return body.split(f"\n{_VERIFICATION_HEADING}\n", 1)[0].rstrip() + "\n"
    # The digest covers the body with the SHA line set to `pending`, which is what verify
    # recomputes. Only the footer slot is swapped: the marker is unique to this export, so
    # message text that quotes the old fixed placeholder (or a previous footer) is intact.
    head, marker, tail = body.rpartition(f"`{sha_marker}`")
    if not marker:
        raise RuntimeError("export body lost its SHA marker")
    digest_body = f"{head}`{_SHA_PENDING}`{tail}"
    digest = hashlib.sha256(digest_body.encode("utf-8")).hexdigest()
    return f"{head}`{digest}`{tail}"


def safe_session_filename(session: dict[str, Any], *, fmt: str = "md") -> str:
    """Return a deterministic, path-safe filename for a session export."""
    _check_fmt(fmt)
    title = str(session.get("title") or "session")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip(".-_").lower() or "session"
    return f"{_session_id(session)}-{slug[:60]}.{fmt}"


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_export_file(path: Path | str, session: dict[str, Any]) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, "file missing"
    text = Path(path).read_text(encoding="utf-8-sig")
    # The exporter always writes the verification block last, so the digest line belongs to
    # the trailing block. Searching from the top would pick a session body that quotes an
    # older export's footer and compare its digest against that stale line (#157707).
    block_at = text.rfind(f"\n{_VERIFICATION_HEADING}\n")
    if block_at == -1:
        return False, "sha256 marker missing"
    match = _SHA_LINE_RE.search(text, block_at + 1)
    if not match:
        return False, "sha256 marker missing"
    digest_body = (
        text[: match.start()]
        + f"- SHA256 of exported body: `{_SHA_PENDING}`"
        + text[match.end() :]
    )
    if hashlib.sha256(digest_body.encode("utf-8")).hexdigest() != match.group(1):
        return False, "sha256 mismatch"
    if f"- Exported messages: `{_message_count(session)}`" not in text:
        return False, "message count mismatch"
    if f"- Session id: `{_session_id(session)}`" not in text:
        return False, "session id mismatch"
    return True, "ok"


def redact_session_data(session: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a session export dict with secrets redacted.

    Runs every string in the export through the force-mode redaction pass
    (``agent.redact.redact_sensitive_text``) — message content, tool-call
    arguments, AND session-level fields outside ``messages``/``segments``.
    The auto-generated session title quotes the first user message, so a
    credential pasted into that message rides into the title and, before
    this walked the whole dict, straight into the export header in
    plaintext despite ``--redact`` (#90361). Force mode ignores the user's
    global ``security.redact_secrets`` preference — an explicit ``--redact``
    export must never emit raw secrets anywhere in the file.
    """
    from agent.redact import redact_sensitive_text

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact_sensitive_text(value, force=True)
        if isinstance(value, list):
            return [_clean(v) for v in value]
        if isinstance(value, dict):
            # Keys are strings too — a credential-shaped key (a misbehaving
            # JSON emitter writing {"sk-...": value}) must not pass through
            # verbatim when the user asked for --redact (review on #90361).
            out: dict[Any, Any] = {}
            for key, item in value.items():
                masked = _clean(key) if isinstance(key, str) else key
                # Two keys can mask to the same text ("***"): number the later ones so no
                # entry is silently dropped; the suffix carries no secret bytes.
                unique, n = masked, 2
                while unique in out:
                    unique, n = f"{masked}#{n}", n + 1
                out[unique] = _clean(item)
            return out
        return value

    return _clean(session)


def _export_dir(output_dir: Path | str) -> Path:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_session_markdown(
    session: dict[str, Any], output_dir: Path | str, *, fmt: str = "md", force: bool = False
) -> Path:
    """Write a Markdown/QMD export file and return its path."""
    path = _export_dir(output_dir) / safe_session_filename(session, fmt=fmt)
    if path.exists() and not force:
        raise FileExistsError(str(path))
    path.write_text(render_session_markdown(session, fmt=fmt), encoding="utf-8")
    return path


def append_manifest_entry(output_dir: Path | str, session: dict[str, Any], path: Path | str, *, fmt: str) -> Path:
    entry = {
        "session_id": _session_id(session),
        "lineage_session_ids": session.get("lineage_session_ids") or [_session_id(session)],
        "path": str(Path(path)),
        "format": fmt,
        "message_count": _message_count(session),
        "sha256": file_sha256(path),
        "exported_at": time.time(),
    }
    manifest = _export_dir(output_dir) / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
