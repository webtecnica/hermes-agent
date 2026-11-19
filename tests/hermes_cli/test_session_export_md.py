import json
from pathlib import Path

import pytest

from hermes_cli.session_export_md import (
    append_manifest_entry,
    file_sha256,
    render_session_markdown,
    safe_session_filename,
    verify_export_file,
    write_session_markdown,
)

_SHA_FOOTER_PREFIX = "- SHA256 of exported body: `"
_PRIOR_EXPORT_FOOTER = _SHA_FOOTER_PREFIX + "a" * 64 + "`"


def _session(**overrides):
    data = {
        "id": "20260706_123456_abcd1234",
        "title": "Export Test",
        "source": "telegram",
        "model": "gpt-5.5",
        "billing_provider": "openai-codex",
        "cwd": "/tmp/project",
        "started_at": 1783331696.0,
        "last_active": 1783331705.0,
        "ended_at": 1783331710.0,
        "message_count": 3,
        "tool_call_count": 1,
        "archived": 0,
        "messages": [
            {"role": "user", "content": "Hello", "created_at": 1783331697.0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": "{\"command\": \"pwd\"}"}}
                ],
                "created_at": 1783331698.0,
            },
            {"role": "tool", "name": "terminal", "content": "output", "created_at": 1783331699.0},
        ],
    }
    data.update(overrides)
    return data




def test_safe_session_filename_is_deterministic_and_path_safe():
    filename = safe_session_filename(
        _session(id="20260706_123456_abcd1234", title="Bad / title: * ?"), fmt="qmd"
    )

    assert filename.startswith("20260706_123456_abcd1234-")
    assert filename.endswith(".qmd")
    assert "/" not in filename
    assert ":" not in filename
    assert "*" not in filename
    assert "?" not in filename






def test_verify_export_file_checks_count_and_sha(tmp_path):
    session = _session()
    path = write_session_markdown(session, tmp_path)

    ok, reason = verify_export_file(path, session)
    assert ok is True
    assert reason == "ok"

    path.write_text(path.read_text(encoding="utf-8").replace("Hello", "Tampered"), encoding="utf-8")
    ok, reason = verify_export_file(path, session)
    assert ok is False
    assert "sha256" in reason


def _self_referential_session(content: str) -> dict:
    return _session(
        messages=[{"role": "user", "content": content, "created_at": 1783331697.0}],
        message_count=1,
    )


@pytest.mark.parametrize(
    "content",
    [
        "__SHA256_PLACEHOLDER__",
        _PRIOR_EXPORT_FOOTER,
        "a plain message that quotes no digest markup",
    ],
    ids=["literal-placeholder", "quoted-prior-footer", "plain"],
)
def test_verify_accepts_bodies_that_quote_digest_markup(tmp_path, content):
    # #157707: a session that quotes the export digest markup is still a valid export.
    # The renderer used a fixed sentinel and replaced it across the whole body, and the
    # verifier anchored on the first footer line, so both self-referential shapes failed
    # their own verification with "sha256 mismatch".
    session = _self_referential_session(content)
    out_dir = tmp_path / "exports"
    path = write_session_markdown(session, out_dir)
    text = path.read_text(encoding="utf-8")

    digest_lines = [line for line in text.splitlines() if line.startswith(_SHA_FOOTER_PREFIX)]
    if content.startswith(_SHA_FOOTER_PREFIX):
        # The quoted footer survives verbatim; the generated footer is a second line.
        assert len(digest_lines) == 2
        assert digest_lines[0] == content
        assert digest_lines[1] != content
    else:
        assert len(digest_lines) == 1

    ok, reason = verify_export_file(path, session)
    assert (ok, reason) == (True, "ok")

    # The manifest still records the whole-file digest of the final export.
    manifest = append_manifest_entry(out_dir, session, path, fmt="md")
    entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["sha256"] == file_sha256(path)


def test_verify_rejects_tampered_self_referential_body(tmp_path):
    session = _self_referential_session(_PRIOR_EXPORT_FOOTER)
    path = write_session_markdown(session, tmp_path / "exports")
    assert verify_export_file(path, session) == (True, "ok")

    # One byte changed inside the quoted footer the body carries — outside the real
    # footer — must still be caught. A fix that simply ignores every digest-looking
    # line in the body would pass verification here and must not.
    text = path.read_text(encoding="utf-8")
    tampered = text.replace(_PRIOR_EXPORT_FOOTER, _SHA_FOOTER_PREFIX + "b" * 64 + "`", 1)
    assert tampered != text
    path.write_text(tampered, encoding="utf-8")

    ok, reason = verify_export_file(path, session)
    assert ok is False
    assert "sha256" in reason
