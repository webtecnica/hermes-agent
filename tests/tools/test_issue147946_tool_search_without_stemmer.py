"""Issue #147946: ``tool_search`` must survive without the optional ``snowballstemmer`` wheel.

``tools/tool_search_catalog.py`` used to import ``snowballstemmer`` unconditionally at module
scope, so an install without the wheel failed at ``import tools.tool_search_catalog`` — the
whole bridge surface (search/describe/call), not just stemming. The dependency is declared
only for ``python_version >= '3.14'``, while ``requires-python`` is ``>=3.11,<3.15``: the
older interpreters exist for the pre-PM updater, whose lazy imports must not explode on a
missing dependency.

Each "without the package" case runs in a *fresh* interpreter with the name shadowed by a
``sys.meta_path`` finder that raises ``ImportError`` (nothing is uninstalled), which is the
real import path a lean install takes — no in-process module surgery.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Shadow the package exactly like a lean install: the finder raises ImportError before any
# path entry (including site-packages) can answer. A fresh interpreter imports the module
# for the first time afterwards, so no previous import can mask the failure.
_BLOCK_SNOWBALLSTEMMER = '''
import sys


class _BlockSnowballstemmer:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "snowballstemmer" or fullname.startswith("snowballstemmer."):
            raise ImportError(
                "No module named 'snowballstemmer' (simulated absence via sys.meta_path)"
            )
        return None


sys.meta_path.insert(0, _BlockSnowballstemmer())
'''


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _last_json(stdout: str) -> dict:
    return json.loads(stdout.strip().splitlines()[-1])


def test_module_imports_and_stem_falls_back_to_unstemmed_tokens():
    """Import succeeds, ``_stem`` returns the token unchanged, and the fallback is cached."""
    result = _run_fresh(_BLOCK_SNOWBALLSTEMMER + '''
import json
import logging

records = []


class _Collect(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())


handler = _Collect()
logger = logging.getLogger("tools.tool_search_catalog")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import tools.tool_search_catalog as catalog

# Distinct tokens so every _stem call misses the lru_cache; the warning must not repeat.
for i in range(25):
    catalog._stem(f"issue{i}")
catalog._stem("issues")
catalog._stem("issues")  # lru_cache hit
info = catalog._stem.cache_info()
stem_issues = catalog._stem("issues")
stem_issue = catalog._stem("issue")
stem_creating = catalog._stem("creating")
tokenize = catalog._tokenize("Post issues to Slack")
print(json.dumps({
    "has_snowball": catalog._HAS_SNOWBALL,
    "stem_issues": stem_issues,
    "stem_issue": stem_issue,
    "stem_creating": stem_creating,
    "tokenize": tokenize,
    "thread_stemmer": getattr(catalog._thread_local, "stemmer", "unset"),
    "stem_cache_misses": info.misses,
    "stem_cache_hits": info.hits,
    "fallback_warnings": [m for m in records if "unstemmed tokens" in m],
}))
''')
    assert result.returncode == 0, result.stderr
    out = _last_json(result.stdout)

    assert out["has_snowball"] is False
    # Fallback: the token comes back exactly as tokenized, not stemmed and not dropped.
    assert out["stem_issues"] == "issues"
    assert out["stem_issue"] == "issue"
    assert out["stem_creating"] == "creating"
    assert out["tokenize"] == ["post", "issues", "to", "slack"]
    # The unavailable stemmer is cached per thread instead of retried on every token.
    assert out["thread_stemmer"] is None
    assert out["stem_cache_misses"] == 26  # 25 distinct + "issues" once; the repeat hit
    assert out["stem_cache_hits"] == 1
    # Observable, and exactly once for 25+ cache-missing tokens.
    assert len(out["fallback_warnings"]) == 1
    assert "snowballstemmer" in out["fallback_warnings"][0]


def test_tool_search_still_finds_tool_by_textual_term_without_stemmer():
    """The bridge keeps answering: a textual query still returns the matching tool."""
    result = _run_fresh(_BLOCK_SNOWBALLSTEMMER + '''
import json

from tools.registry import registry
from tools.tool_search import ToolSearchConfig, dispatch_tool_search


def td(name, desc):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


defs = [
    td("nr_incident_create", "Create a new incident and assign it to the on-call rotation."),
    td("nr_page_send", "Send a page to the primary on-call engineer."),
]
for defn in defs:
    registry.register(
        name=defn["function"]["name"],
        handler=lambda args, **kw: "{}",
        schema=defn,
        toolset="mcp-nr",
    )

out = json.loads(dispatch_tool_search(
    {"queries": ["incident", "on-call"]},
    current_tool_defs=defs,
    config=ToolSearchConfig.from_raw({}),
))
print(json.dumps({
    "has_snowball": __import__("tools.tool_search_catalog", fromlist=["_HAS_SNOWBALL"])._HAS_SNOWBALL,
    "groups": out["results"],
}))
''')
    assert result.returncode == 0, result.stderr
    out = _last_json(result.stdout)

    assert out["has_snowball"] is False
    matches = {group["query"]: group["matches"] for group in out["groups"]}
    assert "nr_incident_create" in matches["incident"]
    assert "nr_incident_create" in matches["on-call"]


def test_stemming_still_applies_when_snowballstemmer_is_available():
    """With the wheel present the morphological match is unchanged (base behavior)."""
    pytest.importorskip("snowballstemmer")

    from tools.tool_search_catalog import _HAS_SNOWBALL, _stem, _tokenize

    assert _HAS_SNOWBALL is True
    assert _stem("issues") == "issu"
    assert _stem("creating") == "creat"
    assert _tokenize("issues") == _tokenize("issue")
