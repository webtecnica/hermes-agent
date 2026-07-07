"""
Lazy dependency installer for opt-in Hermes Agent backends.

Many Hermes features (Mistral TTS, ElevenLabs TTS, Honcho memory, Bedrock,
Slack, Matrix, etc.) require Python packages that not every user needs. The
historical approach was to bundle them all under ``pyproject.toml`` extras
(``hermes-agent[all]``) and install them eagerly at setup time. That has
two problems:

1. **Fragility.** When one extra's transitive dependency becomes
   unavailable on PyPI (quarantined for malware, yanked, broken upload),
   the *entire* ``[all]`` resolve fails and fresh installs silently fall
   back to a stripped tier — losing 10+ unrelated extras at once.

2. **Bloat.** A user who only ever talks to one provider pulls hundreds
   of packages they will never import.

The lazy-install pattern fixes both. Backends call :func:`ensure` at the
top of their first-import path. If the deps are missing, ``ensure`` checks
the ``security.allow_lazy_installs`` config flag (default true) and runs
a venv-scoped pip install. If the user has explicitly disabled lazy
installs, ``ensure`` raises :class:`FeatureUnavailable` with a clear
remediation hint pointing at ``hermes tools`` or the manual pip command.

Security model:

* **Venv-scoped by default.** Installs target ``sys.executable`` in the
  active venv. We never touch the system Python.
* **Durable-target mode (immutable images).** When the deployment seals the
  agent's own venv (the Docker image sets ``HERMES_DISABLE_LAZY_INSTALLS=1``
  and makes ``/opt/hermes`` read-only), setting
  ``HERMES_LAZY_INSTALL_TARGET`` redirects lazy installs to a writable
  directory on the durable data volume (e.g. ``/opt/data/lazy-packages``).
  That directory is **appended to the end of ``sys.path``** — never
  prepended, never exported via ``PYTHONPATH`` — so the agent's own
  site-packages wins every name collision. A package installed this way can
  only ADD new importable modules; it can never shadow, downgrade, or break
  a module the core already ships. The worst a bad/incompatible backend
  package can do is fail to import and report itself unavailable — the agent
  core stays healthy. This is the structural guarantee that a lazily
  installed package cannot brick Hermes, which is what made it safe to seal
  the venv in the first place. Compiled-wheel safety across image rebuilds
  is handled by an ABI/Python-version stamp on the target subdir (see
  :func:`_ensure_target_ready`).
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_DEPS` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs in BOTH modes. Users in
  restricted networks or strict security postures can pin themselves to
  whatever was installed at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

Adding a new backend:

1. Add an entry to :data:`LAZY_DEPS` with the package specs.
2. At the top of the backend module's import path, call
   ``ensure("feature.name")`` inside a try/except that converts
   :class:`FeatureUnavailable` to a useful runtime error.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Allowlist of lazy-installable backends.
#
# Keys are dot-separated feature names ("namespace.backend"). Values are
# tuples of pip-installable specs that match the corresponding extra in
# pyproject.toml. The framework enforces that only specs from this map
# can flow into the pip install command.
# =============================================================================


LAZY_DEPS: dict[str, tuple[str, ...]] = {
    # ─── Inference providers ───────────────────────────────────────────────
    # Native Anthropic SDK — needed when provider=anthropic (not via
    # OpenRouter / aggregators which use the openai SDK).
    "provider.anthropic": ("anthropic==0.87.0",),  # CVE-2026-34450, CVE-2026-34452
    # AWS Bedrock provider
    "provider.bedrock": ("boto3==1.42.89",),
    # Google Vertex AI provider — OAuth2 token minting for the Gemini
    # OpenAI-compatible endpoint. Only loaded when provider=vertex is selected;
    # google-auth is NOT in [all] so plain installs don't carry it.
    "provider.vertex": ("google-auth==2.55.1",),
    # Microsoft Foundry — Entra ID auth (managed identity, workload identity,
    # service principal, az login, VS Code, azd, PowerShell). Only loaded
    # when model.auth_mode=entra_id is selected; key-based azure-foundry
    # users never pay this import.
    "provider.azure_identity": ("azure-identity==1.25.3",),

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": ("exa-py==2.10.2",),
    "search.firecrawl": ("firecrawl-py==4.17.0",),
    "search.parallel": ("parallel-web==0.4.2",),
    "search.ddgs": ("ddgs==9.14.4",),

    # ─── TTS providers ─────────────────────────────────────────────────────
    # Pinned to exact versions to match pyproject.toml's no-ranges policy
    # (see comment at top of [project.dependencies]). When bumping, update
    # both this map AND the corresponding extra in pyproject.toml.
    #
    # mistralai pin tracks the `mistral` extra in pyproject.toml. PyPI
    # quarantined the project 2026-05-12 (malicious 2.4.6, Mini Shai-Hulud);
    # 2.4.6 was removed and clean releases resumed (2.4.7, 2.4.8). Voxtral
    # STT + TTS share the same SDK.
    "tts.mistral": ("mistralai==2.4.8",),
    "tts.edge": ("edge-tts==7.2.7",),
    "tts.elevenlabs": ("elevenlabs==1.59.0",),

    # ─── Speech-to-text providers ──────────────────────────────────────────
    "stt.mistral": ("mistralai==2.4.8",),
    "stt.faster_whisper": (
        "faster-whisper==1.2.1",
        "sounddevice==0.5.5",
        "numpy==2.4.3",
    ),

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": ("fal-client==0.13.1",),

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": ("honcho-ai==2.0.1",),
    "memory.hindsight": ("hindsight-client==0.6.1",),
    # supermemory + mem0 are opt-in cloud memory providers with their own
    # SDKs. On the published Docker image the agent venv is sealed
    # (HERMES_DISABLE_LAZY_INSTALLS=1) and lazy installs are redirected to the
    # durable target — so, like honcho/hindsight, these MUST go through
    # ensure() to be installable there. Without an allowlist entry + an
    # ensure() call at the import site, the SDK never installs on a hosted
    # instance and the provider silently reports itself unavailable.
    "memory.supermemory": ("supermemory==3.50.0",),
    "memory.mem0": ("mem0ai==2.0.10",),

    # ─── Messaging platforms (lazy-installable on demand) ──────────────────
    "platform.telegram": ("python-telegram-bot[webhooks]==22.6",),
    # brotlicffi gives aiohttp a working 2-arg Decompressor.process() for
    # Discord CDN's Brotli-encoded attachments. Without it, aiohttp falls
    # back to google's `Brotli` package (1-arg API), and any .txt/.md/.doc
    # uploaded to the Discord gateway fails to decode at att.read() with
    # "Can not decode content-encoding: br" — see #12511 / #15744.
    "platform.discord": (
        "discord.py[voice]==2.7.1",
        "brotlicffi==1.2.0.1",
        # discord.py pulls aiohttp transitively (>=3.7.4,<4) as its HTTP
        # backbone. Pin the patched floor here too so the lazy Discord path
        # can't keep an already-installed vulnerable aiohttp satisfying that
        # range — mirrors the messaging extra and platform.slack.
        "aiohttp==3.14.1",  # CVE-2026-34513/34518/34519/34520/34525 + 34993(RCE)/47265
    ),
    "platform.slack": (
        "slack-bolt==1.27.0",
        "slack-sdk==3.40.1",
        "aiohttp==3.14.1",  # CVE-2026-34513/34518/34519/34520/34525 + 34993(RCE)/47265
    ),
    "platform.matrix": (
        "mautrix[encryption]==0.21.0",
        "aiosqlite==0.22.1",
        "asyncpg==0.31.0",
        "aiohttp-socks==0.11.0",
        # mautrix (aiohttp>=3,<4) and aiohttp-socks (aiohttp>=3.10.0) only cap
        # aiohttp transitively, so a vulnerable already-installed aiohttp still
        # satisfies both — pin the patched floor here too, like platform.discord.
        "aiohttp==3.14.1",  # CVE-2026-34513/34518/34519/34520/34525 + 34993(RCE)/47265
    ),
    "platform.dingtalk": (
        "dingtalk-stream==0.24.3",
        "alibabacloud-dingtalk==2.2.42",
        "qrcode==7.4.2",
    ),
    "platform.feishu": (
        "lark-oapi==1.5.3",
        "qrcode==7.4.2",
    ),
    # WeCom callback-mode adapter — parses untrusted XML POST bodies. Pulls
    # defusedxml only; aiohttp/httpx are core dependencies of every messaging
    # adapter and ship via `platform.discord` / `platform.slack` / etc.
    "platform.wecom_callback": ("defusedxml==0.7.1",),
    # Microsoft Teams adapter — microsoft-teams-apps pulls a heavy tree
    # (microsoft-teams-api/cards/common, dependency-injector, msal). Lazy-
    # installed on demand like every other messaging platform; also exposed
    # as the `teams` extra in pyproject for packagers / explicit installs.
    "platform.teams": ("microsoft-teams-apps==2.0.13.4", "aiohttp==3.14.1"),  # aiohttp 3.14.1: CVE-2026-34993(RCE)/47265 + 34513/34518/34519/34520/34525

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": ("modal==1.3.4",),
    "terminal.daytona": ("daytona==0.155.0",),

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": (
        "google-api-python-client==2.194.0",
        "google-auth-oauthlib==1.3.1",
        "google-auth-httplib2==0.3.1",
    ),
    "skill.youtube": ("youtube-transcript-api==1.2.4",),

    # ─── Tools ─────────────────────────────────────────────────────────────
    # ACP adapter (VS Code / Zed / JetBrains integration)
    "tool.acp": ("agent-client-protocol==0.9.0",),
    # Dashboard (`hermes dashboard`)
    "tool.dashboard": (
        "fastapi==0.133.1",
        "uvicorn[standard]==0.41.0",
        "starlette==1.0.1",  # CVE-2026-48710 (BadHost) — keep lazy-install in sync with pyproject [web]
        "python-multipart==0.0.27",  # FastAPI UploadFile/Form for streaming uploads (NS-501)
    ),
    # Vision image-resize recovery (Pillow). Pillow is now a CORE dependency
    # (pyproject `dependencies`), so this entry is a belt-and-suspenders fallback
    # for stripped/source-build installs that somehow dropped it. The vision
    # call site uses prompt=False so it can never raise a blocking input()
    # prompt mid-session (#40490).
    "tool.vision": ("Pillow==12.2.0",),
    # Computer Use (cua-driver) — the MCP client SDK used to spawn and talk
    # to the cua-driver process over stdio. Matches the `mcp` / `computer-use`
    # extras in pyproject.toml. The one-liner installer pulls this in via
    # `[all]`; lazy-installing here covers lean / partial / broken-extra
    # installs so computer_use never dead-ends on `No module named 'mcp'`.
    "tool.computer_use": (
        "mcp==1.26.0",
        "starlette==1.0.1",  # CVE-2026-48710 — keep in sync with pyproject [computer-use]
    ),
    # HF Agent Trace Viewer upload (hermes trace upload / /upload-trace).
    "tool.trace_upload": ("huggingface-hub==1.2.3",),
}