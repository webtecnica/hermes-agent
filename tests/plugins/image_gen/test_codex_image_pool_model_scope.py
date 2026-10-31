"""Image availability through the Codex OAuth pool is scoped to the image model (#146984).

``CredentialPool.select(model=...)`` keeps rate-limit benches per model, so a chat model the
account cannot use says nothing about the same credential's standing for image generation. The
image provider resolved the pool *unscoped* though, and an unscoped selection stays deliberately
conservative: ``model_cooldown_until(entry, None)`` treats any active model cooldown as blocking.
A chat model benched on every pooled account (a cohort withdrawal or a usage window) therefore
reported the Codex image provider as unavailable while every credential was healthy for images.

The pool is real but entirely local: rows seeded into a throwaway ``HERMES_HOME`` ``auth.json`` with
synthetic tokens and cooldown metadata. No network, no live credentials.
"""

from __future__ import annotations

import base64
import importlib
import json
import time

import pytest

codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")

# The chat model that is out of allowance/entitlement on every pooled account, and the image model
# the provider's requests actually carry (a catalog value, never a literal repeated here).
CHAT_MODEL = "gpt-5.3-codex"
IMAGE_MODEL = codex_plugin.API_MODEL

_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _seed_pool(tmp_path, monkeypatch, cooldowns: dict[str, float]) -> None:
    """Three healthy Codex OAuth rows in a throwaway HERMES_HOME, all carrying *cooldowns*.

    Every row carries them so the pool has no uncooled fallback: that is the reported shape (a
    chat model the whole plan refuses), and it is also the only shape where an unscoped
    ``select()`` comes back empty.
    """
    root = tmp_path / "hermes-root"
    root.mkdir()
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    rows = [
        {
            "id": f"cred-{index}", "label": f"acct-{index}", "auth_type": "oauth", "priority": index,
            "source": "manual", "access_token": f"tok-synthetic-{index}",
            "refresh_token": f"rt-synthetic-{index}", "expires_at_ms": 4_000_000_000_000,
            "base_url": "https://chatgpt.com/backend-api/codex", "model_cooldowns": dict(cooldowns),
        }
        for index in range(3)
    ]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": rows}}))


@pytest.fixture
def chat_cooldown(tmp_path, monkeypatch):
    """Every credential is benched for an unrelated *chat* model, and healthy otherwise."""
    _seed_pool(tmp_path, monkeypatch, {CHAT_MODEL: time.time() + 3600})


@pytest.fixture
def image_cooldown(tmp_path, monkeypatch):
    """Every credential is benched for the *image* model itself: the negative control."""
    _seed_pool(tmp_path, monkeypatch, {IMAGE_MODEL: time.time() + 3600})


def test_a_chat_model_cooldown_does_not_bench_image_availability(chat_cooldown):
    assert codex_plugin.OpenAICodexImageGenProvider().is_available() is True


def test_the_image_model_cooldown_still_benches_image_availability(image_cooldown):
    assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False


def test_unscoped_selection_keeps_treating_any_model_cooldown_as_blocking(chat_cooldown):
    """The scope is fixed at the image call site: ``select(model=None)`` stays conservative."""
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    assert pool.select() is None
    assert pool.select(model=CHAT_MODEL) is None
    # ...while the image model reaches an entry that is eligible for it.
    assert pool.select(model=IMAGE_MODEL).id == "cred-0"


def test_availability_scopes_the_pool_to_the_model_the_request_sends(monkeypatch):
    """Drift guard: the selection scope must be the ``model`` the image request body carries."""
    from agent import auxiliary_client

    seen: dict[str, object] = {}

    def _fake_resolve(*, model=None):
        seen["model"] = model
        return None, ""

    monkeypatch.setattr(auxiliary_client, "_resolve_codex_credential_and_base", _fake_resolve)
    assert codex_plugin.OpenAICodexImageGenProvider().is_available() is False
    _path, body = codex_plugin._build_image_request(prompt="a red cube", size="1024x1024", quality="low")
    assert seen["model"] == body["model"] == IMAGE_MODEL


def test_generate_resolves_the_pool_token_despite_a_chat_model_cooldown(chat_cooldown, monkeypatch):
    """Same scope on the request path: availability and the token actually sent agree."""
    used: dict[str, object] = {}

    def _fake_post(token, *, prompt, size, quality, input_images=None, base_url):
        used["token"] = token
        used["base_url"] = base_url
        return {"data": [{"b64_json": _b64_png()}], "quality": "medium", "size": "1024x1024"}

    monkeypatch.setattr(codex_plugin, "_post_image_request", _fake_post)
    result = codex_plugin.OpenAICodexImageGenProvider().generate("a red cube")
    assert result["success"] is True, result
    assert used["token"] == "tok-synthetic-0"
    assert used["base_url"] == "https://chatgpt.com/backend-api/codex"
