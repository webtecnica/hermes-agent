"""Tests for the ntfy platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_ntfy`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests in the same process.

Most tests target the adapter class directly. The plugin-shape tests
(``register()``, ``_env_enablement``, ``_standalone_send``, registry
presence) replace the core-file grep tests from the original PR — the
ntfy adapter no longer modifies ``gateway/config.py``, ``gateway/run.py``,
``cron/scheduler.py``, ``toolsets.py``, etc.  Everything routes through
the ``platform_registry``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_ntfy = load_plugin_adapter("ntfy")

NtfyAdapter = _ntfy.NtfyAdapter
check_requirements = _ntfy.check_requirements
validate_config = _ntfy.validate_config
is_connected = _ntfy.is_connected
register = _ntfy.register
_env_enablement = _ntfy._env_enablement
_standalone_send = _ntfy._standalone_send
DEFAULT_SERVER = _ntfy.DEFAULT_SERVER
DEDUP_WINDOW_SECONDS = _ntfy.DEDUP_WINDOW_SECONDS
DEDUP_MAX_SIZE = _ntfy.DEDUP_MAX_SIZE
MAX_MESSAGE_LENGTH = _ntfy.MAX_MESSAGE_LENGTH


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform("ntfy")."""
    from gateway.config import Platform
    p = Platform("ntfy")
    assert p.value == "ntfy"
    # Identity stability — repeated lookups return the same pseudo-member
    assert Platform("ntfy") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestNtfyRequirements:



    def test_is_connected_from_extra(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert is_connected(PlatformConfig(enabled=True, extra={"topic": "t"})) is True
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------


class TestNtfyAdapterInit:


    def test_topic_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "env-topic")
        config = PlatformConfig(enabled=True, extra={})
        adapter = NtfyAdapter(config)
        assert adapter._topic == "env-topic"




    def test_token_read_from_env(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOKEN", "env-token")
        config = PlatformConfig(enabled=True, extra={"topic": "t"})
        adapter = NtfyAdapter(config)
        assert adapter._token == "env-token"


# ---------------------------------------------------------------------------
# 4. Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:

    def _make_adapter(self, token=""):
        config = PlatformConfig(enabled=True, extra={"topic": "t", "token": token})
        return NtfyAdapter(config)

    def test_no_token_returns_empty_dict(self):
        adapter = self._make_adapter(token="")
        assert adapter._auth_headers() == {}

    def test_bearer_token_for_plain_token(self):
        adapter = self._make_adapter(token="myapitoken")
        headers = adapter._auth_headers()
        assert headers["Authorization"] == "Bearer myapitoken"


# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def _make_adapter(self):
        return NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

    def test_first_message_not_duplicate(self):
        adapter = self._make_adapter()
        assert adapter._dedup.is_duplicate("msg-1") is False

    def test_second_occurrence_is_duplicate(self):
        adapter = self._make_adapter()
        adapter._dedup.is_duplicate("msg-1")
        assert adapter._dedup.is_duplicate("msg-1") is True


# ---------------------------------------------------------------------------
# 6. connect() / disconnect()
# ---------------------------------------------------------------------------


class TestConnect:


    def test_connect_starts_stream_task(self, monkeypatch):
        monkeypatch.setattr(_ntfy, "HTTPX_AVAILABLE", True)
        config = PlatformConfig(enabled=True, extra={"topic": "hermes-test"})
        adapter = NtfyAdapter(config)

        with patch.object(adapter, "_run_stream", new_callable=AsyncMock):
            with patch.object(_ntfy, "httpx") as mock_httpx:
                mock_httpx.AsyncClient.return_value = MagicMock()
                result = _run(adapter.connect())

        assert result is True
        assert adapter._stream_task is not None
        adapter._stream_task.cancel()
        try:
            _run(adapter._stream_task)
        except (asyncio.CancelledError, Exception):
            pass


    def test_disconnect_cancels_stream_task(self):
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))

        async def _hang():
            await asyncio.sleep(0.2)

        loop = asyncio.get_event_loop()
        adapter._stream_task = loop.create_task(_hang())
        adapter._http_client = AsyncMock()
        adapter._running = True

        _run(adapter.disconnect())
        assert adapter._stream_task is None


# ---------------------------------------------------------------------------
# 7. send()
# ---------------------------------------------------------------------------


class TestSend:

    def _make_adapter(self, topic="hermes-in", publish_topic="", token="", markdown=False):
        extra: dict = {"topic": topic, "token": token}
        if publish_topic:
            extra["publish_topic"] = publish_topic
        if markdown:
            extra["markdown"] = True
        return NtfyAdapter(PlatformConfig(enabled=True, extra=extra))


    def test_send_posts_to_publish_topic(self):
        adapter = self._make_adapter(topic="hermes-in", publish_topic="hermes-out")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "abc123"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send("hermes-in", "Hello ntfy!"))
        assert result.success is True
        assert result.message_id == "abc123"

        posted_url = mock_client.post.call_args[0][0]
        assert posted_url.endswith("/hermes-out")

    def test_send_falls_back_to_subscribe_topic(self):
        adapter = self._make_adapter(topic="hermes-in")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send("hermes-in", "Hello!"))
        assert result.success is True
        posted_url = mock_client.post.call_args[0][0]
        assert posted_url.endswith("/hermes-in")

    def test_send_uses_metadata_publish_topic(self):
        adapter = self._make_adapter(topic="hermes-in")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        adapter._http_client = mock_client

        result = _run(adapter.send(
            "hermes-in", "Hi!", metadata={"publish_topic": "override-out"}
        ))
        assert result.success is True
        posted_url = mock_client.post.call_args[0][0]
        assert posted_url.endswith("/override-out")


    def test_send_handles_timeout(self):
        adapter = self._make_adapter(topic="hermes-in")

        class _FakeTimeout(Exception):
            pass

        fake_httpx = MagicMock()
        fake_httpx.TimeoutException = _FakeTimeout

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_FakeTimeout("timed out"))
        adapter._http_client = mock_client

        with patch.object(_ntfy, "httpx", fake_httpx):
            result = _run(adapter.send("hermes-in", "Hello!"))

        assert result.success is False
        assert "timeout" in result.error.lower()




# ---------------------------------------------------------------------------
# 8. Inbound message processing (identity invariant — security-critical)
# ---------------------------------------------------------------------------


class TestOnMessage:

    def _make_adapter(self):
        return NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "hermes-in"}))

    def test_message_dispatched_to_handler(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)

        event = {
            "id": "evt-001",
            "event": "message",
            "topic": "hermes-in",
            "message": "Hello from ntfy",
            "time": 1700000000,
        }
        _run(adapter._on_message(event))
        assert len(calls) == 1
        assert calls[0].text == "Hello from ntfy"

    def test_empty_message_skipped(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "x", "event": "message", "topic": "t", "message": "", "time": None
        }))
        assert calls == []

    def test_duplicate_message_skipped(self):
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        event = {"id": "dup-1", "event": "message", "topic": "hermes-in", "message": "hi", "time": None}
        _run(adapter._on_message(event))
        _run(adapter._on_message(event))
        assert len(calls) == 1

    def test_own_tagged_message_skipped(self):
        """An incoming event carrying the adapter's echo tag is the agent's
        own reply echoed back by ntfy — it must not be dispatched, otherwise
        the agent replies to itself forever (issue #34447)."""
        adapter = self._make_adapter()
        calls = []

        async def handler(event):
            calls.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message({
            "id": "echo-1",
            "event": "message",
            "topic": "hermes-in",
            "message": "my own reply",
            "tags": [_ntfy._ECHO_TAG],
            "time": None,
        }))
        assert calls == []


# ---------------------------------------------------------------------------
# 9. _env_enablement() — env-only auto-config
# ---------------------------------------------------------------------------


class TestEnvEnablement:

    def test_returns_none_without_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        assert _env_enablement() is None


    def test_markdown_truthy_values(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        for val in ("true", "1", "yes", "TRUE"):
            monkeypatch.setenv("NTFY_MARKDOWN", val)
            assert _env_enablement()["markdown"] is True


    def test_home_channel_override(self, monkeypatch):
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        monkeypatch.setenv("NTFY_HOME_CHANNEL", "alerts")
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "alerts"
        assert seed["home_channel"]["name"] == "Alerts Channel"


# ---------------------------------------------------------------------------
# 10. _standalone_send() — out-of-process cron delivery
# ---------------------------------------------------------------------------


class TestStandaloneSend:

    def test_errors_without_topic(self, monkeypatch):
        monkeypatch.delenv("NTFY_TOPIC", raising=False)
        monkeypatch.delenv("NTFY_PUBLISH_TOPIC", raising=False)
        pconfig = MagicMock()
        pconfig.extra = {}
        result = _run(_standalone_send(pconfig, "", "hello"))
        assert "error" in result


    def test_emits_echo_tag_header(self, monkeypatch):
        """Out-of-process cron / send_message deliveries also carry the echo
        tag, so a gateway subscribed to the same topic skips them too."""
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in"}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "id-99"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            _run(_standalone_send(pconfig, "hermes-in", "hi"))

        headers = mock_client.post.call_args[1]["headers"]
        assert headers.get("X-Tags") == _ntfy._ECHO_TAG


# ---------------------------------------------------------------------------
# 10b. Notification title (#156887)
# ---------------------------------------------------------------------------
#
# A publish with no X-Title makes ntfy fall back to the topic name as the
# notification heading, which on a private/randomly-suffixed topic is an opaque
# token (`alfred-404f6688a123`) on the user's lock screen. Both call sites now
# resolve a heading: metadata["title"] -> extra["title"] -> NTFY_HOME_CHANNEL_NAME.


class TestPublishTitle:

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("NTFY_HOME_CHANNEL_NAME", raising=False)
        monkeypatch.setenv("NTFY_TOPIC", "hermes-in")

    @staticmethod
    def _adapter(extra=None, token="", markdown=False):
        merged = {"topic": "hermes-in", **(extra or {})}
        if token:
            merged["token"] = token
        if markdown:
            merged["markdown"] = True
        return NtfyAdapter(PlatformConfig(enabled=True, extra=merged))

    @staticmethod
    def _client():
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "id-1"}
        client = AsyncMock()
        client.post = AsyncMock(return_value=mock_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    def _send_headers(self, adapter, client, **kwargs):
        _run(adapter.send("hermes-in", "hi", **kwargs))
        return client.post.call_args[1]["headers"]

    def _standalone_headers(self, pconfig, client):
        with patch.object(_ntfy, "httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = client
            result = _run(_standalone_send(pconfig, "hermes-in", "hi"))
        assert result["success"] is True
        return client.post.call_args[1]["headers"]

    # -- send() ------------------------------------------------------------

    def test_send_metadata_title_becomes_x_title(self):
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client, metadata={"title": "Cron: daily report"})

        assert headers["X-Title"] == "Cron: daily report"
        assert headers["X-Tags"] == _ntfy._ECHO_TAG

    def test_send_metadata_title_wins_over_extra_and_home_name(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        adapter = self._adapter(extra={"title": "Configured"})
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client, metadata={"title": "Explicit"})

        assert headers["X-Title"] == "Explicit"

    def test_send_falls_back_to_extra_title(self):
        adapter = self._adapter(extra={"title": "Configured"})
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client)

        assert headers["X-Title"] == "Configured"

    def test_send_falls_back_to_home_channel_name(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client)

        assert headers["X-Title"] == "Alerts Channel"

    def test_send_without_any_title_sends_no_x_title(self):
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client)

        assert "X-Title" not in headers
        # Nothing else moved: today's header order is unchanged without a title.
        assert list(headers) == ["Content-Type", "X-Tags"]

    def test_send_title_strips_control_characters(self):
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(
            adapter, client, metadata={"title": "Cron\r\njob\tfailed\nX-Evil: 1"})

        assert headers["X-Title"] == "Cron job failed X-Evil: 1"
        assert not any(ch in headers["X-Title"] for ch in "\r\n\t")

    def test_send_non_ascii_title_degrades_without_raising(self):
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client, metadata={"title": "Aviso café 日本語"})

        assert headers["X-Title"].isascii()
        assert headers["X-Title"].startswith("Aviso cafe")
        assert headers["X-Tags"] == _ntfy._ECHO_TAG

    def test_send_non_string_title_falls_through(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        adapter = self._adapter()
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client, metadata={"title": {"nested": "obj"}})

        assert headers["X-Title"] == "Alerts Channel"

    def test_send_preserves_header_order_with_title(self):
        adapter = self._adapter(token="tok")
        client = self._client()
        adapter._http_client = client

        headers = self._send_headers(adapter, client, metadata={"title": "Heads up"})

        assert list(headers) == ["Authorization", "Content-Type", "X-Tags", "X-Title"]

    # -- _standalone_send() ------------------------------------------------

    def test_standalone_extra_title_becomes_x_title(self):
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in", "title": "Cron: daily report"}
        client = self._client()

        headers = self._standalone_headers(pconfig, client)

        assert headers["X-Title"] == "Cron: daily report"
        assert headers["X-Tags"] == _ntfy._ECHO_TAG

    def test_standalone_falls_back_to_home_channel_name(self, monkeypatch):
        monkeypatch.setenv("NTFY_HOME_CHANNEL_NAME", "Alerts Channel")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in"}
        client = self._client()

        headers = self._standalone_headers(pconfig, client)

        assert headers["X-Title"] == "Alerts Channel"

    def test_standalone_without_any_title_sends_no_x_title(self):
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in"}
        client = self._client()

        headers = self._standalone_headers(pconfig, client)

        assert "X-Title" not in headers
        assert list(headers) == ["Content-Type", "X-Tags"]

    def test_standalone_preserves_header_order_with_title_and_markdown(self, monkeypatch):
        monkeypatch.setenv("NTFY_MARKDOWN", "true")
        pconfig = MagicMock()
        pconfig.extra = {"topic": "hermes-in", "title": "Heads up"}
        client = self._client()

        headers = self._standalone_headers(pconfig, client)

        assert list(headers) == ["Content-Type", "X-Tags", "X-Markdown", "X-Title"]
        assert headers["X-Tags"] == _ntfy._ECHO_TAG


# ---------------------------------------------------------------------------
# 11. register() — plugin-side metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 12. Robustness — token hygiene + fatal-state propagation
# ---------------------------------------------------------------------------


class TestTokenHygiene:
    """``_build_auth_header`` must strip pasted-token whitespace; pasted
    tokens often carry trailing newlines that break the Authorization line."""

    def test_trailing_whitespace_stripped(self):
        assert _ntfy._build_auth_header("  tok123  ") == {"Authorization": "Bearer tok123"}


    def test_whitespace_only_returns_empty(self):
        assert _ntfy._build_auth_header("   \n  ") == {}


class TestFatalErrorPropagation:
    """When the stream hits 401/404, the adapter must transition to the
    ``fatal`` state via ``_set_fatal_error`` so the gateway's runtime
    status reflects reality instead of staying 'connected'."""

    def test_401_sets_fatal_unauthorized(self):
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={"topic": "t"}))
        adapter._http_client = MagicMock()

        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status_code = 401
        # async-context-manager flavor for httpx.stream
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        adapter._http_client.stream = MagicMock(return_value=mock_cm)

        fake_httpx = MagicMock()
        fake_httpx.Timeout = MagicMock()
        with patch.object(_ntfy, "httpx", fake_httpx):
            with pytest.raises(_ntfy._FatalStreamError):
                _run(adapter._consume_stream("https://ntfy.example/t/json", {}))

        assert adapter.has_fatal_error is True
        assert adapter._fatal_error_code == "ntfy_unauthorized"
        assert adapter._fatal_error_retryable is False




# ---------------------------------------------------------------------------
# 13. Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# __init__'s server/topic/publish_topic, _env_enablement's topic/server/
# publish_topic/markdown/home_channel, and check_requirements/validate_config/
# is_connected's topic reads, all previously read raw os.getenv
# unconditionally (only NTFY_TOKEN was already scoped). Under multiplex,
# os.environ holds the DEFAULT profile's YAML-to-env bridge output -- a
# secondary profile with its own (different or absent) ntfy config would
# silently subscribe to / publish on the default profile's topic, or get
# auto-enabled using the default profile's topic entirely. Mirrors the
# LINE/Buzz/SimpleX fix for #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("NTFY_TOPIC", "default-topic")
    monkeypatch.setenv("NTFY_SERVER_URL", "https://default.example.com")
    monkeypatch.setenv("NTFY_PUBLISH_TOPIC", "default-out")


class TestMultiplexProfileScope:

    def test_secondary_extra_wins_over_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """The secondary profile's own config.yaml extra is authoritative,
        not the default profile's bridged topic/server/publish_topic."""
        multiplex_scope()
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "topic": "profile-topic",
                "server": "https://profile.example.com",
                "publish_topic": "profile-out",
            },
        )
        adapter = NtfyAdapter(cfg)
        assert adapter._topic == "profile-topic"
        assert adapter._server == "https://profile.example.com"
        assert adapter._publish_topic == "profile-out"

    def test_secondary_missing_keys_fail_closed(
        self, multiplex_scope, default_profile_env
    ):
        """Keys absent from the profile's own scope must NOT borrow the
        default profile's bridged env values -- that would silently
        subscribe/publish on the wrong topic."""
        multiplex_scope()
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter._topic == ""
        assert adapter._server == DEFAULT_SERVER
        assert adapter._publish_topic == ""
        # Nor may the registry auto-enable ntfy for this profile off the default's topic.
        assert _env_enablement() is None
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False

