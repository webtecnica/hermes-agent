"""Label-prefixed typed field dumps: ``field password: CONCEALED = "…"`` (#157615).

Both anchored assignment passes need the key at line start (behind the optional line-number
gutter) or right after ``export``. A serializer's struct dump puts a short label in front of
the field name and spells the declared type between the name and the value, so the value
reached every redaction surface verbatim. The pattern added for this shape keeps label, key
and type and masks only the value; the gate stays on the typed assignment (``: TYPE =``), so
prose that merely contains a keyword is left byte-identical.
"""

import pytest

from agent.redact import redact_sensitive_text

PW = "synthpass.d'X9"
TOK = "synthtok123"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _both_modes(text: str) -> list[str]:
    return [redact_sensitive_text(text, force=True),
            redact_sensitive_text(text, force=True, file_read=True, secret_file=True)]


def _assert_masked(text: str, secret: str) -> None:
    for out in _both_modes(text):
        assert secret not in out, out


def _assert_unchanged(text: str) -> None:
    assert _both_modes(text) == [text, text]


class TestLabelPrefixedDumpMasked:
    """The leak from the issue: label, then ``key: TYPE = value``, all mid-line."""

    @pytest.mark.parametrize("text, secret", [
        ('field password: CONCEALED = "synthpass.d\'X9"', PW),
        ("field token: STRING = 'synthtok123'", TOK),
        ("field token: STRING = synthtok123", TOK),
        # A label may be several words, and a rendered line-number gutter may precede it.
        ('db field token: STRING = "synthtok123"', TOK),
        ('5|      field password: CONCEALED = "synthpass.d\'X9"', PW),
        # Declared container / array types are the same dump shape.
        ('field token: Optional[str] = "synthtok123"', TOK),
        # A single-word angle value is not a docs placeholder, so it is still a value.
        ("field token: STRING = <Zq8x>", "Zq8x"),
    ])
    def test_value_masked(self, text, secret):
        _assert_masked(text, secret)

    def test_label_key_and_type_preserved(self):
        assert redact_sensitive_text(f'field password: CONCEALED = "{PW}"', force=True) == \
            'field password: CONCEALED = "***"'
        assert redact_sensitive_text(f'field token: STRING = "{TOK}"', force=True) == \
            'field token: STRING = "***"'

    def test_quotes_preserved(self):
        assert redact_sensitive_text(f"field token: STRING = '{TOK}'", force=True) == \
            "field token: STRING = '***'"

    def test_nonreusable_mode_uses_the_sentinel(self):
        out = redact_sensitive_text(f'field password: CONCEALED = "{PW}"',
                                    force=True, file_read=True, secret_file=True)
        assert out == 'field password: CONCEALED = "«redacted-secret»"'


class TestExistingControls:
    """The #16413 controls keep their old answer."""

    @pytest.mark.parametrize("text, secret", [
        ("PASSWORD=synthpass", "synthpass"),
        ("password: synthpass", "synthpass"),
    ])
    def test_still_masked(self, text, secret):
        _assert_masked(text, secret)

    def test_already_masked_uppercase_stays_a_marker(self):
        # ``***`` is the mask itself: it is neither re-masked nor expanded.
        assert _both_modes("PASSWORD=***") == ["PASSWORD=***", "PASSWORD=***"]


class TestProseUntouched:
    """A keyword inside prose is not a dump: no ``: TYPE =``, no match."""

    @pytest.mark.parametrize("text", [
        "Secretary: J.Smith",
        "tokenizer: cl100k_base",
        "token: discussão",
        "author=Smith",
        "The Secretary: J.Smith signed the token: discussão",
    ])
    def test_unchanged(self, text):
        _assert_unchanged(text)


class TestNonCredentialsKept:
    """No value, no ``=``, or a placeholder/sample value: nothing to mask."""

    @pytest.mark.parametrize("text", [
        "field password:",
        "field token: STRING",
        'field password: CONCEALED = ""',
        'field password: CONCEALED = "${API_KEY}"',
        'field token: STRING = "${OPENAI_API_KEY}"',
        'field password: CONCEALED = "{{ .Values.dbPassword }}"',
        'field token: STRING = "true"',
        'field token: STRING = "<your token>"',
        # Unquoted docs/env placeholders are left whole; masking only the first word would
        # leave a mangled ``*** token>`` behind.
        "field token: STRING = <your token>",
        "field password: CONCEALED = ${API_KEY}",
        'field password: CONCEALED = "$DB_PASSWORD"',
    ])
    def test_unchanged(self, text):
        _assert_unchanged(text)
