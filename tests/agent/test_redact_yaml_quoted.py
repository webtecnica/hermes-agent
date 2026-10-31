"""Redaction of a QUOTED YAML scalar under an unquoted key (#147240).

``db_password: "hunter2"`` escaped every pass: ``_YAML_ASSIGN_RE`` skips a value
that opens with a quote ("quoted values defer to the JSON rule") and
``_JSON_FIELD_RE`` only fires when the KEY itself is double-quoted. The value is
therefore the one config shape that reached the log, the prompt and the tool
transcript in cleartext, so these tests assert the mask on every surface that
runs the assignment passes, not just one.
"""

import pytest

from agent.redact import redact_sensitive_text, redact_terminal_output


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Keep HERMES_REDACT_SECRETS from a previous import from disabling the passes."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


# Surfaces that run the ENV/JSON/YAML assignment passes:
#   * plain text (log lines, prompts, transcript replay);
#   * content of a file the caller classified as secret-bearing (read_file on
#     ``$HERMES_HOME/config.yaml``, ``.env``), where the mask is the
#     non-reusable sentinel;
#   * ``cat``/``grep`` of that same file on the terminal side.
SURFACES = {
    "texto": lambda text: redact_sensitive_text(text),
    "leitura_secreta": lambda text: redact_sensitive_text(text, file_read=True, secret_file=True),
    "terminal": lambda text: redact_terminal_output(text, "cat ~/.hermes/config.yaml"),
}


class TestQuotedYamlValueIsRedacted:
    QUOTED_VALUES = [
        ('api_key: "s3cr-t-double-quoted-0123456789abc"\n', "s3cr-t-double-quoted-0123456789abc"),
        ("api_key: 's3cr-t-single-quoted-0123456789abc'\n", "s3cr-t-single-quoted-0123456789abc"),
        # ``#`` starts a YAML comment only OUTSIDE quotes: the whole scalar is secret.
        ('api_key: "s3cr#t-value-with-hash-abc123"\n', "s3cr#t-value-with-hash-abc123"),
        # Short/truncated key material: still a credential-shaped key.
        ('api_key: "sk-trunc"\n', "sk-trunc"),
        # Indented mapping (the shape a real config.yaml dump has).
        ('  db_password: "hunter2-REALSECRET99"\n', "hunter2-REALSECRET99"),
        # Spaces and an apostrophe inside double quotes are part of the value.
        ('client_secret: "my real secret # don\'t tell"\n', "my real secret # don't tell"),
        # A line-number gutter in front (read_file / grep -n / cat -n renderings).
        ('42|  api_key: "gutter-quoted-secret-0123456789"\n', "gutter-quoted-secret-0123456789"),
    ]

    @pytest.mark.parametrize("surface", sorted(SURFACES))
    @pytest.mark.parametrize("text,secret", QUOTED_VALUES)
    def test_quoted_value_never_reaches_the_surface(self, surface, text, secret):
        result = SURFACES[surface](text)
        assert secret not in result, result
        # The key itself stays readable, so the reader knows WHICH key was masked.
        assert "api_key" in result or "db_password" in result or "client_secret" in result

    def test_quotes_survive_so_the_line_stays_valid_yaml(self):
        yaml = pytest.importorskip("yaml")

        text = 'model: gpt-x\napi_key: "s3cr-t-quoted-0123456789abcdef"\nsecret: \'single-0123456789abcdef\'\n'
        result = redact_sensitive_text(text, file_read=True, secret_file=True)
        parsed = yaml.safe_load(result)
        # The line still parses, and the value is no longer the cleartext secret.
        assert parsed["api_key"] != "s3cr-t-quoted-0123456789abcdef"
        assert parsed["secret"] != "single-0123456789abcdef"
        # ``read_file`` renders the sentinel for a classified secret-bearing file.
        assert parsed["api_key"].startswith("«redacted")

    def test_trailing_yaml_comment_is_preserved(self):
        result = redact_sensitive_text('db_password: "hunter2-real"  # producao\n')
        assert "hunter2-real" not in result
        assert result.rstrip().endswith('# producao')


class TestQuotedYamlValueThatMustNotChange:
    # An already masked value must not be masked again: a second pass over a
    # sentinel only erases what the sentinel deliberately kept.
    ALREADY_MASKED = [
        'api_key: "***"\n',
        'api_key: \'***\'\n',
        'db_password: "«redacted-secret»"\n',
        'token: "«redacted:ghp_…»"\n',
    ]

    @pytest.mark.parametrize("text", ALREADY_MASKED)
    def test_already_masked_value_is_left_alone(self, text):
        assert redact_sensitive_text(text, file_read=True, secret_file=True) == text

    # Keys without a credential keyword, and prose that merely CONTAINS an
    # apostrophe (no closing quote), are ordinary content.
    BENIGN = [
        'note: "just a normal note here"\n',
        "note: it's fine\n",
        'author: "Jane Doe"\n',
        'tokenizer: "cl100k_base"\n',
        'model: "gpt-x"\n',
    ]

    @pytest.mark.parametrize("text", BENIGN)
    def test_non_credential_quoted_yaml_is_untouched(self, text):
        assert redact_sensitive_text(text, file_read=True, secret_file=True) == text

    def test_json_shaped_field_still_masks_once(self):
        # The JSON pass runs first; the quoted-YAML pass must not double-mask it.
        text = '{"api_key": "s3cr-t-json-0123456789abcdef"}\n'
        result = redact_sensitive_text(text, file_read=True, secret_file=True)
        assert "s3cr-t-json-0123456789abcdef" not in result
        assert result.count("«redacted") == 1, result

    def test_empty_quoted_scalar_is_left_alone(self):
        # Nothing to leak, and a rewritten empty scalar would stop being valid YAML.
        text = 'api_key: ""\n'
        assert redact_sensitive_text(text, file_read=True, secret_file=True) == text
