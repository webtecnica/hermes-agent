"""Affixed keys and prefixed credentials that used to come back in clear (#147240).

Both shapes are variants of a rule the redactor already had: the YAML and env
passes accept affixes around a secret word (``DB_PASSWORD=``, ``app.api.key=``),
and a vendor-prefixed credential is masked when it stands alone. The gaps were
in the *separator*:

* a JSON key that merely CONTAINS a secret word (``"API_TOKEN"``,
  ``"DB_PASSWORD"``, ``"client_secret"``) missed ``_JSON_FIELD_RE``, whose key
  side is a closed list of exact names;
* a prefixed credential preceded by ``-`` or ``_`` (``backup-sk-…``,
  ``backup_sk-…``, ``x-ghp_…``) missed ``_PREFIX_RE``, whose lookbehind rejected
  a token glued to a letter, a digit, ``-`` or ``_`` — i.e. the ordinary
  file-name and identifier positions.
"""

import pytest

from agent.redact import redact_sensitive_text

V = "q7Zr4Lm9Xc2Vb5Nw8Kd3Hf6J"
SK = "sk-" + V + "t1Pa0Sg7Ye4Uo2"
GH = "ghp_" + "Ab3dE5gH7jK9mN1pQ3sT5vX7zB9dF1hJ3kL5"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _both(text: str) -> list[str]:
    return [
        redact_sensitive_text(text, force=True),
        redact_sensitive_text(text, force=True, file_read=True, secret_file=True),
    ]


class TestJsonKeyContainingSecretWord:
    @pytest.mark.parametrize("key", [
        "API_TOKEN", "DB_PASSWORD", "client_secret", "ACCESS_TOKEN", "my_api_key", "refresh-token",
        "SESSION_TOKEN", "x-api-key", "secret_value", "AUTH_TOKEN",
    ])
    def test_chave_com_afixo_e_mascarada(self, key):
        for saida in _both(f'{{"{key}": "{V}"}}'):
            assert V not in saida, saida

    @pytest.mark.parametrize("key,value", [
        ("tokenizer", "cl100k_base"),
        ("secretary", "J.Smith"),
        ("name", V),
        ("author", "someone"),
        ("key", "value"),
    ])
    def test_chave_sem_palavra_de_credencial_fica_intacta(self, key, value):
        texto = f'{{"{key}": "{value}"}}'
        for saida in _both(texto):
            assert saida == texto

    def test_lista_fechada_continua_mascarando_valor_curto(self):
        # ``"token"`` is an exact name: it keeps masking even a short, non-opaque
        # value — the affixed pass must not change that (it runs second).
        assert V not in _both(f'{{"token": "{V}"}}')[0]


class TestPrefixedCredentialAfterSeparator:
    @pytest.mark.parametrize("texto,segredo", [
        (f"backup-{SK}.txt", SK),
        (f"backup_{SK}.txt", SK),
        (f"x-{GH}", GH),
        (f"backup_{GH}", GH),
        (SK, SK),
        (f"copy {SK}", SK),
        (f"/{SK}.txt", SK),
    ])
    def test_credencial_depois_de_hifen_ou_underscore(self, texto, segredo):
        for saida in _both(texto):
            assert segredo not in saida, saida

    def test_token_glued_em_palavra_continua_intacto(self):
        # A run of letters/digits around the whole token is still a single
        # identifier, not a credential: the lookbehind only stopped rejecting
        # ``-`` and ``_``.
        texto = "nosuchprefix" + SK
        for saida in _both(texto):
            assert SK in saida
