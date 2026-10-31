"""The four assignment patterns grow with the text, not faster (#147257).

Each shape below is a run of key-like characters with a secret word in it plus a
suffix that opens the assignment passes. Nothing in it is a credential; a
minified blob, a generated fixture or a path list looks like this. On the base
commit the costs were:

* ``_CFG_DOTTED_RE`` CUBIC — ``a.token`` repeated: ~1 s at 2 KiB, 7-10 s at 4 KiB,
  no finish at 8 KiB in 45 s. The second alternative had a backtrackable ``*``
  before the word, a backtrackable ``*`` after it, then ``\\.`` and a possessive
  tail that scanned the rest of the run: every (word x dot) combination scanned
  the run.
* ``_CFG_ANCHORED_RE`` and ``_YAML_ASSIGN_RE`` QUADRATIC — the leading class was
  backtracked to each occurrence of the word and the tail scanned the run each
  time, to fail on the missing ``=``/``:``.
* ``_SHELL_VAR_REF`` QUADRATIC — ``\\w*`` followed by ``[^}]*`` accepted the same
  characters, so an unclosed ``${`` was retried for every split of the run.

These are measured at 4 KiB, where the base commit already needed seconds: a
ceiling with a wide margin, like the other redactor guards, and a growth check
that fails if the cost per byte keeps climbing with the size.
"""

import time

import pytest

from agent.redact import redact_sensitive_text

TETO_S = 0.5


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _texto(run: str, suffix: str, kib: int, prefix: str = "") -> str:
    tamanho = kib * 1024
    return prefix + (run * (tamanho // len(run) + 1))[:tamanho] + suffix


def _medir(texto: str) -> float:
    inicio = time.perf_counter()
    redact_sensitive_text(texto, force=True)
    return time.perf_counter() - inicio


FORMATOS = {
    "dotted_a.token": ("a.token", " k=v", ""),
    "dotted_token.": ("token.", " k=v", ""),
    "anchored_token": ("token", " k=v", ""),
    "anchored_token_a": ("token_a", " k=v", ""),
    "yaml_token": ("token", " k:v", ""),
    "shell_var_unclosed": ("a", "", "API_KEY=${A"),
}


class TestAssignmentPatternsStayLinear:
    @pytest.mark.parametrize("nome", sorted(FORMATOS))
    def test_4kib_abaixo_do_teto(self, nome):
        run, suffix, prefix = FORMATOS[nome]
        decorrido = _medir(_texto(run, suffix, 4, prefix))
        assert decorrido < TETO_S, f"{nome}: {decorrido * 1000:.0f} ms em 4 KiB"

    @pytest.mark.parametrize("nome", sorted(FORMATOS))
    def test_custo_por_byte_nao_cresce_ao_dobrar(self, nome):
        run, suffix, prefix = FORMATOS[nome]
        t2 = _medir(_texto(run, suffix, 2, prefix))
        t4 = _medir(_texto(run, suffix, 4, prefix))
        # Dobrar o texto pode custar ate ~2,5x; o cubico custava 6x a 10x.
        assert t4 < max(0.02, t2 * 3), f"{nome}: 2 KiB {t2 * 1000:.0f} ms -> 4 KiB {t4 * 1000:.0f} ms"

    @pytest.mark.parametrize("texto", [
        "app.api.key=abcdef123456",
        "spring.datasource.password=abcdef123456",
        "s3.secret-key=abcdef123456",
        "token.value=abcdef123456",
        "password=hunter2",
        "export API_KEY=abcdef123456",
        "42|  db_password: abcdef123456",
        "api_key: abcdef123456",
        "API_KEY=${HOME}/x",
        "API_KEY=/home/user/.ssh/id_rsa",
    ])
    def test_formatos_reais_continuam_mascarados(self, texto):
        saida = redact_sensitive_text(texto, force=True)
        if "${HOME}" in texto or "id_rsa" in texto:
            assert saida == texto  # lookup de variavel / caminho: nao e credencial
        else:
            assert "abcdef123456" not in saida and "hunter2" not in saida, saida

    @pytest.mark.parametrize("texto", [
        "the token value in prose stays",
        "Secretary: J.Smith",
        "tokenizer: cl100k_base",
        "note: it's fine",
    ])
    def test_prosa_e_chaves_benignas_ficam_intactas(self, texto):
        assert redact_sensitive_text(texto, force=True) == texto
