"""Performance guards for the ENV / config / YAML assignment passes (#147257).

Every one of these passes used to scan the WHOLE text with a backtracking
pattern, once per pass, on every call — a multi-MB tool result (a config dump, a
PR diff, a compaction payload) took seconds and pinned the GIL while it did. The
fix gates both the pass and the text: a linear keyword scan decides whether the
pass can match at all, and above ``_ASSIGN_LINE_SCAN_MIN_CHARS`` the pass only
walks the lines that carry a secret word.

Two things are pinned here: the wall clock (a ceiling with a wide margin, in the
style of the other redactor guards) and — the important one — that the gated
path produces byte-identical output to the whole-text path it replaced.
"""

import time

import pytest

from agent.redact import _redact_assignments, redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _dump_realista(linhas: int = 4000, blob_kib: int = 2048) -> str:
    """Saida de terminal realista: log, ~100 blocos de credencial e um payload opaco.

    O corpus e gerado no teste de proposito: versionar 2 MB de fixture seria pior.
    Sem token com prefixo de fornecedor, para isolar os passes de atribuicao.
    """
    partes = []
    for i in range(linhas):
        partes.append(f"2026-10-31T12:00:{i % 60:02d} INFO worker_{i} iniciou o ciclo {i} em 12.3ms")
        if i % 40 == 0:
            partes.append(f"OPENAI_API_KEY_{i}=valorOpaco{i:04d}abcdefghijklmnopqrstuvwxyz")
            partes.append(f"  api_key: valorOpaco{i:04d}abcdefghijklmnopqrstuvwxyz0123")
            partes.append(f"  db_password: {'p' * 28}{i:04d}")
    partes.append("payload_opaco:")
    partes.append("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZn" * (blob_kib * 1024 // 44))
    return "\n".join(partes) + "\n"


MASKS_ESPERADAS = 300  # 100 blocos x (env + api_key + db_password)


class TestAssignmentPassesStayFast:
    def test_dump_de_2mb_nao_leva_segundos(self):
        """Os passes de atribuicao num dump de ~2,3 MB.

        Base (c419cf7c86): 737 ms. Com o fix do #147240 e sem esta otimizacao:
        884 ms. Depois: ~59 ms nesta VPS.
        """
        texto = _dump_realista()
        assert len(texto) > 2_000_000

        inicio = time.perf_counter()
        saida = _redact_assignments(texto, mask_nonreusable=True)
        decorrido = time.perf_counter() - inicio

        assert saida.count("«redacted") == MASKS_ESPERADAS
        assert decorrido < 0.30, f"{decorrido * 1000:.0f} ms para {len(texto) / 1024:.0f} KiB"

    def test_payload_opaco_sem_palavra_de_credencial_nao_e_varrido(self):
        texto = ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVph" * 60_000) + "\n"
        assert len(texto) > 2_000_000

        inicio = time.perf_counter()
        assert _redact_assignments(texto, mask_nonreusable=True) == texto
        decorrido = time.perf_counter() - inicio

        assert decorrido < 0.05, f"{decorrido * 1000:.0f} ms"

    def test_chamada_publica_grande_fica_abaixo_do_teto(self):
        texto = _dump_realista()
        inicio = time.perf_counter()
        saida = redact_sensitive_text(texto, file_read=True, secret_file=True)
        decorrido = time.perf_counter() - inicio
        assert saida.count("«redacted") >= MASKS_ESPERADAS
        assert decorrido < 2.0, f"{decorrido * 1000:.0f} ms"


def _corpo(caso: str) -> str:
    if caso == "dump":
        return _dump_realista()
    if caso == "chave_no_fim_da_linha":
        return "OPENAI_API_KEY\n= valorNaLinhaSeguinte0123456789\n" * 400
    if caso == "valor_na_linha_seguinte":
        return "OPENAI_API_KEY=\nvalorNaLinhaSeguinte0123456789\n" * 400
    if caso == "yaml_aspas_e_gutter":
        return ('42|  db_password: "hunter2-real"  # producao\n'
                'api_key: "«redacted-secret»"\n'
                'secret: plainvalue0123456789abcdef\n') * 300
    if caso == "cabecalho":
        return "x-api-key: headerValue0123456789\napi-key: outro\n" * 600
    raise AssertionError(caso)


CASOS = ["dump", "chave_no_fim_da_linha", "valor_na_linha_seguinte", "yaml_aspas_e_gutter", "cabecalho"]


class TestGatedPathMatchesWholeTextPath:
    """O caminho por linha tem de ser identico ao passe sobre o texto inteiro."""

    @pytest.mark.parametrize("caso", CASOS, ids=CASOS)
    def test_mesma_saida_com_e_sem_o_corte_por_linha(self, monkeypatch, caso):
        from agent import redact

        texto = _corpo(caso)
        monkeypatch.setattr(redact, "_ASSIGN_LINE_SCAN_MIN_CHARS", 10**9)
        inteiro = _redact_assignments(texto, mask_nonreusable=True)

        monkeypatch.setattr(redact, "_ASSIGN_LINE_SCAN_MIN_CHARS", 1)
        por_linha = _redact_assignments(texto, mask_nonreusable=True)

        assert por_linha == inteiro

    @pytest.mark.parametrize("texto,segredo", [
        ("OPENAI_API_KEY\n= valorNaLinhaSeguinte0123456789\n", "valorNaLinhaSeguinte0123456789"),
        ("OPENAI_API_KEY=\nvalorNaLinhaSeguinte0123456789\n", "valorNaLinhaSeguinte0123456789"),
        ("API_KEY =\n\tvalorComTab0123456789\n", "valorComTab0123456789"),
    ], ids=["chave_no_fim", "valor_na_seguinte", "com_tab"])
    def test_atribuicao_quebrada_em_duas_linhas_continua_redigida(self, monkeypatch, texto, segredo):
        """O ``\\s*=\\s*`` do padrao ENV aceita a quebra de linha; o corte por linha
        re-varre a linha candidata junto com a seguinte para nao perder esse caso."""
        from agent import redact

        texto_grande = texto * 300
        assert len(texto_grande) > 8192
        monkeypatch.setattr(redact, "_ASSIGN_LINE_SCAN_MIN_CHARS", 1)
        resultado = _redact_assignments(texto_grande, mask_nonreusable=True)
        assert segredo not in resultado
        assert resultado == _redact_assignments(texto_grande, mask_nonreusable=True)
