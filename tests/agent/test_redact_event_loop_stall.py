"""The redactor runs synchronously on the event-loop thread (#147242).

``redact_sensitive_text`` makes no ``await`` and its passes are C-level regex
scans, so the GIL is held for the WHOLE call: while it runs, nothing else in the
process makes progress — an asyncio loop gets no tick, a gateway keeps no
heartbeat, and the bot looks hung. The only fix is to make the call short (the
assignment passes were the seconds-long part, see
``test_redact_assignment_perf.py``); this guard pins the loop stall with a
ticker coroutine, so a regression to a whole-text scan shows up as a starved
loop rather than as a merely slow call.
"""

import asyncio
import time

import pytest

from agent.redact import _redact_assignments, redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _dump_realista(linhas: int = 4000, blob_kib: int = 2048) -> str:
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


def _maior_travamento_do_loop(chamada, texto: str) -> float:
    """Segundos que o loop fica sem rodar enquanto ``chamada(texto)`` roda.

    A chamada e sincrona: o ticker so volta a rodar quando ela termina, entao a
    maior lacuna entre dois ticks e exatamente o travamento que o loop sofre.
    """

    async def _medir() -> float:
        pior = 0.0

        async def _ticker() -> None:
            nonlocal pior
            anterior = time.perf_counter()
            while True:
                await asyncio.sleep(0.002)
                agora = time.perf_counter()
                pior = max(pior, agora - anterior - 0.002)
                anterior = agora

        tarefa = asyncio.create_task(_ticker())
        await asyncio.sleep(0.02)  # deixa o ticker estabilizar antes da chamada
        pior = 0.0
        chamada(texto)
        # O ticker so registra a lacuna no tick SEGUINTE: sem esta espera o
        # travamento da chamada nem chegaria a ser medido.
        await asyncio.sleep(0.01)
        tarefa.cancel()
        try:
            await tarefa
        except asyncio.CancelledError:
            pass
        return pior

    return asyncio.run(_medir())


class TestEventLoopStall:
    def test_passes_de_atribuicao_nao_travam_o_loop(self):
        """2,3 MB pelos passes de atribuicao.

        Base (c419cf7c86): ~750 ms de travamento continuo — o loop nao roda uma
        vez sequer durante a chamada. Depois: ~60 ms.
        """
        texto = _dump_realista()
        travamento = _maior_travamento_do_loop(
            lambda t: _redact_assignments(t, mask_nonreusable=True), texto)
        assert travamento < 0.20, f"o loop ficou {travamento * 1000:.0f} ms sem rodar"

    def test_chamada_publica_grande_nao_trava_o_loop(self):
        texto = _dump_realista()
        travamento = _maior_travamento_do_loop(
            lambda t: redact_sensitive_text(t, file_read=True, secret_file=True), texto)
        assert travamento < 0.60, f"o loop ficou {travamento * 1000:.0f} ms sem rodar"

    def test_token_de_telegram_em_texto_grande_continua_mascarado(self):
        """O gate de forma do passe de Telegram nao pode pular um token real."""
        token = "bot123456789:" + "ABCDEfghij" + "0123456789abcdefghij"
        texto = _dump_realista() + f"\nchat: {token}\n"
        resultado = redact_sensitive_text(texto, force=True)
        assert "ABCDEfghij" not in resultado
        assert "123456789:***" in resultado
