"""#159745: the failed-turn notices follow ``display.language``.

The three assistant rows that close a failed turn used to be fixed English strings. They are now
catalog keys (``gateway.errors.failed_turn_notice`` / ``...partial_failed_turn_notice`` /
``...automation_interrupted_notice``) resolved where they are written. English must stay
byte-identical — those exact bytes identify turn boundaries already persisted on disk — a configured
language must win, and legacy English boundary rows must still be recognised under another language.
No gateway process is started: a temp ``HERMES_HOME`` with ``display.language`` plus a user locale
overlay is the whole fixture.
"""

from __future__ import annotations

import inspect
import re

import hermes_yaml as yaml
import pytest

from agent import i18n
from agent.turn_failure_copy import (
    AUTOMATION_INTERRUPTED_NOTICE,
    FAILED_TURN_DISPLAY_KIND,
    FAILED_TURN_NOTICE,
    PARTIAL_FAILED_TURN_NOTICE,
    failed_turn_notice,
    untyped_failed_turn_display_kind,
)

# The shipped English copy written out, not read back from the constants: these bytes are the
# on-disk identity of every boundary row already persisted, so the catalog must reproduce them.
EN_FAILED = "Your request was not processed. Send it again if you still want me to carry it out."
EN_PARTIAL = (
    "This turn did not complete. Some actions may already have run; verify their effects "
    "before resending."
)
EN_AUTOMATION = (
    "An automated delivery was interrupted and was not run again. Some actions may already "
    "have run; verify their effects."
)

ES_ERRORS = {
    "failed_turn_notice": (
        "Tu solicitud no se procesó. Envíala de nuevo si aún quieres que la lleve a cabo."
    ),
    "partial_failed_turn_notice": (
        "Este turno no se completó. Puede que algunas acciones ya se hayan ejecutado; "
        "verifica sus efectos antes de reenviar."
    ),
    "automation_interrupted_notice": (
        "Una entrega automatizada se interrumpió y no se volvió a ejecutar. Puede que algunas "
        "acciones ya se hayan ejecutado; verifica sus efectos."
    ),
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp profile home; the bundled catalogs still come from this checkout."""
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.delenv("HERMES_BUNDLED_LOCALES", raising=False)
    i18n.reset_language_cache()
    yield home
    i18n.reset_language_cache()


def _configure_language(home, lang: str) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump({"display": {"language": lang}}), encoding="utf-8"
    )
    i18n.reset_language_cache()


def _write_es_overlay(home) -> None:
    (home / "locales" / "es.yaml").write_text(
        yaml.safe_dump({"gateway": {"errors": ES_ERRORS}}, allow_unicode=True), encoding="utf-8"
    )
    i18n.reset_language_cache()


def test_english_copy_is_byte_identical(home):
    """(b) en resolves to exactly the text that identifies boundaries already on disk."""
    _configure_language(home, "en")

    assert failed_turn_notice([]) == EN_FAILED
    assert failed_turn_notice([{"role": "tool", "content": "ok"}]) == EN_PARTIAL
    assert i18n.t("gateway.errors.failed_turn_notice") == EN_FAILED
    assert i18n.t("gateway.errors.partial_failed_turn_notice") == EN_PARTIAL
    assert i18n.t("gateway.errors.automation_interrupted_notice") == EN_AUTOMATION
    # The module constants stay importable and stay the English default.
    assert (FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE, AUTOMATION_INTERRUPTED_NOTICE) == (
        EN_FAILED,
        EN_PARTIAL,
        EN_AUTOMATION,
    )


def test_configured_language_localizes_failed_turn_notice(home):
    """(a) the notice comes from the catalog of the configured language, partial still when a tool ran."""
    _write_es_overlay(home)
    _configure_language(home, "es")
    assert i18n.get_language() == "es"

    assert failed_turn_notice([]) == ES_ERRORS["failed_turn_notice"]
    assert failed_turn_notice([]) != EN_FAILED
    # A tool (or an assistant tool call) ran: the hedge copy, also localized.
    assert failed_turn_notice([{"role": "tool", "content": "ok"}]) == ES_ERRORS["partial_failed_turn_notice"]
    assert failed_turn_notice([{"role": "assistant", "tool_calls": [{"id": "1"}]}]) == (
        ES_ERRORS["partial_failed_turn_notice"]
    )


def test_gateway_writer_resolves_through_the_catalog(home):
    """(a) the gateway's own writer path emits catalog text too, and never the fixed constant."""
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    _write_es_overlay(home)
    _configure_language(home, "es")

    plain = GatewayTurnPersistenceMixin._hmwa_failed_turn_notice(
        object(), {"messages": [{"role": "user", "content": "hi"}], "history_offset": 0}
    )
    with_tool = GatewayTurnPersistenceMixin._hmwa_failed_turn_notice(
        object(),
        {
            "messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "ok"}],
            "history_offset": 0,
        },
    )
    assert plain == ES_ERRORS["failed_turn_notice"]
    assert with_tool == ES_ERRORS["partial_failed_turn_notice"]

    # Every other failed-turn write in this module goes through the same catalog keys.
    source = inspect.getsource(inspect.getmodule(GatewayTurnPersistenceMixin))
    assert not re.search(r"\b(?:PARTIAL_)?FAILED_TURN_NOTICE\b(?!_KEY)", source)


def test_legacy_english_rows_stay_boundaries_under_another_language(home):
    """(c) rows written in English before the catalogs keep their display kind under es."""
    _write_es_overlay(home)
    _configure_language(home, "es")

    for legacy in (EN_FAILED, EN_PARTIAL, EN_AUTOMATION):
        assert untyped_failed_turn_display_kind("assistant", legacy) == FAILED_TURN_DISPLAY_KIND
    # Whitespace-insensitive, as before.
    assert untyped_failed_turn_display_kind("assistant", f"  {EN_FAILED}\n") == FAILED_TURN_DISPLAY_KIND
    # A boundary written in the active language is recognised too.
    assert untyped_failed_turn_display_kind("assistant", ES_ERRORS["failed_turn_notice"]) == (
        FAILED_TURN_DISPLAY_KIND
    )
    # A real reply that quotes a notice is still a reply.
    assert untyped_failed_turn_display_kind("assistant", f"Quoting Hermes: {EN_FAILED}") is None
    assert untyped_failed_turn_display_kind("user", EN_FAILED) is None
