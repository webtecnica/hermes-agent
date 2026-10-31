"""Guards for the platform-SDK mocks installed by ``tests/e2e/conftest.py``.

The conftest fakes the ``telegram`` / ``discord`` / ``slack`` packages so the
adapters import without their SDKs installed. Faking is only correct when the
SDK really is absent: a real, installed SDK must never be swapped for a mock,
otherwise the e2e suite exercises a MagicMock while claiming to exercise the
real client interface (issue #147239).
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.e2e import conftest as e2e_conftest

# platform id -> module names the conftest fakes for it
FAKED_SDKS = {
    "telegram": ("telegram",),
    "discord": ("discord",),
    "slack": ("slack_bolt", "slack_sdk"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure(platform_name: str) -> None:
    {
        "telegram": e2e_conftest._ensure_telegram_mock,
        "discord": e2e_conftest._ensure_discord_mock,
        "slack": e2e_conftest._ensure_slack_mock,
    }[platform_name]()


@pytest.fixture
def platform_modules():
    """Snapshot the faked platform modules and restore them after the test.

    Tests here delete entries from :data:`sys.modules` to simulate "not imported
    yet" / "not installed"; without the restore the rest of the session would run
    without the conftest mocks it depends on.
    """
    def _owned(name: str) -> bool:
        return name.split(".", 1)[0] in FAKED_SDKS

    snapshot = {name: mod for name, mod in sys.modules.items() if _owned(name)}
    yield
    for name in [name for name in sys.modules if _owned(name)]:
        del sys.modules[name]
    sys.modules.update(snapshot)


def _drop(*names: str) -> None:
    for name in [name for name in sys.modules if name.split(".", 1)[0] in names]:
        del sys.modules[name]


def _write_fake_sdk(root: Path, name: str, body: str = "") -> None:
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text(
        body or f'"""Installed stand-in for the {name} SDK."""\n', encoding="utf-8"
    )


@pytest.mark.parametrize("platform_name", sorted(FAKED_SDKS))
def test_installed_sdk_is_not_replaced_by_the_mock(platform_name, tmp_path, monkeypatch, platform_modules):
    """An installed-but-not-yet-imported SDK must survive the conftest."""
    names = FAKED_SDKS[platform_name]
    for name in names:
        _write_fake_sdk(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop(*names)

    _ensure(platform_name)

    for name in names:
        assert name not in sys.modules, f"{name} is installed but conftest installed a fake"
        mod = importlib.import_module(name)
        assert Path(mod.__file__).resolve().is_relative_to(tmp_path)
        assert not isinstance(mod, MagicMock)


@pytest.mark.parametrize("platform_name", sorted(FAKED_SDKS))
def test_missing_sdk_still_gets_the_mock(platform_name, monkeypatch, platform_modules):
    """With no SDK on the path the fake is still installed — that is the legit case."""
    names = FAKED_SDKS[platform_name]
    _drop(*names)
    # Pretend nothing of this name is installed, whatever the test env has.
    monkeypatch.setattr(importlib.util, "find_spec", lambda *args, **kwargs: None)

    _ensure(platform_name)

    for name in names:
        assert isinstance(sys.modules[name], MagicMock), f"{name} was not mocked"


def test_telegram_mock_shape_is_unchanged(monkeypatch, platform_modules):
    """The fake must keep the attributes TelegramAdapter reads at import time."""
    _drop("telegram")
    monkeypatch.setattr(importlib.util, "find_spec", lambda *args, **kwargs: None)

    e2e_conftest._ensure_telegram_mock()

    mod = sys.modules["telegram"]
    assert mod.Update.ALL_TYPES == []
    assert mod.constants.ParseMode.MARKDOWN_V2 == "MarkdownV2"
    assert mod.ext.Application is not None
    assert mod.ext.filters is not None
    for submodule in ("telegram.constants", "telegram.ext", "telegram.request"):
        assert submodule in sys.modules


def test_stand_in_with_a_file_attribute_is_not_the_real_sdk(platform_modules):
    """The old check was ``hasattr(mod, "__file__")`` — an object-shape test.

    A stand-in that mimics a module (``__file__`` set) passed as "real library
    installed", so the conftest declined to mock while the cached module was a
    fake. Being a ``ModuleType`` is what actually makes it the real SDK here.
    """
    fake = MagicMock()
    fake.__file__ = "/nowhere/telegram/__init__.py"
    assert hasattr(fake, "__file__")
    sys.modules["telegram"] = fake

    assert e2e_conftest._real_module_available("telegram") is False


def test_imported_real_module_is_recognised(platform_modules):
    """A module that really lives on disk is the real thing."""
    assert e2e_conftest._real_module_available("json") is True
    sys.modules["telegram"] = importlib.import_module("json")
    assert e2e_conftest._real_module_available("telegram") is True


def test_real_import_error_is_not_masked_by_a_mock(tmp_path, platform_modules):
    """A broken install must fail through the adapter's own ImportError path.

    Masking the import with a MagicMock would flip ``TELEGRAM_AVAILABLE`` to True
    and hide the breakage; the conftest must leave the failing package alone.
    """
    _write_fake_sdk(tmp_path, "telegram", body='raise ImportError("broken telegram install")\n')
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"), PYTHONPATH=str(tmp_path))
    probe = (
        "import sys;"
        "import tests.e2e.conftest;"
        "import plugins.platforms.telegram.adapter as adapter;"
        "print('RESULT=' + __import__('json').dumps({"
        "'mocked': type(sys.modules.get('telegram')).__name__ == 'MagicMock',"
        "'available': adapter.TELEGRAM_AVAILABLE}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout.rsplit("RESULT=", 1)[1].strip())
    assert result == {"mocked": False, "available": False}
