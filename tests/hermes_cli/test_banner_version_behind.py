"""``hermes --version`` must name the commit that is actually running.

When a checkout on the ``main`` channel is *behind* ``origin/main`` the banner
used to show only the upstream SHA, so the running commit appeared nowhere.
The label must carry ``behind N`` and the local HEAD, and the ahead/current
cases must keep their existing shape.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import banner


@pytest.fixture
def label_for(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: root)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {})
    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda project_root: {})
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda *a, **k: "main")
    monkeypatch.setattr(banner, "get_version_info", lambda: SimpleNamespace(derived_version="0.0.0"))

    def _label(state):
        monkeypatch.setattr(banner, "get_git_banner_state", lambda: state)
        return banner.format_banner_version_label()

    return _label


def test_ahead_shows_upstream_local_and_carried_commits(label_for):
    label = label_for({"upstream": "deadbeef", "local": "cafebabe", "ahead": 2, "behind": 0})
    assert "upstream deadbeef" in label
    assert "local cafebabe" in label
    assert "+2 carried commits" in label


def test_behind_shows_behind_count_and_running_head(label_for):
    label = label_for({"upstream": "deadbeef", "local": "cafebabe", "ahead": 0, "behind": 3})
    assert "upstream deadbeef" in label
    assert "behind 3" in label
    # The commit that is actually running must always be visible.
    assert "local cafebabe" in label


def test_current_shows_only_upstream(label_for):
    label = label_for({"upstream": "deadbeef", "local": "deadbeef", "ahead": 0, "behind": 0})
    assert "upstream deadbeef" in label
    assert "local" not in label
    assert "behind" not in label


@pytest.mark.parametrize("state", [None, {}])
def test_missing_state_falls_back_to_base_label(label_for, state):
    label = label_for(state)
    assert label.startswith("Hermes Agent v0.0.0")
    assert "upstream" not in label
