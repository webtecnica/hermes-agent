from agent.context_compressor import _build_anchor_index, _LEAN_ANCHOR_HEADING


def test_build_anchor_index_captures_media_and_data_paths():
    turns = [
        {
            "role": "user",
            "content": (
                "Please process the media at ~/.hermes/state.db and /var/log/syslog.log. "
                "Also check output at /home/u/proj/out/clip_orbit360.mp4 and generated image "
                "out/balcony_sunset_20261023.png as well as ./data/dataset.csv."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Saved backup to ~/.hermes/data/backup.bak and systemd service at /etc/systemd/system/agent.service. "
                "I also examined src/agent.py and docs/index.md for context."
            ),
        },
    ]

    index = _build_anchor_index(turns)
    assert _LEAN_ANCHOR_HEADING in index
    assert "media-paths:" in index
    assert "files:" in index

    # Verify media-paths contains the expected media/data artefacts
    for path in [
        "~/.hermes/state.db",
        "/var/log/syslog.log",
        "/home/u/proj/out/clip_orbit360.mp4",
        "out/balcony_sunset_20261023.png",
        "./data/dataset.csv",
        "~/.hermes/data/backup.bak",
        "/etc/systemd/system/agent.service",
    ]:
        assert path in index, f"Expected {path} in anchor index: {index}"

    # Verify standard code files remain in files:
    for code_file in ["src/agent.py", "docs/index.md"]:
        assert code_file in index, f"Expected {code_file} in anchor index: {index}"
