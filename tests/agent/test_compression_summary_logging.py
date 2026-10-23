import logging
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def test_compression_summary_dispatched_and_served_logging(caplog):
    compressor = ContextCompressor(
        model="primary/gpt-4o",
        provider="primary-provider",
        threshold_percent=0.50,
        quiet_mode=True,
        config_context_length=100_000,
    )

    fake_response = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "Summary text"}}],
    }

    def fake_call_llm(**kwargs):
        route_info = kwargs.get("route_info")
        if isinstance(route_info, dict):
            route_info["provider"] = "mock-aux-provider"
            route_info["model"] = "mock-aux-model"
        return fake_response

    # Test 1: auxiliary.compression has a configured model
    with (
        patch("agent.context_compressor.call_llm", side_effect=fake_call_llm),
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={"model": "configured/aux-model", "provider": "aux-prov"}),
        caplog.at_level(logging.INFO, logger="agent.context_compressor"),
    ):
        compressor._call_summary_llm("test prompt", 0.0)

    dispatched_records = [
        r.getMessage() for r in caplog.records if "Compression summary call dispatched:" in r.getMessage()
    ]
    served_records = [
        r.getMessage() for r in caplog.records if "Compression summary call served:" in r.getMessage()
    ]

    assert len(dispatched_records) == 1
    assert "model=configured/aux-model" in dispatched_records[0]
    assert "model=primary/gpt-4o" not in dispatched_records[0]

    assert len(served_records) == 1
    assert "aux_provider=mock-aux-provider" in served_records[0]
    assert "aux_model=mock-aux-model" in served_records[0]

    # Test 2: auxiliary.compression has no configured model, falls back to self.model
    caplog.clear()
    with (
        patch("agent.context_compressor.call_llm", side_effect=fake_call_llm),
        patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}),
        caplog.at_level(logging.INFO, logger="agent.context_compressor"),
    ):
        compressor._call_summary_llm("test prompt", 0.0)

    dispatched_records2 = [
        r.getMessage() for r in caplog.records if "Compression summary call dispatched:" in r.getMessage()
    ]
    assert len(dispatched_records2) == 1
    assert "model=primary/gpt-4o" in dispatched_records2[0]
