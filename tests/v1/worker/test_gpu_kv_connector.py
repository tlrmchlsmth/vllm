# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
import vllm.v1.worker.kv_connector_model_runner_mixin as mixin_module
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.kv_connector_output import KVConnectorOutputBuffer


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(kv_connector_module, "get_forward_context", object)

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(  # type: ignore[arg-type]
        SimpleNamespace(kv_transfer_config=kv_config), {}
    )
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
        kv_connector_step_id=None,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    attn_metadata = {"layer": object()}
    connector.pre_forward(  # type: ignore[arg-type]
        output,
        request_state_indices=request_indices,
        request_ids=request_ids,
        attn_metadata=attn_metadata,
    )
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]

    kwargs = connector.kv_connector.start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))  # type: ignore[arg-type]
    assert connector.kv_connector.start_load_kv.call_count == 2
    assert connector.kv_connector.start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(False))  # type: ignore[arg-type]

    assert events == ["handle", "bind", "start", "wait", "clear"]


@pytest.mark.parametrize("runner", ["v1", "v2"])
@pytest.mark.parametrize("sync_load", [False, True])
def test_preserve_consumed_notifications_after_forward_failure(
    monkeypatch, runner, sync_load
):
    """Preserve consumed notifications without requiring an async output API."""
    connector = _make_connector(monkeypatch, [])
    backend = connector.kv_connector
    backend.get_transfer_results.return_value = KVConnectorTransferResults(
        finished_sending={"old"}, finished_recving={"load"}, failed_recving={"load"}
    )
    backend.get_block_ids_with_load_errors.return_value = {7}
    buffer = KVConnectorOutputBuffer()
    monkeypatch.setattr(kv_connector_module, "kv_connector_output_buffer", buffer)
    monkeypatch.setattr(mixin_module, "kv_connector_output_buffer", buffer)
    step = _scheduler_output(sync_load)
    step.kv_connector_step_id = 0
    if runner == "v1":
        monkeypatch.setattr(mixin_module, "get_kv_transfer_group", lambda: backend)
        monkeypatch.setattr(mixin_module, "KVConnectorBase", type(backend))
        monkeypatch.setattr(mixin_module, "get_forward_context", object)
        with (
            pytest.raises(RuntimeError, match="forward failed"),
            mixin_module.KVConnectorModelRunnerMixin._get_kv_connector_output(
                step
            ) as output,
        ):
            assert output is None
            raise RuntimeError("forward failed")
    else:
        connector.pre_forward(step)
        # Failed forward never reached post_forward.
        connector.recover()
        connector.recover()
    recovered = buffer.take(None)
    assert set(recovered) == {0}
    assert recovered[0].finished_sending == {"old"}
    assert recovered[0].finished_recving == {"load"}
    assert recovered[0].failed_recving == {"load"}
    assert recovered[0].invalid_block_ids == {7}
    assert buffer.take(None) == {}


def test_explicit_recovery_rejects_incomplete_notification_collection(monkeypatch):
    """Explicit KV recovery cannot reuse blocks when collection itself failed."""
    connector = _make_connector(monkeypatch, [])
    buffer = KVConnectorOutputBuffer()
    monkeypatch.setattr(kv_connector_module, "kv_connector_output_buffer", buffer)
    step = _scheduler_output(True)
    step.kv_connector_step_id = 0
    connector.pre_forward(step)
    connector.kv_connector.wait_for_save.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        connector.post_forward(set())
    with pytest.raises(RuntimeError, match="restart required"):
        buffer.take(None)
