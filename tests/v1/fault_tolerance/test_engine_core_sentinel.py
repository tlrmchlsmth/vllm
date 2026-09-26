# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

import threading
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.outputs import KVConnectorOutput


def _make_sentinel(batch_queue):
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine_index = 0
    sentinel.resumed = threading.Event()
    sentinel._push_status = Mock()
    sentinel.engine = SimpleNamespace(
        batch_queue=batch_queue,
        scheduler=Mock(),
        model_executor=Mock(is_failed=False),
        _send_abort_outputs=Mock(),
    )
    return sentinel


@pytest.mark.parametrize("has_batch_queue", [False, True])
def test_resolve_block_failures_before_requests_release_blocks(has_batch_queue):
    """Resolve block failures before reuse, even without a batch queue."""
    failed: Future[None] = Future()
    failed.set_exception(RuntimeError("forward failed"))
    queue = deque([(failed, None, None)]) if has_batch_queue else None
    sentinel = _make_sentinel(queue)
    engine = sentinel.engine
    output = KVConnectorOutput(invalid_block_ids={7}, finished_sending={"old"})
    engine.model_executor.recover_kv_connector_outputs.return_value = [output]
    calls = []
    engine.scheduler.update_from_kv_connector_recovery.side_effect = (
        lambda _: calls.append("resolve")
    )
    engine.scheduler.finish_requests.side_effect = lambda *_: calls.append("abort")

    sentinel.on_fault(RuntimeError("forward failed"))

    assert calls == ["resolve", "abort"]
    engine.scheduler.update_from_kv_connector_recovery.assert_called_once_with([output])
    engine.scheduler.update_from_output.assert_not_called()
    assert sentinel.status_type == EngineStatusType.UNHEALTHY
    assert not engine.batch_queue


def test_explicit_kv_recovery_refuses_to_resume_after_drain_failure():
    """Explicit KV recovery must fail closed if a worker buffer is unavailable."""
    sentinel = _make_sentinel(None)
    sentinel.engine.model_executor.recover_kv_connector_outputs.side_effect = (
        RuntimeError("worker unavailable")
    )
    sentinel.on_fault(RuntimeError("forward failed"))
    assert sentinel.status_type == EngineStatusType.DEAD
    sentinel.engine.scheduler.update_from_kv_connector_recovery.assert_not_called()
