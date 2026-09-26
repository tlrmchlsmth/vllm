# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVConnectorOutput,
    KVOutputAggregator,
)
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)
from vllm.v1.outputs import ModelRunnerOutput


class Mock: ...


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])
    assert proc.terminate_called is expected_terminate


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
        ec_output_aggregator: ECOutputAggregator | None = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
            ec_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


class _FakeResponseMQ:
    """Stands in for a worker response MessageQueue."""

    def __init__(self, responses: list[tuple[Any, Any]]):
        self._responses = list(responses)

    def dequeue(self, timeout: float | None = None):
        return self._responses.pop(0)


@pytest.mark.parametrize("enable_ft", [True, False])
@pytest.mark.parametrize("has_kv", [True, False])
def test_drain_worker_replies_before_raising(enable_ft, has_kv):
    """Drain worker replies so the next RPC cannot consume an old response."""
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_fault_tolerance=enable_ft)
    )
    executor.is_failed = False
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda *_: None)
    success = WorkerProc.ResponseStatus.SUCCESS
    executor.response_mqs = [
        _FakeResponseMQ(
            [(WorkerProc.ResponseStatus.FAILURE, "forward failed"), (success, "new-0")]
        ),
        _FakeResponseMQ([(success, "old-1"), (success, "new-1")]),
    ]
    executor.futures_queue = deque()
    aggregator = KVOutputAggregator(2, defer_outputs=True) if has_kv else None
    with pytest.raises(RuntimeError, match="forward failed"):
        executor.collective_rpc("execute_model", kv_output_aggregator=aggregator)
    assert executor.collective_rpc("next") == ["new-0", "new-1"]


@pytest.mark.parametrize("executor_cls", [MultiprocExecutor, UniProcExecutor])
def test_explicit_kv_recovery_keeps_later_steps_in_worker_buffers(executor_cls):
    """Explicit KV recovery drains old steps once without needing model outputs."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.kv_connector_output import KVConnectorOutputBuffer

    executor = executor_cls.__new__(executor_cls)
    executor._use_kv_recovery = True
    executor._kv_connector_step_id = 0
    executor._kv_recovery_failed = False
    executor.parallel_config = SimpleNamespace(
        world_size=2,
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=120),
    )
    executor.kv_output_aggregator = KVOutputAggregator(2, defer_outputs=True)
    buffers = [KVConnectorOutputBuffer(), KVConnectorOutputBuffer()]
    for buffer in buffers:
        buffer.put(0, KVConnectorOutput(finished_sending={"first"}))
        buffer.put(1, KVConnectorOutput(finished_recving={"failed-step"}))
    executor.collective_rpc = lambda method, args, timeout: [
        b.take(args[0]) for b in buffers
    ]
    step = SchedulerOutput.make_empty()
    executor.prepare_kv_connector_step(step)
    model_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    output = executor.collect_kv_connector_output(step, model_output)
    assert model_output.kv_connector_output is None
    assert output.kv_connector_output.finished_sending == {"first"}
    recovered = executor.recover_kv_connector_outputs()
    assert len(recovered) == 1
    assert recovered[0].finished_recving == {"failed-step"}
    assert executor.recover_kv_connector_outputs() == []


def test_explicit_kv_recovery_rejects_unsupported_executor_before_startup():
    """Explicit KV recovery requires an executor that implements the contract."""
    executor = SimpleNamespace(supports_kv_recovery=False, _init_executor=MagicMock())
    config = MagicMock()
    config.parallel_config.enable_fault_tolerance = True
    with pytest.raises(ValueError, match="does not support fault tolerance"):
        Executor.__init__(executor, config)
    executor._init_executor.assert_not_called()


def test_explicit_kv_recovery_does_not_retry_partially_consumed_rpc():
    """Preserve consumed notifications by refusing recovery after partial retrieval."""
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor._use_kv_recovery = True
    executor._kv_recovery_failed = False
    executor.parallel_config = SimpleNamespace(
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=120),
    )
    executor.collective_rpc = MagicMock(side_effect=RuntimeError("worker failed"))
    with pytest.raises(RuntimeError, match="worker failed"):
        executor.recover_kv_connector_outputs()
    with pytest.raises(RuntimeError, match="restart required"):
        executor.recover_kv_connector_outputs()
    executor.collective_rpc.assert_called_once()
