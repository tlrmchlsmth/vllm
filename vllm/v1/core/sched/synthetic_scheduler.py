# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


class SyntheticScheduler(AsyncScheduler):
    """Run caller-constructed scheduler outputs through EngineCore.

    This scheduler bypasses request admission and output-token processing. It
    is intended for deterministic benchmarks that construct complete
    ``SchedulerOutput`` objects while retaining vLLM's normal model runner,
    executor, KV cache manager, and asynchronous batch queue.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._submitted_outputs: deque[SchedulerOutput] = deque()
        self._retained_kv_blocks: dict[int, list[KVCacheBlock]] = {}
        self.submitted_steps = 0
        self.completed_steps = 0

    def submit(self, scheduler_output: SchedulerOutput) -> int:
        """Queue a prebuilt scheduler output and return its one-based ordinal."""
        self._submitted_outputs.append(scheduler_output)
        self.submitted_steps += 1
        return self.submitted_steps

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        del throttle_prefills
        if not self._submitted_outputs:
            raise RuntimeError("synthetic scheduler has no submitted batch")
        return self._submitted_outputs.popleft()

    def add_request(self, request: Request) -> None:
        raise RuntimeError(
            "SyntheticScheduler does not admit frontend request "
            f"{request.request_id!r}; submit a prebuilt SchedulerOutput instead"
        )

    def get_num_unfinished_requests(self) -> int:
        return len(self._submitted_outputs)

    def has_finished_requests(self) -> bool:
        return False

    def retain_kv_blocks_until_complete(
        self, scheduler_output: SchedulerOutput, blocks: list[KVCacheBlock]
    ) -> None:
        """Retain CoW copy endpoints until the output finishes executing."""
        output_id = id(scheduler_output)
        if output_id in self._retained_kv_blocks:
            raise ValueError("scheduler output already has retained KV blocks")
        self._retained_kv_blocks[output_id] = blocks

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        del model_runner_output
        blocks = self._retained_kv_blocks.pop(id(scheduler_output), None)
        if blocks:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
        self.completed_steps += 1
        return {0: EngineCoreOutputs()}

    def shutdown(self) -> None:
        for blocks in self._retained_kv_blocks.values():
            self.kv_cache_manager.block_pool.free_blocks(blocks)
        self._retained_kv_blocks.clear()
        self._submitted_outputs.clear()
        super().shutdown()
