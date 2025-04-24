import copy

import torch

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.sampling_params import KVTransferParams, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

# SPDX-License-Identifier: Apache-2.0

EOS_TOKEN_ID = 50256


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    num_blocks: int = 10000,
    block_size: int = 8,
) -> Scheduler:
    '''Create scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (None)

    Returns:
      :class:`Scheduler` instance
    '''
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
    )
    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    max_tokens: int = 16,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
) -> list[Request]:
    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = KVTransferParams(do_remote_prefill=True, )
    elif do_remote_prefill:
        kv_transfer_params = KVTransferParams(
            do_remote_prefill=True,
            remote_engine_id="abc",
            remote_block_ids=[1, 2, 3],
        )
    else:
        kv_transfer_params = None

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        kv_transfer_params=kv_transfer_params,
    )
    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{i}",
            prompt=None,
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            multi_modal_inputs=None,
            multi_modal_placeholders=None,
            multi_modal_hashes=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0,
        )
        requests.append(request)
    return requests


def test_basic_remote_prefill():
    scheduler = create_scheduler()
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    NUM_TOKENS = 16
    request = create_requests(num_requests=1,
                              num_tokens=NUM_TOKENS,
                              do_remote_prefill=True)[0]

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # Remote Prefill: req should be scheduled with 0 tokens
    # but have the entire prompt "computed" from the POV of
    # the scheduler + persistent batch (since the KVConnector
    # will write directly into allocated blocks).
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.recving_KV_req_ids) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    assert scheduled_req.num_computed_tokens == NUM_TOKENS - 1
    assert scheduler_output.num_scheduled_tokens[scheduled_req.req_id] == 0

    # Blocks should not be cached until the KVs are recv,
    # but they should be touched so that they are not preempted.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert len(block_pool.cached_block_hash_to_block) == 0
    assert (block_pool.free_block_queue.num_free_blocks
            < START_FREE_BLOCK_QUEUE_SIZE)
    assert request_id not in scheduler.kv_cache_manager.num_cached_block

    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    # Request should still be in the running and recving state.
    assert len(scheduler.running) == 1
    assert len(scheduler.recving_KV_req_ids) == 1
    assert len(engine_core_outputs.outputs) == 0

    # STEP (2):
    # Remote Prefill: req should be running.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler.recving_KV_req_ids) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 0

    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving.append(request_id)
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)

    # Request should be out of the recving state.
    assert len(scheduler.running) == 1
    assert len(scheduler.recving_KV_req_ids) == 0
    assert len(engine_core_outputs.outputs) == 0

    # TODO(rob): once we support caching, we should check that the
    # blocks are cached here.

    # STEP (3):
    # Remote Prefill: the request should now have scheduled tokens.
    scheduler_output = scheduler.schedule()
    assert (len(scheduler_output.scheduled_cached_reqs)) == 1

    # req_to_index = {
    #     request.request_id: i
    #     for i, request in enumerate(requests)
    # }
    # model_runner_output = ModelRunnerOutput(
    #     req_ids=[request.request_id for request in requests],
    #     req_id_to_index=req_to_index,
    #     # Only the first request has a sampled token id because
    #     # the rest requests are still being prefilled.
    #     sampled_token_ids=[[0], [], []],
    #     spec_token_ids=None,
    #     logprobs=None,
    #     prompt_logprobs_dict={},
    # )
