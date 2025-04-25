# SPDX-License-Identifier: Apache-2.0
import copy

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import Request, RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config

def test_single_remote_prefill():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    
    # 2 and a half full external blocks.
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(vllm_config.cache_config.block_size * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    
    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()

    # Nothing running and empty scheduler output.
    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert len(scheduler_output.scheduled_cached_reqs) == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0

    # Req waiting for KVs with no computed
    # or scheduled tokens.
    assert len(scheduler.waiting) == 1
    assert request in scheduler.waiting
    assert (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
    assert (request.num_computed_tokens == 0)
    
    # ... but should have (uncached) blocks allocated to it.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert (block_pool.free_block_queue.num_free_blocks
            < START_FREE_BLOCK_QUEUE_SIZE)
    assert len(block_pool.cached_block_hash_to_block) == 0
    for block in scheduler.kv_cache_manager.req_to_blocks[request_id]:
        assert block._block_hash is None

    # (1b): forward()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output)
    assert len(engine_core_outputs.outputs) == 0

    # STEP (2):
    # (2a): schedule(): nothing happens!
    scheduler_output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 0

    # (2b): forward(): request finishes recv.
    model_runner_output = copy.deepcopy(
        EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.finished_recving = [request_id]

    # (2c): update_from_output(): 
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output)
    assert len(scheduler.waiting) == 1
    assert (request_id in scheduler.finished_recving_KV_req_ids)

    # (3a): schedule(): this should actually schedule.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    
    # Confirm the block are actually allocated.
    num_hashed_blocks = 0
    for block in scheduler.kv_cache_manager.req_to_blocks[request_id]:
        assert block.ref_cnt == 1
        num_hashed_blocks += (1 if block._block_hash is not None else 0)
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS

    # Confirm the rest of the prompt is scheduled.
    print(f"{scheduler_output=}")
    # # Request should be out of the recving state.
    # assert len(scheduler.running) == 1
    # assert len(scheduler.recving_KV_req_ids) == 0
    # assert len(engine_core_outputs.outputs) == 0

    # # STEP (3):
    # # Remote Prefill: the request should now have scheduled tokens.
    # scheduler_output = scheduler.schedule()
    # assert (len(scheduler_output.scheduled_cached_reqs)) == 1
    # assert (scheduler_output.num_scheduled_tokens[request_id]) == 1

    # # req_to_index = {
    # #     request.request_id: i
    # #     for i, request in enumerate(requests)
    # # }
    # # model_runner_output = ModelRunnerOutput(
    # #     req_ids=[request.request_id for request in requests],
    # #     req_id_to_index=req_to_index,
    # #     # Only the first request has a sampled token id because
    # #     # the rest requests are still being prefilled.
    # #     sampled_token_ids=[[0], [], []],
    # #     spec_token_ids=None,
    # #     logprobs=None,
    # #     prompt_logprobs_dict={},
    # # )
