import copy

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

from .utils import create_requests, create_scheduler, create_vllm_config

# SPDX-License-Identifier: Apache-2.0


def test_basic_remote_prefill():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

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
    assert (scheduler_output.num_scheduled_tokens[request_id]) == 1

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
