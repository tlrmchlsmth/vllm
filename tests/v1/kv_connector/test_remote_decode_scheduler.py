# SPDX-License-Identifier: Apache-2.0
import copy

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus, FinishReason

from .utils import (create_request, create_scheduler,
                    create_vllm_config, create_model_runner_output)

def test_basic_remote_decode_cycle():
    """Test Remote Decode Lifecycle."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    
    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    
    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_decode=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 1

    # (1b): execute_model()
    model_runner_output = create_model_runner_output(reqs=[request])

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output)
    
    # Ensure the request is finished after 1 tokens.
    assert request.is_finished()
    assert request.status == RequestStatus.FINISHED_REMOTE_DECODE
    output = engine_core_outputs.outputs[0]
    assert output.finish_reason == FinishReason.REMOTE_DECODE
    assert output.kv_transfer_params is not None

    # Request freed in Scheduler and in Persistent Batch.
    # This causes the request to be freed in the scheduler.
    

    # This causes the request to be freed in the PB on next step().
    assert request_id in scheduler.finished_req_ids
    assert len(scheduler.running) == 0

    # ... but blocks should not be freed.
    blocks = scheduler.kv_cache_manager.req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1
