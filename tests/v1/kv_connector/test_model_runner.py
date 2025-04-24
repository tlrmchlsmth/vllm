# SPDX-License-Identifier: Apache-2.0
import torch

from .utils import (create_model_runner, create_request, create_scheduler,
                    create_vllm_config)


def test_basic_remote_prefill():
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)
    model_runner = create_model_runner(vllm_config=vllm_config,
                                       device=torch.device(type="cpu"))

    NUM_TOKENS = 16

    normal_request = create_request(request_id=0, num_tokens=NUM_TOKENS)

    remote_request = create_request(
        request_id=1,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )

    scheduler.add_request(normal_request)
    scheduler.add_request(remote_request)

    scheduler_output = scheduler.schedule()

    # Both should be running, but only the normal request
    # should have scheduled tokens.
    assert len(scheduler.running) == 2
    assert scheduler_output.num_scheduled_tokens[
        normal_request.request_id] == NUM_TOKENS
    assert scheduler_output.num_scheduled_tokens[
        remote_request.request_id] == 0

    for scheduled_new_req in scheduler_output.scheduled_new_reqs:
        # Remote request has all tokens computed externally.
        if scheduled_new_req.req_id == remote_request.request_id:
            assert scheduled_new_req.num_computed_tokens == NUM_TOKENS - 1
        # Normal request has no tokens computed externally.
        if scheduled_new_req.req_id == normal_request.request_id:
            assert scheduled_new_req.num_computed_tokens == 0

    # model_runner.execute_model does:
    #   * _update_states
    #   * returns if no tokens scheduled
    #   * _prepare_inputs
    model_runner._update_states(scheduler_output)
    attn_metadata, logits_indices, spec_decode_metadata = (
        model_runner._prepare_inputs(scheduler_output))

    print(f"{attn_metadata=}")
    print(f"{logits_indices=}")
    print(f"{spec_decode_metadata=}")
