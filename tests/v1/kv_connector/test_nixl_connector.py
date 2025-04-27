# SPDX-License-Identifier: Apache-2.0
import copy
from typing import Optional

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnectorMetadata)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import RequestStatus, Request

from .utils import create_request, create_scheduler, create_vllm_config

def test_scheduler_worker_inferface():

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    
    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetdata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, NixlConnectorMetadata)
    
    assert len(kv_connector_metadata.requests) == 1
    assert request_id in kv_connector_metadata.requests
    print(f"{kv_connector_metadata.requests=}")

