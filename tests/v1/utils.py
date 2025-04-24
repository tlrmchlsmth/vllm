# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def create_scheduler(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: Optional[bool] = None,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    use_kv_connector: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
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
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
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
    kwargs_cache = ({} if enable_prefix_caching is None else {
        'enable_prefix_caching': enable_prefix_caching
    })
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        **kwargs_cache,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    ) if use_kv_connector else None

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


def create_requests(num_requests: int,
                    num_tokens: int = 10,
                    mm_positions: Optional[list[PlaceholderRange]] = None,
                    max_tokens: int = 16,
                    stop_token_ids: Optional[list[int]] = None,
                    prompt_logprobs: Optional[int] = None):
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    requests = []
    for i in range(num_requests):
        if mm_positions is not None:
            mm_position = mm_positions[i]
            mm_inputs = [MultiModalKwargs({})] * len(mm_position)
        else:
            mm_position = None
            mm_inputs = None
        request = Request(
            request_id=f"{i}",
            prompt=None,
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            multi_modal_inputs=mm_inputs,
            multi_modal_placeholders=mm_position,
            multi_modal_hashes=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0,
        )
        requests.append(request)
    return requests
