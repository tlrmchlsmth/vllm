# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig,
                         ModelConfig, SchedulerConfig, VllmConfig)
from vllm.sampling_params import KVTransferParams, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

EOS_TOKEN_ID = 50256


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
) -> VllmConfig:
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
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_both",
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu")
    )


def create_model_runner(
    vllm_config: VllmConfig,
    device: torch.device,
) -> GPUModelRunner:
    return GPUModelRunner(
        vllm_config=vllm_config,
        device=device,
    )


def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def create_request(
    request_id: int,
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
    return Request(
        request_id=f"id-{request_id}",
        prompt=None,
        prompt_token_ids=[i * request_id for i in range(num_tokens)],
        sampling_params=sampling_params,
        multi_modal_inputs=None,
        multi_modal_placeholders=None,
        multi_modal_hashes=None,
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=0,
    )
