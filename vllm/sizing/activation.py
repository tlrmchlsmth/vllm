# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import ceil
from typing import Any


@dataclass(frozen=True)
class ActivationEstimate:
    min_bytes: int
    max_bytes: int
    note: str


def estimate_activation_range_from_config(vllm_config: Any) -> ActivationEstimate:
    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    parallel_config = vllm_config.parallel_config
    hf_text_config = model_config.hf_text_config

    hidden_size = int(getattr(hf_text_config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        return ActivationEstimate(
            0,
            0,
            "activation/workspace sizing unsupported: missing hidden_size",
        )

    num_tokens = int(scheduler_config.max_num_batched_tokens)
    dtype_size = dtype_size_bytes(model_config.dtype)
    tp_size = int(parallel_config.tensor_parallel_size)

    num_heads = int(getattr(hf_text_config, "num_attention_heads", 0) or 0)
    local_heads = ceil(num_heads / tp_size) if num_heads > 0 else 0
    attention_window = min(int(model_config.max_model_len), num_tokens)
    attention_workspace = num_tokens * attention_window * local_heads * dtype_size * 2

    intermediate_size = int(getattr(hf_text_config, "intermediate_size", 0) or 0)
    local_intermediate_size = ceil(intermediate_size / tp_size)

    lower_per_token = hidden_size * dtype_size * 4
    upper_per_token = hidden_size * dtype_size * 10
    if local_intermediate_size > 0:
        upper_per_token += local_intermediate_size * dtype_size
    if model_config.is_moe:
        num_experts_per_tok = int(
            getattr(hf_text_config, "num_experts_per_tok", 2) or 2
        )
        lower_per_token += hidden_size * dtype_size
        upper_per_token += hidden_size * dtype_size * num_experts_per_tok

    min_bytes = num_tokens * lower_per_token
    max_bytes = num_tokens * upper_per_token + attention_workspace
    return ActivationEstimate(
        min_bytes,
        max(max_bytes, min_bytes),
        "activations: config-derived range; KV reserves upper bound",
    )


def dtype_size_bytes(dtype: Any) -> int:
    dtype_str = str(dtype).lower()
    if "float8" in dtype_str or "int8" in dtype_str or "uint8" in dtype_str:
        return 1
    if "float16" in dtype_str or "bfloat16" in dtype_str or "int16" in dtype_str:
        return 2
    if "float32" in dtype_str or "int32" in dtype_str:
        return 4
    if "float64" in dtype_str or "int64" in dtype_str:
        return 8
    if "bool" in dtype_str:
        return 1
    return 2
