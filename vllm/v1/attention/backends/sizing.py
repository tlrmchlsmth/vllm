# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend used by the sizing utility.

This backend is only for model construction and KV-cache sizing. It deliberately
does not import or execute CUDA kernels.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.kv_cache_interface import AttentionSpec


class SizingAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int4_per_token_head",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SizingAttentionImpl"]:
        return SizingAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SizingAttentionMetadataBuilder"]:
        return SizingAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if get_kv_quant_mode(cache_dtype_str) == KVQuantMode.INT4_PER_TOKEN_HEAD:
            return (num_blocks, 2, block_size, num_kv_heads, head_size // 2)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        if cache_layout == "NHD":
            return (0, 1, 2, 3, 4)
        if cache_layout == "HND" and include_num_layers_dimension:
            return (1, 4, 0, 2, 3, 5)
        if cache_layout == "HND":
            return (0, 1, 3, 2, 4)
        raise ValueError(f"Unknown cache layout: {cache_layout}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def use_cascade_attention(*args: Any, **kwargs: Any) -> bool:
        return False


class SizingAttentionMetadata(AttentionMetadata):
    pass


class SizingAttentionMetadataBuilder(
    AttentionMetadataBuilder[SizingAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SizingAttentionMetadata:
        raise RuntimeError("The sizing attention backend cannot execute.")


class SizingAttentionImpl(AttentionImpl[SizingAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SizingAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise RuntimeError("The sizing attention backend cannot execute.")
