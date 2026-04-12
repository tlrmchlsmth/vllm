# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA graph NaN padding tests for attention backends.

In vLLM v1, the full model forward runs inside a CUDA graph. All tensors
are padded to a fixed size (num_tokens_padded > num_actual_tokens). The
padding region may contain stale data — including NaN — from previous
iterations or uninitialized GPU memory.

These tests verify that NaN values in the padding region of Q/K/V tensors
do NOT corrupt the attention output for real tokens. Each attention backend
slices Q/K/V to [:num_actual_tokens] before computing attention, and this
test validates that contract.
"""

import contextlib
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    try_backend_includes_kv_cache_update,
    try_get_attention_backend,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    set_random_seed,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout

DEVICE_TYPE = current_platform.device_type


def _nan_empty(*args, **kwargs):
    """Drop-in for torch.empty that fills with NaN."""
    t = torch._nan_empty_original(*args, **kwargs)
    if t.is_floating_point():
        try:
            t.fill_(float('nan'))
        except RuntimeError:
            pass  # skip if tensor is compiled/requires_grad
    return t


def _nan_empty_like(*args, **kwargs):
    """Drop-in for torch.empty_like that fills with NaN."""
    t = torch._nan_empty_like_original(*args, **kwargs)
    if t.is_floating_point():
        try:
            t.fill_(float('nan'))
        except RuntimeError:
            pass  # skip if tensor is compiled/requires_grad
    return t


@contextlib.contextmanager
def _patch_empty():
    """Patch torch.empty and torch.empty_like to return NaN-filled tensors.
    Simulates worst-case recycled GPU memory."""
    torch._nan_empty_original = torch.empty
    torch._nan_empty_like_original = torch.empty_like
    with (
        patch.object(torch, 'empty', _nan_empty),
        patch.object(torch, 'empty_like', _nan_empty_like),
    ):
        yield
    del torch._nan_empty_original
    del torch._nan_empty_like_original


BACKENDS_TO_TEST = [
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.FLASHINFER,
    AttentionBackendEnum.TRITON_ATTN,
]

try:
    import flashinfer  # noqa: F401
except ImportError:
    BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASHINFER)


def _to_torch_dtype(dtype):
    if isinstance(dtype, str):
        return STR_DTYPE_TO_TORCH_DTYPE.get(dtype, torch.float16)
    return dtype


class MockAttentionLayer:
    def __init__(self, device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def _run_attention_nan_padding_test(
    backend: AttentionBackendEnum,
    batch_spec: BatchSpec,
    num_padding_tokens: int,
    model: str = "meta-llama/Meta-Llama-3-8B",
    block_size: int = 16,
):
    """Run attention with NaN-filled padding and verify real output is clean.

    1. Create Q/K/V with valid data for real tokens, NaN for padding
    2. Patch torch.empty/empty_like to return NaN (simulates recycled memory)
    3. Build attention metadata with num_actual_tokens < total tensor size
    4. Run the backend forward
    5. Verify output[:num_actual_tokens] has no NaN
    """
    set_random_seed(42)
    device = torch.device(f"{DEVICE_TYPE}:0")

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    dtype = _to_torch_dtype(vllm_config.model_config.dtype)
    scale = 1.0 / (head_size ** 0.5)

    # Build metadata for real tokens only
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, device, arange_block_indices=True,
    )
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    num_tokens_padded = num_actual_tokens + num_padding_tokens

    # Generate real Q/K/V data and KV context
    k_contexts, v_contexts = [], []
    all_q, all_k, all_v = [], [], []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads, head_size,
                         dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size,
                              dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size,
                              dtype=dtype, device=device)

        all_q.append(q)
        all_k.append(k_full[context_len:])
        all_v.append(v_full[context_len:])
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    # Populate KV cache (format: [2, num_blocks, block_size, heads, head_dim])
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts, v_contexts, block_size, num_kv_heads, head_size,
        dtype, device, 8192, common_attn_metadata, randomize_blocks=False,
    )

    # Create padded Q/K/V: real data + NaN padding
    q_real = torch.cat(all_q, dim=0)
    k_real = torch.cat(all_k, dim=0)
    v_real = torch.cat(all_v, dim=0)

    query = torch.full(
        (num_tokens_padded, num_q_heads, head_size),
        float('nan'), dtype=dtype, device=device,
    )
    key = torch.full(
        (num_tokens_padded, num_kv_heads, head_size),
        float('nan'), dtype=dtype, device=device,
    )
    value = torch.full(
        (num_tokens_padded, num_kv_heads, head_size),
        float('nan'), dtype=dtype, device=device,
    )
    query[:num_actual_tokens] = q_real
    key[:num_actual_tokens] = k_real
    value[:num_actual_tokens] = v_real

    # Also patch empty/empty_like so any internal allocations get NaN

    # Per-backend KV cache format
    kv_cache_for_backend = kv_cache
    reset_kv_cache_layout = False

    if backend in (
        AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.TRITON_ATTN,
    ):
        kv_cache_for_backend = kv_cache.transpose(0, 1)

    if backend == AttentionBackendEnum.FLASHINFER:
        kv_cache_for_backend = (
            kv_cache_for_backend.transpose(2, 3).contiguous().transpose(2, 3)
        )
        set_kv_cache_layout("HND")
        reset_kv_cache_layout = True
    elif backend == AttentionBackendEnum.TRITON_ATTN:
        kv_cache_for_backend = kv_cache_for_backend.contiguous()

    try:
        builder_cls, impl_cls = try_get_attention_backend(backend)
        layer_names = ["model.layers.0.self_attn"]

        with set_current_vllm_config(vllm_config):
            if backend == AttentionBackendEnum.FLASHINFER:
                import unittest.mock

                from vllm.v1.attention.backends.utils import (
                    PerLayerParameters,
                )

                def mock_get_per_layer_parameters(vc, ln, ic):
                    return {
                        name: PerLayerParameters(
                            window_left=-1,
                            logits_soft_cap=0.0,
                            sm_scale=scale,
                        )
                        for name in ln
                    }

                with unittest.mock.patch(
                    "vllm.v1.attention.backends.flashinfer"
                    ".get_per_layer_parameters",
                    mock_get_per_layer_parameters,
                ):
                    builder = builder_cls(
                        kv_cache_spec, layer_names, vllm_config, device,
                    )
                    attn_metadata = builder.build(
                        common_prefix_len=0,
                        common_attn_metadata=common_attn_metadata,
                    )
            else:
                builder = builder_cls(
                    kv_cache_spec, layer_names, vllm_config, device,
                )
                attn_metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=common_attn_metadata,
                )

            impl = impl_cls(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                attn_type=AttentionType.DECODER,
                kv_cache_dtype="auto",
            )

            mock_layer = MockAttentionLayer(device)
            output = torch.full_like(query, float('nan'))

            with _patch_empty():
                if not try_backend_includes_kv_cache_update(backend):
                    impl.do_kv_cache_update(
                        mock_layer, key, value, kv_cache_for_backend,
                        attn_metadata.slot_mapping,
                    )

                # Check if this is a decode-only batch (all query_lens=1)
                is_decode = all(
                    q == 1 for q in batch_spec.query_lens
                )

                if is_decode:
                    # Warmup
                    output = impl.forward(
                        mock_layer, query, key, value,
                        kv_cache_for_backend, attn_metadata,
                        output=output,
                    )
                    torch.cuda.synchronize()

                    # Capture in CUDA graph
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = impl.forward(
                            mock_layer, query, key, value,
                            kv_cache_for_backend, attn_metadata,
                            output=output,
                        )

                    # Pollute buffers before replay — simulates stale
                    # data from previous iteration
                    query[num_actual_tokens:] = float('nan')
                    key[num_actual_tokens:] = float('nan')
                    value[num_actual_tokens:] = float('nan')
                    output.fill_(float('nan'))

                    # Replay twice: first replay puts NaN into internal
                    # buffers for padding positions. Second replay tests
                    # whether stale NaN in those buffers corrupts real
                    # token output.
                    graph.replay()
                    torch.cuda.synchronize()

                    # Re-pollute for second replay
                    query[num_actual_tokens:] = float('nan')
                    key[num_actual_tokens:] = float('nan')
                    value[num_actual_tokens:] = float('nan')
                    output.fill_(float('nan'))
                    graph.replay()
                else:
                    output = impl.forward(
                        mock_layer, query, key, value,
                        kv_cache_for_backend, attn_metadata,
                        output=output,
                    )
        torch.cuda.synchronize()
    finally:
        if reset_kv_cache_layout:
            set_kv_cache_layout(None)

    # Verify real token output is NaN-free
    real_output = output[:num_actual_tokens]
    assert not torch.isnan(real_output).any(), (
        f"{backend.name}: attention output has NaN in real tokens "
        f"(num_actual={num_actual_tokens}, padded={num_tokens_padded})"
    )
    assert not torch.isinf(real_output).any(), (
        f"{backend.name}: attention output has Inf in real tokens"
    )


# (batch_name, batch_spec, num_padding_tokens)
# Padding tokens round up to next multiple of 8 to match production buckets.
BATCH_CONFIGS = [
    # Single token decode, padded to 8
    ("1tok_decode", BatchSpec(
        seq_lens=[512],
        query_lens=[1],
    ), 7),
    # 5 decode tokens padded to 8
    ("5tok_decode", BatchSpec(
        seq_lens=[128] * 5,
        query_lens=[1] * 5,
    ), 3),
    # 13 decode tokens padded to 16
    ("13tok_decode", BatchSpec(
        seq_lens=[128] * 13,
        query_lens=[1] * 13,
    ), 3),
    # 31 decode tokens padded to 32 (nearly full bucket)
    ("31tok_decode", BatchSpec(
        seq_lens=[128, 256, 512, 1024] * 7 + [128, 256, 512],
        query_lens=[1] * 31,
    ), 1),
    # 33 decode tokens padded to 40 (just crosses bucket)
    ("33tok_decode", BatchSpec(
        seq_lens=[128] * 33,
        query_lens=[1] * 33,
    ), 7),
    # Long context decode
    ("long_ctx_decode", BatchSpec(
        seq_lens=[4096] * 5,
        query_lens=[1] * 5,
    ), 3),
    # Prefill
    ("prefill", BatchSpec(
        seq_lens=[128, 256],
        query_lens=[16, 32],
    ), 8),
    # Mixed: decode + prefill in same batch
    ("mixed", BatchSpec(
        seq_lens=[128, 256, 512],
        query_lens=[1, 1, 16],
    ), 6),
]


@pytest.mark.parametrize("backend", BACKENDS_TO_TEST,
                         ids=lambda b: b.name if hasattr(b, 'name') else b)
@pytest.mark.parametrize(
    "batch_name,batch_spec,num_padding_tokens",
    BATCH_CONFIGS,
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_attention_nan_padding(backend, batch_name, batch_spec,
                               num_padding_tokens):
    """Attention output for real tokens must be NaN-free even when
    padding tokens in Q/K/V contain NaN."""
    _run_attention_nan_padding_test(
        backend=backend,
        batch_spec=batch_spec,
        num_padding_tokens=num_padding_tokens,
    )


# ============================================================================
# MLA backends: NaN padding in query/kv_c/k_pe
# ============================================================================

MLA_BACKENDS_TO_TEST = [
    AttentionBackendEnum.CUTLASS_MLA,
    AttentionBackendEnum.FLASHMLA,
    AttentionBackendEnum.FLASH_ATTN_MLA,
    AttentionBackendEnum.FLASHINFER_MLA,
    AttentionBackendEnum.TRITON_MLA,
]

# Filter to available backends
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported

if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 10:
    for _b in (AttentionBackendEnum.CUTLASS_MLA,
               AttentionBackendEnum.FLASHINFER_MLA):
        if _b in MLA_BACKENDS_TO_TEST:
            MLA_BACKENDS_TO_TEST.remove(_b)
if not flash_attn_supports_mla():
    if AttentionBackendEnum.FLASH_ATTN_MLA in MLA_BACKENDS_TO_TEST:
        MLA_BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASH_ATTN_MLA)
if not is_flashmla_dense_supported()[0]:
    if AttentionBackendEnum.FLASHMLA in MLA_BACKENDS_TO_TEST:
        MLA_BACKENDS_TO_TEST.remove(AttentionBackendEnum.FLASHMLA)

MLA_BACKEND_BLOCK_SIZES = {}
for _b in MLA_BACKENDS_TO_TEST:
    _sizes = _b.get_class().get_supported_kernel_block_sizes()
    if _sizes:
        _default = _sizes[0]
        MLA_BACKEND_BLOCK_SIZES[_b] = (
            _default if isinstance(_default, int) else _default.base
        )
    else:
        MLA_BACKEND_BLOCK_SIZES[_b] = 16


def _run_mla_nan_padding_test(
    backend: AttentionBackendEnum,
    batch_spec: BatchSpec,
    num_padding_tokens: int,
    model: str = "nvidia/DeepSeek-R1-0528-NVFP4-v2",
    kv_cache_dtype: str = "auto",
):
    """Run MLA attention with NaN-filled padding and verify real output."""
    from tests.v1.attention.test_mla_backends import (
        BACKEND_BLOCK_SIZES,
        MockMLAAttentionLayer,
        create_and_prepopulate_kv_cache as mla_create_kv_cache,
        run_attention_backend as mla_run_attention_backend,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    set_random_seed(42)
    device = torch.device(f"{DEVICE_TYPE}:0")

    block_size = MLA_BACKEND_BLOCK_SIZES[backend]
    required_blocks = sum(
        (s + block_size - 1) // block_size for s in batch_spec.seq_lens
    )
    num_gpu_blocks = required_blocks + 1 + 100

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
    )
    vllm_config.cache_config.cache_dtype = kv_cache_dtype

    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    dtype = _to_torch_dtype(vllm_config.model_config.dtype)

    # MLA dimensions (DeepSeek-R1)
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128

    # Generate real data
    k_contexts, kpe_contexts = [], []
    all_q, all_kv_c, all_k_pe = [], [], []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads,
                         qk_nope_head_dim + qk_rope_head_dim,
                         dtype=dtype, device=device)
        kv_c_full = torch.randn(s_len, kv_lora_rank,
                                 dtype=dtype, device=device)
        k_pe_full = torch.randn(s_len, 1, qk_rope_head_dim,
                                 dtype=dtype, device=device)

        all_q.append(q)
        all_kv_c.append(kv_c_full[context_len:])
        all_k_pe.append(k_pe_full[context_len:])
        k_contexts.append(kv_c_full[:context_len])
        kpe_contexts.append(k_pe_full[:context_len])

    # Build metadata
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, device, arange_block_indices=True,
    )
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    num_tokens_padded = num_actual_tokens + num_padding_tokens

    # Pad block table for MLA alignment
    required_divisor = max(1, int(128 / block_size))
    current_cols = common_attn_metadata.block_table_tensor.shape[1]
    if current_cols % required_divisor != 0:
        padded_cols = ((current_cols + required_divisor - 1)
                       // required_divisor) * required_divisor
        padding = torch.zeros(
            (common_attn_metadata.block_table_tensor.shape[0],
             padded_cols - current_cols),
            dtype=torch.int32, device=device,
        )
        common_attn_metadata.block_table_tensor = torch.cat(
            [common_attn_metadata.block_table_tensor, padding], dim=1,
        )

    # KV cache
    kv_cache = mla_create_kv_cache(
        kv_c_contexts=k_contexts,
        k_pe_contexts=kpe_contexts,
        block_size=block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
    )

    # Concat real tokens then pad with NaN
    q_real = torch.cat(all_q, dim=0)
    kv_c_real = torch.cat(all_kv_c, dim=0)
    k_pe_real = torch.cat(all_k_pe, dim=0)

    query = torch.full(
        (num_tokens_padded, num_q_heads,
         qk_nope_head_dim + qk_rope_head_dim),
        float('nan'), dtype=dtype, device=device,
    )
    kv_c = torch.full(
        (num_tokens_padded, kv_lora_rank),
        float('nan'), dtype=dtype, device=device,
    )
    k_pe = torch.full(
        (num_tokens_padded, 1, qk_rope_head_dim),
        float('nan'), dtype=dtype, device=device,
    )
    query[:num_actual_tokens] = q_real
    kv_c[:num_actual_tokens] = kv_c_real
    k_pe[:num_actual_tokens] = k_pe_real

    # Extend slot_mapping with -1 for padding tokens so cache writes
    # skip them (mirrors production behavior in gpu_model_runner.py)
    real_slot_mapping = common_attn_metadata.slot_mapping
    padded_slot_mapping = torch.full(
        (num_tokens_padded,), -1, dtype=torch.int64, device=device,
    )
    padded_slot_mapping[:num_actual_tokens] = real_slot_mapping
    common_attn_metadata.slot_mapping = padded_slot_mapping

    # Create mock kv_b_proj
    W_UK = torch.randn(kv_lora_rank, num_q_heads, qk_nope_head_dim,
                         dtype=dtype, device=device) / (kv_lora_rank ** 0.5)
    W_UV = torch.randn(kv_lora_rank, num_q_heads, v_head_dim,
                         dtype=dtype, device=device) / (kv_lora_rank ** 0.5)
    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
    kv_b_proj_weight_2d = kv_b_proj_weight.view(
        kv_lora_rank, num_q_heads * (qk_nope_head_dim + v_head_dim),
    )
    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_q_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(
        kv_b_proj_weight_2d.T, requires_grad=False,
    )

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=head_size,
        dtype=vllm_config.model_config.dtype,
        sliding_window=vllm_config.model_config.get_sliding_window(),
        cache_dtype_str=kv_cache_dtype,
    )

    with _patch_empty():
        output = mla_run_attention_backend(
            backend, kv_cache_spec, ["placeholder"],
            vllm_config, device, common_attn_metadata,
            query, kv_c, k_pe, kv_cache,
            kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            mock_kv_b_proj, q_scale=1.0, k_scale=1.0,
            kv_cache_dtype=kv_cache_dtype,
        )
    torch.cuda.synchronize()

    real_output = output[:num_actual_tokens]
    assert not torch.isnan(real_output).any(), (
        f"{backend.name}: MLA output has NaN in real tokens "
        f"(num_actual={num_actual_tokens}, padded={num_tokens_padded})"
    )
    assert not torch.isinf(real_output).any(), (
        f"{backend.name}: MLA output has Inf in real tokens"
    )


# (batch_name, batch_spec, num_padding_tokens)
# Padded to next multiple of 8.
MLA_BATCH_CONFIGS = [
    # Single token decode padded to 8
    ("1tok_decode", BatchSpec(
        seq_lens=[512],
        query_lens=[1],
    ), 7),
    # 5 decode tokens padded to 8
    ("5tok_decode", BatchSpec(
        seq_lens=[128] * 5,
        query_lens=[1] * 5,
    ), 3),
    # 13 decode tokens padded to 16
    ("13tok_decode", BatchSpec(
        seq_lens=[128] * 13,
        query_lens=[1] * 13,
    ), 3),
    # 33 decode tokens padded to 40
    ("33tok_decode", BatchSpec(
        seq_lens=[128] * 33,
        query_lens=[1] * 33,
    ), 7),
    # Long context decode
    ("long_ctx_decode", BatchSpec(
        seq_lens=[4096] * 5,
        query_lens=[1] * 5,
    ), 3),
]


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@pytest.mark.parametrize("backend", MLA_BACKENDS_TO_TEST,
                         ids=lambda b: b.name if hasattr(b, 'name') else b)
@pytest.mark.parametrize(
    "batch_name,batch_spec,num_padding_tokens",
    MLA_BATCH_CONFIGS,
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_mla_nan_padding(default_vllm_config, dist_init,
                         backend, batch_name, batch_spec,
                         num_padding_tokens, kv_cache_dtype):
    """MLA attention output for real tokens must be NaN-free even when
    padding tokens in Q/kv_c/k_pe contain NaN."""
    # Skip backends that don't support the requested kv_cache_dtype
    supported = backend.get_class().supported_kv_cache_dtypes
    if kv_cache_dtype not in supported:
        pytest.skip(f"{backend.name} doesn't support {kv_cache_dtype}")

    _run_mla_nan_padding_test(
        backend=backend,
        batch_spec=batch_spec,
        num_padding_tokens=num_padding_tokens,
        kv_cache_dtype=kv_cache_dtype,
    )


# ============================================================================
# Per-token ops: verify NaN in padding rows doesn't corrupt real rows
# ============================================================================


class TestPerTokenOpsNaNPadding:
    """Verify that per-token ops (RMSNorm, Linear, RoPE) process each
    token independently, so NaN in padding rows cannot corrupt real rows.

    These ops run on the full padded tensor inside a CUDA graph.
    """

    def test_rmsnorm_nan_padding(self):
        """RMSNorm computes variance over the hidden dim (last dim) only,
        so NaN in one token row cannot affect another token's output."""
        from vllm.config import VllmConfig
        from vllm.model_executor.layers.layernorm import RMSNorm

        device = torch.device(f"{DEVICE_TYPE}:0")
        hidden_size = 256
        num_real = 32
        num_padded = 64

        with set_current_vllm_config(VllmConfig()):
            norm = RMSNorm(hidden_size).to(device)

        x = torch.randn(num_padded, hidden_size, device=device,
                         dtype=torch.float16)
        x[num_real:] = float('nan')

        output = norm(x)
        torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            "RMSNorm: NaN in padding rows corrupted real token output"
        )
        assert not torch.isinf(real_output).any(), (
            "RMSNorm: Inf in real token output"
        )

    def test_linear_nan_padding(self):
        """Linear (matmul) is per-row: each token independently multiplies
        with the weight matrix."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        in_features = 256
        out_features = 512
        num_real = 32
        num_padded = 64

        weight = torch.randn(out_features, in_features, device=device,
                              dtype=torch.float16)

        x = torch.randn(num_padded, in_features, device=device,
                         dtype=torch.float16)
        x[num_real:] = float('nan')

        output = x @ weight.t()
        torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            "Linear: NaN in padding rows corrupted real token output"
        )

    def test_silu_activation_nan_padding(self):
        """SiLU activation is element-wise: NaN in padding cannot affect
        real tokens."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        hidden_size = 256
        num_real = 32
        num_padded = 64

        x = torch.randn(num_padded, hidden_size, device=device,
                         dtype=torch.float16)
        x[num_real:] = float('nan')

        output = torch.nn.functional.silu(x)
        torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            "SiLU: NaN in padding rows corrupted real token output"
        )

    def test_rotary_embedding_nan_padding(self):
        """Rotary embedding applies per-token position-based rotation.
        NaN in padding positions should not affect real token output."""
        from vllm.config import VllmConfig
        from vllm.model_executor.layers.rotary_embedding.base import (
            RotaryEmbedding,
        )

        device = torch.device(f"{DEVICE_TYPE}:0")
        head_size = 128
        num_heads = 8
        num_real = 32
        num_padded = 64
        max_position = 2048

        with set_current_vllm_config(VllmConfig()):
            rope = RotaryEmbedding(
                head_size=head_size,
                rotary_dim=head_size,
                max_position_embeddings=max_position,
                base=10000.0,
                is_neox_style=True,
                dtype=torch.float16,
            ).to(device)

        # Positions: real tokens get valid positions, padding gets 0
        positions = torch.zeros(num_padded, dtype=torch.long, device=device)
        positions[:num_real] = torch.arange(num_real, device=device)

        # Q/K: real tokens have valid data, padding has NaN
        query = torch.randn(num_padded, num_heads * head_size,
                             device=device, dtype=torch.float16)
        key = torch.randn(num_padded, num_heads * head_size,
                           device=device, dtype=torch.float16)
        query[num_real:] = float('nan')
        key[num_real:] = float('nan')

        query_out, key_out = rope.forward(positions, query, key)
        torch.cuda.synchronize()

        assert not torch.isnan(query_out[:num_real]).any(), (
            "RoPE: NaN in padding rows corrupted real query output"
        )
        assert not torch.isnan(key_out[:num_real]).any(), (
            "RoPE: NaN in padding rows corrupted real key output"
        )


# ============================================================================
# Full DeepSeek-R1 decoder layer: end-to-end NaN padding test
# ============================================================================


def _run_moe_nan_padding_test(
    num_real: int,
    num_padded: int,
    E: int,
    topk: int,
    K: int,
    N: int,
    active_experts: list[int] | None = None,
):
    """NVFP4 MoE NaN padding test with CUDA graph capture/replay.

    Args:
        active_experts: If set, route all real tokens to only these experts
            (others get 0 tokens). If None, use normal router.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernel,
        MoEActivation,
    )

    device = torch.device(f"{DEVICE_TYPE}:0")

    from tests.kernels.moe.utils import make_test_weights
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        CutlassExpertsFp4,
    )

    (_, w1_q, w1_bs, w1_gs), (_, w2_q, w2_bs, w2_gs) = (
        make_test_weights(E, N, K, in_dtype=torch.bfloat16,
                          quant_dtype="nvfp4")
    )
    a1_gs = torch.ones((E,), device=device, dtype=torch.float32)
    a2_gs = torch.ones((E,), device=device, dtype=torch.float32)
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=(1 / w1_gs), g2_alphas=(1 / w2_gs),
        a1_gscale=a1_gs, a2_gscale=a2_gs,
        w1_scale=w1_bs, w2_scale=w2_bs,
    )
    moe_config = make_dummy_moe_config()
    kernel = FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config, quant_config=quant_config,
            allow_new_interface=True, use_monolithic=False,
        ),
        CutlassExpertsFp4(
            moe_config=moe_config, quant_config=quant_config,
        ),
        inplace=False,
    )
    w1, w2 = w1_q, w2_q
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(pipeline_parallel_size=1)
    )

    # Gate weights
    gate_weight = torch.randn(E, K, device=device,
                               dtype=torch.bfloat16) * 0.02

    # Padded hidden_states with NaN in padding region
    hidden_states = torch.randn(
        num_padded, K, device=device, dtype=torch.bfloat16,
    ) * 0.1
    hidden_states[num_real:] = float('nan')

    # Router
    if active_experts is not None:
        # Force routing to specific experts only
        topk_ids = torch.zeros(num_padded, topk, dtype=torch.int32,
                                device=device)
        for ki in range(topk):
            topk_ids[:, ki] = active_experts[ki % len(active_experts)]
        topk_weights = torch.ones(num_padded, topk, dtype=torch.float32,
                                   device=device) / topk
    else:
        router_logits = hidden_states @ gate_weight.t()
        routing_weights = torch.softmax(
            router_logits, dim=-1, dtype=torch.float32,
        )
        topk_weights, topk_ids = torch.topk(
            routing_weights, topk, dim=-1,
        )
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)

    apply_kwargs = dict(
        hidden_states=hidden_states,
        w1=w1, w2=w2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=E,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    with set_current_vllm_config(vllm_cfg), _patch_empty():
        # Warmup (triton JIT compilation)
        output = kernel.apply(**apply_kwargs)
        torch.cuda.synchronize()

        # Capture in CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = kernel.apply(**apply_kwargs)

        # Pollute buffers before replay — simulates stale data from
        # previous iteration and fresh NaN in padding regions
        hidden_states[num_real:] = float('nan')
        output.fill_(float('nan'))

        # Replay twice: first replay puts NaN-derived values into
        # intermediate graph buffers for padding rows. Second replay
        # tests whether those stale intermediate NaNs corrupt real
        # token output on the next iteration.
        graph.replay()
        torch.cuda.synchronize()

        # Re-pollute input padding for the second replay
        hidden_states[num_real:] = float('nan')
        graph.replay()

    torch.cuda.synchronize()

    real_output = output[:num_real]
    assert not torch.isnan(real_output).any(), (
        f"NVFP4 MoE CUDA graph: NaN in real tokens "
        f"(real={num_real}, padded={num_padded}, E={E}, topk={topk})"
    )
    assert not torch.isinf(real_output).any(), (
        f"NVFP4 MoE CUDA graph: Inf in real tokens"
    )


# Padding configs: (num_real, num_padded)
# In production, padded size is rounded up to the next multiple of 8.
MOE_PADDING_CONFIGS = [
    (1, 8),       # single token, padded to min bucket
    (5, 8),       # small batch, next-8 padding
    (13, 16),     # tight padding, 3 extra
    (17, 24),     # crosses 16 boundary
    (31, 32),     # nearly full bucket, 1 padding token
    (33, 40),     # just over a bucket boundary
    (57, 64),     # larger bucket, 7 padding tokens
]

# Expert counts (including very small)
MOE_EXPERT_COUNTS = [2, 4, 8, 32]

# topk values
MOE_TOPK_VALUES = [1, 2, 8]

# Hidden size configs: (K, N) - different sizes hit different alignment paths
MOE_HIDDEN_CONFIGS = [
    (256, 512),       # small, fast
    (2560, 1024),     # DeepSeek production hidden_size
]

@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize("E", MOE_EXPERT_COUNTS)
@pytest.mark.parametrize("topk", MOE_TOPK_VALUES)
@pytest.mark.parametrize(
    "num_real,num_padded", MOE_PADDING_CONFIGS,
    ids=[f"{r}to{p}" for r, p in MOE_PADDING_CONFIGS],
)
def test_moe_cudagraph_nan_padding(workspace_init, num_real, num_padded,
                                    topk, E):
    """NVFP4 MoE with router + CUDA graph capture/replay + NaN padding."""
    if topk > E:
        pytest.skip(f"topk={topk} > E={E}")
    _run_moe_nan_padding_test(
        num_real, num_padded, E, topk, K=1024, N=512,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "K,N", MOE_HIDDEN_CONFIGS,
    ids=[f"K{k}_N{n}" for k, n in MOE_HIDDEN_CONFIGS],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(13, 16), (33, 40)],
    ids=["13to16", "33to40"],
)
def test_moe_hidden_sizes_cudagraph_nan_padding(workspace_init,
                                                  num_real, num_padded, K, N):
    """NVFP4 MoE with different hidden sizes + CUDA graph + NaN padding."""
    _run_moe_nan_padding_test(
        num_real, num_padded, E=8, topk=2, K=K, N=N,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "E,zero_experts",
    [
        (8, [2, 3, 4, 5, 6, 7]),     # most experts get 0 tokens
        (8, [0]),                      # expert 0 gets 0 (corruption target)
        (8, [0, 1, 2, 3, 4, 5, 6]),   # only expert 7 gets tokens
        (32, list(range(2, 32))),      # 30 of 32 experts get 0
        (4, [1, 2, 3]),               # only expert 0 gets tokens
        (2, [1]),                      # minimal: 2 experts, 1 gets 0
        (4, [0, 2]),                   # scattered zeros including expert 0
        (8, list(range(8))),           # ALL experts get 0 tokens
    ],
    ids=["most_zero", "expert0_zero", "only_expert7", "30of32_zero",
         "only_expert0", "minimal_2e", "scattered_zeros", "all_zero"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_moe_zero_token_experts_cudagraph_nan_padding(
    workspace_init, num_real, num_padded, E, zero_experts,
):
    """NVFP4 MoE with specific experts receiving 0 tokens + CUDA graph +
    NaN padding."""
    active = [e for e in range(E) if e not in zero_experts]
    if not active:
        active = None
        topk = 2
    else:
        topk = min(2, len(active))
    _run_moe_nan_padding_test(
        num_real, num_padded, E, topk, K=1024, N=512,
        active_experts=active,
    )


# ============================================================================
# Multi-GPU EP MoE: DeepEP LL + NVFP4 and FlashInfer NVLink
# ============================================================================

from tests.utils import multi_gpu_test
from vllm.utils.import_utils import has_deep_ep

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(), reason="Requires deep_ep",
)


def _deepep_ll_nan_padding_worker(
    pgi,
    dp_size: int,
    num_experts: int,
    num_real_per_rank: list[int] | int,
    num_padded: int,
    n: int,
    k: int,
    topk: int,
    use_nvfp4: bool,
):
    """Worker for DeepEP LL MoE with router + NaN padding.

    Args:
        num_real_per_rank: Either a single int (same for all ranks) or a
            list with per-rank real token counts. This lets us test the
            case where some DP ranks have zero real tokens (all padding).

    Each rank:
    1. Runs the router (gate linear → softmax → topk) on padded input
    2. Dispatches via DeepEP LL
    3. Runs batched experts (NVFP4 or FP8)
    4. Combines results
    5. Verifies real tokens are NaN-free
    """
    if isinstance(num_real_per_rank, int):
        num_real = num_real_per_rank
    else:
        num_real = num_real_per_rank[pgi.rank]
    from tests.kernels.moe.parallel_utils import DeepEPLLArgs, make_deepep_a2a
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernel,
        MoEActivation,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)
    pg = torch.distributed.new_group(list(range(pgi.world_size)))

    num_local_experts = num_experts // pgi.world_size
    max_tokens_per_rank = num_padded

    if use_nvfp4:
        import vllm.envs as envs_mod
        envs_mod.VLLM_DEEPEPLL_NVFP4_DISPATCH = True

        from tests.kernels.moe.utils import make_test_weights
        from vllm.model_executor.layers.fused_moe.config import (
            nvfp4_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (  # noqa: E501
            FlashInferCuteDSLBatchedExperts,
        )

        (_, w1_q, w1_bs, w1_gs), (_, w2_q, w2_bs, w2_gs) = (
            make_test_weights(num_local_experts, n, k,
                              in_dtype=torch.bfloat16, quant_dtype="nvfp4")
        )
        a1_gs = torch.ones((num_local_experts,), device=device,
                            dtype=torch.float32)
        a2_gs = torch.ones((num_local_experts,), device=device,
                            dtype=torch.float32)
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs), g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs, a2_gscale=a2_gs,
            w1_scale=w1_bs, w2_scale=w2_bs,
        )
        moe_config = make_dummy_moe_config()
        fused_experts = FlashInferCuteDSLBatchedExperts(
            max_num_tokens=max_tokens_per_rank,
            num_dispatchers=pgi.world_size // dp_size,
            moe_config=moe_config, quant_config=quant_config,
        )
        w1, w2 = w1_q, w2_q
    else:
        # FP8 per-token quant
        e = num_local_experts
        w1_bf16 = torch.randn(e, 2 * n, k, device=device,
                               dtype=torch.bfloat16) / 10
        w2_bf16 = torch.randn(e, k, n, device=device,
                               dtype=torch.bfloat16) / 10
        w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
        w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty(e, 2 * n, 1, device=device,
                                dtype=torch.float32)
        w2_scale = torch.empty(e, k, 1, device=device,
                                dtype=torch.float32)
        for eid in range(e):
            w1[eid], w1_scale[eid] = ops.scaled_fp8_quant(
                w1_bf16[eid], use_per_token_if_dynamic=True)
            w2[eid], w2_scale[eid] = ops.scaled_fp8_quant(
                w2_bf16[eid], use_per_token_if_dynamic=True)

        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
        )
        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            w1_scale=w1_scale, w2_scale=w2_scale,
            per_act_token_quant=True,
        )
        moe_config = make_dummy_moe_config()
        fused_experts = BatchedTritonExperts(
            max_num_tokens=max_tokens_per_rank,
            num_dispatchers=pgi.world_size // dp_size,
            moe_config=moe_config, quant_config=quant_config,
        )

    # Gate weights (router)
    gate_weight = torch.randn(num_experts, k, device=device,
                               dtype=torch.bfloat16) * 0.02

    # Padded hidden_states with NaN
    hidden_states = torch.randn(num_padded, k, device=device,
                                 dtype=torch.bfloat16) * 0.1
    hidden_states[num_real:] = float('nan')

    # Router on padded input
    router_logits = hidden_states @ gate_weight.t()
    routing_weights = torch.softmax(router_logits, dim=-1,
                                     dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int64)

    # Expert map
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32,
                             device=device)
    e_start = pgi.rank * num_local_experts
    expert_map[e_start:e_start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device,
    )

    # DeepEP LL all2all
    ll_args = DeepEPLLArgs(
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_size=k, num_experts=num_experts,
        use_fp8_dispatch=False,
    )
    a2a = make_deepep_a2a(pg=pg, pgi=pgi, dp_size=dp_size,
                           deepep_ht_args=None, deepep_ll_args=ll_args)

    kernel = FusedMoEKernel(
        prepare_finalize=a2a, fused_experts=fused_experts, inplace=False,
    )

    from tests.v1.attention.test_cudagraph_nan_padding import _patch_empty
    with set_current_vllm_config(VllmConfig()), _patch_empty():
        output = kernel.apply(
            hidden_states=hidden_states, w1=w1, w2=w2,
            topk_weights=topk_weights, topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )

    torch.cuda.synchronize()
    mode = "NVFP4" if use_nvfp4 else "FP8"
    real_output = output[:num_real]
    assert not torch.isnan(real_output).any(), (
        f"DeepEP LL {mode} MoE with router: NaN in real tokens on "
        f"rank {pgi.rank} (real={num_real}, padded={num_padded})"
    )
    assert not torch.isinf(real_output).any(), (
        f"DeepEP LL {mode} MoE with router: Inf in real tokens on "
        f"rank {pgi.rank}"
    )


DEEPEP_PADDING_CONFIGS = [
    (1, 8),
    (5, 8),
    (13, 16),
    (33, 40),
    (57, 64),
]

DEEPEP_EXPERT_TOPK_CONFIGS = [
    (32, 2),     # standard
    (32, 8),     # DeepSeek-R1 topk=8
    (8, 2),      # fewer experts
    (4, 1),      # minimal
    (2, 1),      # absolute minimum
]


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Requires sm89+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "num_experts,topk", DEEPEP_EXPERT_TOPK_CONFIGS,
    ids=[f"E{e}_top{t}" for e, t in DEEPEP_EXPERT_TOPK_CONFIGS],
)
@pytest.mark.parametrize(
    "num_real,num_padded", DEEPEP_PADDING_CONFIGS,
    ids=[f"{r}to{p}" for r, p in DEEPEP_PADDING_CONFIGS],
)
def test_deepep_ll_fp8_ep2_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + FP8 + router + NaN padding (EP=2)."""
    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        2, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 128, 2560, topk, False,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "num_experts,topk", DEEPEP_EXPERT_TOPK_CONFIGS,
    ids=[f"E{e}_top{t}" for e, t in DEEPEP_EXPERT_TOPK_CONFIGS],
)
@pytest.mark.parametrize(
    "num_real,num_padded", DEEPEP_PADDING_CONFIGS,
    ids=[f"{r}to{p}" for r, p in DEEPEP_PADDING_CONFIGS],
)
def test_deepep_ll_nvfp4_ep2_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + NVFP4 dispatch + router + NaN padding (EP=2).

    Production path for nvidia/DeepSeek-R1-0528-NVFP4-v2 with EP.
    Uses VLLM_DEEPEPLL_NVFP4_DISPATCH=1 and FlashInfer CuteDSL experts.
    """
    from vllm.utils.flashinfer import (
        has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    )
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("Requires FlashInfer CuteDSL kernels")

    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        2, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 1024, 2560, topk, True,
    )


# EP=4 tests (need 4 GPUs)
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Requires sm89+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "num_experts,topk",
    [(32, 2), (32, 8), (8, 2)],
    ids=["E32_top2", "E32_top8", "E8_top2"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_fp8_ep4_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + FP8 + router + NaN padding (EP=4)."""
    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        4, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 128, 2560, topk, False,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "num_experts,topk",
    [(32, 2), (32, 8), (8, 2)],
    ids=["E32_top2", "E32_top8", "E8_top2"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_nvfp4_ep4_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + NVFP4 + router + NaN padding (EP=4)."""
    from vllm.utils.flashinfer import (
        has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    )
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("Requires FlashInfer CuteDSL kernels")

    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        4, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 1024, 2560, topk, True,
    )


# EP=2 with DP=2 (need 4 GPUs) — production-like EP+DP combo
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Requires sm89+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_fp8_ep2_dp2_nan_padding(
    workspace_init, num_real, num_padded,
):
    """DeepEP LL + FP8 + EP=2 + DP=2 + NaN padding.
    Tests the DP+EP combination used in production."""
    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        4, _deepep_ll_nan_padding_worker,
        2, 32, num_real, num_padded, 128, 2560, 2, False,  # dp_size=2
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_nvfp4_ep2_dp2_nan_padding(
    workspace_init, num_real, num_padded,
):
    """DeepEP LL + NVFP4 + EP=2 + DP=2 + NaN padding."""
    from vllm.utils.flashinfer import (
        has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    )
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("Requires FlashInfer CuteDSL kernels")

    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        4, _deepep_ll_nan_padding_worker,
        2, 32, num_real, num_padded, 1024, 2560, 2, True,  # dp_size=2
    )


# Asymmetric DP padding: some ranks have real tokens, others are pure NaN
# This is the critical production case — when load is uneven across DP ranks
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Requires sm89+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "num_real_per_rank,num_padded",
    [
        ([33, 0], 40),     # rank 1 is completely idle (all NaN padding)
        ([1, 0], 8),       # rank 0 has 1 token, rank 1 all padding
        ([33, 1], 40),     # rank 1 has just 1 real token
        ([0, 33], 40),     # rank 0 is completely idle
        ([13, 5], 16),     # both have tokens but different amounts
    ],
    ids=["r0_full_r1_empty", "r0_1tok_r1_empty", "r0_full_r1_1tok",
         "r0_empty_r1_full", "asymmetric_13_5"],
)
def test_deepep_ll_fp8_asymmetric_dp_nan_padding(
    workspace_init, num_real_per_rank, num_padded,
):
    """DeepEP LL + FP8 with asymmetric DP padding.

    Some DP ranks have real tokens, others are entirely NaN padding.
    This is the production scenario when load is uneven across DP ranks.
    """
    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        2, _deepep_ll_nan_padding_worker,
        1, 32, num_real_per_rank, num_padded, 128, 2560, 2, False,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "num_real_per_rank,num_padded",
    [
        ([33, 0], 40),     # rank 1 is completely idle (all NaN padding)
        ([1, 0], 8),       # rank 0 has 1 token, rank 1 all padding
        ([0, 33], 40),     # rank 0 is completely idle
        ([13, 5], 16),     # both have tokens but different amounts
    ],
    ids=["r0_full_r1_empty", "r0_1tok_r1_empty",
         "r0_empty_r1_full", "asymmetric_13_5"],
)
def test_deepep_ll_nvfp4_asymmetric_dp_nan_padding(
    workspace_init, num_real_per_rank, num_padded,
):
    """DeepEP LL + NVFP4 with asymmetric DP padding.

    Production scenario: some DP ranks idle with pure NaN padding while
    others process real tokens. The all2all must handle NaN-padded tokens
    from idle ranks without corrupting real tokens on active ranks.
    """
    from vllm.utils.flashinfer import (
        has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    )
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("Requires FlashInfer CuteDSL kernels")

    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        2, _deepep_ll_nan_padding_worker,
        1, 32, num_real_per_rank, num_padded, 1024, 2560, 2, True,
    )


# EP=8 tests (need 8 GPUs)
@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="Requires sm89+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "num_experts,topk",
    [(32, 8), (64, 8)],
    ids=["E32_top8", "E64_top8"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_fp8_ep8_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + FP8 + router + NaN padding (EP=8)."""
    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        8, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 128, 2560, topk, False,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@requires_deep_ep
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "num_experts,topk",
    [(32, 8), (64, 8)],
    ids=["E32_top8", "E64_top8"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(5, 8), (13, 16), (33, 40)],
    ids=["5to8", "13to16", "33to40"],
)
def test_deepep_ll_nvfp4_ep8_nan_padding(
    workspace_init, num_real, num_padded, num_experts, topk,
):
    """DeepEP LL + NVFP4 + router + NaN padding (EP=8)."""
    from vllm.utils.flashinfer import (
        has_flashinfer_cutedsl_grouped_gemm_nt_masked,
    )
    if not has_flashinfer_cutedsl_grouped_gemm_nt_masked():
        pytest.skip("Requires FlashInfer CuteDSL kernels")

    from tests.kernels.moe.parallel_utils import parallel_launch

    parallel_launch(
        8, _deepep_ll_nan_padding_worker,
        1, num_experts, num_real, num_padded, 1024, 2560, topk, True,
    )
