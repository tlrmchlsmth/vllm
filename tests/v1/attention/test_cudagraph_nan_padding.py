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

import threading

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


def _poison_cuda_allocator(device: torch.device | str = "cuda"):
    """Poison the CUDA caching allocator's free list with NaN.

    Allocates buffers of various sizes, fills with 0xFF (NaN for all
    IEEE float types: fp16, bf16, fp32, fp8_e4m3fn), then frees them
    back to the allocator cache. Subsequent torch.empty() calls that
    recycle this memory will get NaN-filled tensors.

    This is more reliable than patching torch.empty at the Python level
    because it also catches allocations made from C++ code (CUTLASS
    workspace buffers, triton intermediates, etc.).
    """
    # Poison at multiple sizes to cover different allocator buckets.
    # The caching allocator rounds up to block sizes, so we poison
    # a range of sizes to maximize coverage.
    # Clear the allocator cache first so our poisoned blocks are the
    # only ones in the free list. This maximizes the chance that
    # subsequent torch.empty() calls recycle poisoned memory.
    # NOTE: This approach is inherently probabilistic — the allocator
    # may split/coalesce blocks differently across CUDA versions.
    # TestPoisonVerification validates that poisoning is effective.
    torch.cuda.empty_cache()

    # Poison one size at a time: allocate, fill, free. The freed buffer
    # stays in the allocator cache with NaN. We don't accumulate all
    # sizes simultaneously to avoid OOM.
    for size_mb in [1, 2, 4, 8, 16, 32, 64]:
        buf = torch.empty(size_mb * 1024 * 1024, dtype=torch.uint8,
                          device=device)
        buf.fill_(0xFF)
        del buf  # returns to allocator cache with NaN pattern


@pytest.fixture
def sm_pressure():
    """Run a background kernel on a separate CUDA stream to occupy SMs.

    In production, other operations (e.g. DP all-reduce, shared experts,
    KV cache management) run concurrently. This fixture simulates that
    SM contention, which can change kernel scheduling and expose
    timing-dependent bugs (especially PDL issues).
    """
    device = torch.device(f"{DEVICE_TYPE}:0")
    stop_event = threading.Event()
    stream = torch.cuda.Stream(device=device)

    def _hog():
        with torch.cuda.stream(stream):
            a = torch.randn(2048, 2048, device=device,
                             dtype=torch.bfloat16)
            b = torch.randn(2048, 2048, device=device,
                             dtype=torch.bfloat16)
            while not stop_event.is_set():
                torch.mm(a, b, out=a)

    thread = threading.Thread(target=_hog, daemon=True)
    thread.start()
    yield
    stop_event.set()
    thread.join(timeout=5)


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

            _poison_cuda_allocator(device)

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

                # First replay: fill ALL inputs with NaN to
                # maximally pollute every internal graph buffer.
                # Set slot_mapping to -1 so NaN K/V don't corrupt
                # the KV cache.
                saved_slot_mapping = attn_metadata.slot_mapping.clone()
                attn_metadata.slot_mapping.fill_(-1)
                query.fill_(float('nan'))
                key.fill_(float('nan'))
                value.fill_(float('nan'))
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()

                # Second replay: restore real data + NaN padding.
                # Tests whether stale NaN in internal graph buffers
                # from the first replay corrupts real token output.
                attn_metadata.slot_mapping.copy_(saved_slot_mapping)
                query[:num_actual_tokens] = q_real
                query[num_actual_tokens:] = float('nan')
                key[:num_actual_tokens] = k_real
                key[num_actual_tokens:] = float('nan')
                value[:num_actual_tokens] = v_real
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

    # Verify KV cache — null block (block 0) must not have been
    # corrupted by padding tokens. Use the original kv_cache (not the
    # transposed backend-specific variant) for consistent indexing.
    # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    null_block = kv_cache[:, 0]  # block 0 across K and V
    assert not torch.isnan(null_block).any(), (
        f"{backend.name}: null block (block 0) has NaN — padding "
        f"tokens wrote to KV cache despite slot_mapping=-1"
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

# Filter to available backends — guarded against non-CUDA environments
if torch.cuda.is_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
    from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported

    if torch.cuda.get_device_properties(0).major < 10:
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
else:
    MLA_BACKENDS_TO_TEST = []

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

    _poison_cuda_allocator(device)
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

    # Verify MLA KV cache — real slots must not have NaN.
    # MLA cache shape: [num_blocks, block_size, head_size]
    real_slots = common_attn_metadata.slot_mapping[:num_actual_tokens]
    real_slots = real_slots[real_slots >= 0]
    if real_slots.numel() > 0:
        kv_cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
        real_cache = kv_cache_flat[real_slots]
        assert not torch.isnan(real_cache).any(), (
            f"{backend.name}: MLA KV cache has NaN in real slots"
        )

    # Null block (block 0) must not be corrupted
    null_block = kv_cache[0]
    assert not torch.isnan(null_block).any(), (
        f"{backend.name}: MLA null block (block 0) has NaN — "
        f"padding tokens wrote to KV cache"
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


class TestPoisonVerification:
    """Verify that allocator poisoning actually works — torch.empty
    returns NaN-filled tensors after poisoning."""

    def test_poison_covers_torch_empty(self):
        """After poisoning, torch.empty should return NaN for float types."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        _poison_cuda_allocator(device)

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            t = torch.empty(1024, 1024, dtype=dtype, device=device)
            assert torch.isnan(t).any(), (
                f"Allocator poisoning failed: torch.empty({dtype}) "
                f"did not return NaN. Poisoning is not effective."
            )

    def test_poison_covers_torch_empty_like(self):
        """torch.empty_like should also get poisoned memory."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        _poison_cuda_allocator(device)

        ref = torch.empty(1024, 1024, dtype=torch.bfloat16, device=device)
        t = torch.empty_like(ref)
        assert torch.isnan(t).any(), (
            "Allocator poisoning failed: torch.empty_like did not "
            "return NaN."
        )

    def test_poison_covers_fp8(self):
        """FP8 tensors should also get NaN from poisoned allocator."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        _poison_cuda_allocator(device)

        t = torch.empty(1024, 1024, dtype=torch.float8_e4m3fn,
                         device=device)
        # 0xFF in FP8 E4M3FN is NaN
        assert torch.isnan(t.float()).any(), (
            "Allocator poisoning failed: torch.empty(fp8) did not "
            "return NaN."
        )

    def test_poison_covers_workspace_sizes(self):
        """Verify poisoning covers the workspace sizes used by MoE/attention
        kernels (typically 1-64MB)."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        _poison_cuda_allocator(device)

        for size_kb in [64, 256, 1024, 4096, 16384]:
            numel = size_kb * 1024 // 2  # bf16 = 2 bytes
            t = torch.empty(numel, dtype=torch.bfloat16, device=device)
            has_nan = torch.isnan(t).any().item()
            assert has_nan, (
                f"Allocator poisoning missed {size_kb}KB allocation. "
                f"Workspace buffers of this size won't be NaN-poisoned."
            )
            del t


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

    def test_embedding_nan_padding(self):
        """Embedding lookup with out-of-range padding indices must not
        corrupt real token embeddings or crash."""
        device = torch.device(f"{DEVICE_TYPE}:0")
        vocab_size = 32000
        hidden_size = 256
        num_real = 32
        num_padded = 40

        embedding = torch.nn.Embedding(vocab_size, hidden_size).to(
            device=device, dtype=torch.float16,
        )

        # Real tokens get valid IDs, padding gets 0 (safe index)
        input_ids = torch.zeros(num_padded, dtype=torch.long, device=device)
        input_ids[:num_real] = torch.randint(
            0, vocab_size, (num_real,), device=device,
        )

        output = embedding(input_ids)
        torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            "Embedding: NaN in real token output"
        )
        assert not torch.isinf(real_output).any(), (
            "Embedding: Inf in real token output"
        )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16), (33, 40)],
    ids=["1to8", "5to8", "13to16", "33to40"],
)
def test_deepseek_dense_mlp_nvfp4_cudagraph_nan_padding(
    default_vllm_config, dist_init, num_real, num_padded,
):
    """DeepSeek-R1 dense MLP with NVFP4 quantized weights, CUDA graph
    capture/replay, and NaN padding.

    This is the MLP used in layers 0-2 (before first_k_dense_replace)
    with nvidia/DeepSeek-R1-0528-NVFP4-v2 NVFP4 weight format.
    """
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16

    vllm_config = create_vllm_config(
        model_name="nvidia/DeepSeek-R1-0528-NVFP4-v2",
        max_model_len=128,
        num_gpu_blocks=8192,
        dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config

    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )

            # Fill NVFP4 weights with random valid data
            for name, param in mlp.named_parameters():
                if param.dtype == torch.uint8:
                    # Packed FP4 weights — random bytes
                    param.data.copy_(torch.randint(
                        0, 256, param.shape, dtype=torch.uint8))
                elif param.dtype == torch.float8_e4m3fn:
                    # Per-block weight scales — random positive fp8
                    param.data.copy_(torch.randint(
                        1, 127, param.shape,
                        dtype=torch.uint8).view(torch.float8_e4m3fn))
                elif param.dtype == torch.float32:
                    # Global scales — set to reasonable values
                    param.data.fill_(1.0)
                elif param.is_floating_point():
                    random_data = torch.randn_like(
                        param, dtype=torch.float32) * 0.02
                    param.data.copy_(random_data.to(param.dtype))

            mlp = mlp.to(device=device)

            # Process weights (computes alpha, renames scales)
            for module in mlp.modules():
                if hasattr(module, 'quant_method') and hasattr(
                    module.quant_method, 'process_weights_after_loading'
                ):
                    module.quant_method.process_weights_after_loading(
                        module)

            # Padded input with NaN
            hidden = torch.randn(
                num_padded, config.hidden_size, dtype=dtype, device=device,
            ) * 0.02
            hidden[num_real:] = float('nan')

            _poison_cuda_allocator(device)

            # Warmup
            output = mlp(hidden)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = mlp(hidden)

            # First replay: all NaN to pollute internal buffers
            saved_hidden = hidden[:num_real].clone()
            hidden.fill_(float('nan'))
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()

            # Second replay: restore real data + NaN padding
            hidden[:num_real] = saved_hidden
            hidden[num_real:] = float('nan')
            output.fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            f"DeepSeek NVFP4 dense MLP CUDA graph: NaN in real tokens "
            f"(real={num_real}, padded={num_padded})"
        )
        assert not torch.isinf(real_output).any(), (
            f"DeepSeek NVFP4 dense MLP CUDA graph: Inf in real tokens"
        )
    finally:
        torch.set_default_dtype(old_dtype)


# ============================================================================
# Full DeepSeek-R1 MLA layer with CUDA graph: projections + attention
# ============================================================================


def _build_deepseek_mla_layer(vllm_config, device, dtype,
                              quant_config=None):
    """Build a DeepseekV2MLAAttention layer with random weights."""
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2MLAAttention,
    )

    config = vllm_config.model_config.hf_config
    layer = DeepseekV2MLAAttention(
        vllm_config=vllm_config,
        config=config,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        max_position_embeddings=config.max_position_embeddings,
        cache_config=vllm_config.cache_config,
        quant_config=quant_config,
        prefix="model.layers.3.self_attn",
    )

    # Initialize weights — handle different dtypes from quantization
    for name, param in layer.named_parameters():
        if param.dtype == torch.uint8:
            param.data.copy_(torch.randint(
                0, 256, param.shape, dtype=torch.uint8))
        elif param.dtype == torch.float8_e4m3fn:
            param.data.copy_(torch.randint(
                1, 127, param.shape,
                dtype=torch.uint8).view(torch.float8_e4m3fn))
        elif param.dtype == torch.float32:
            param.data.fill_(1.0)
        elif param.is_floating_point():
            random_data = torch.randn_like(
                param, dtype=torch.float32) * 0.02
            param.data.copy_(random_data.to(param.dtype))

    layer = layer.to(device=device)

    # Process quantized weights
    if quant_config is not None:
        for module in layer.modules():
            if hasattr(module, 'quant_method') and hasattr(
                module.quant_method, 'process_weights_after_loading'
            ):
                module.quant_method.process_weights_after_loading(module)

    layer.mla_attn.mla_attn.process_weights_after_loading(dtype)
    return layer


def _get_nvfp4_quant_config():
    """Create ModelOptNvFp4Config for NVFP4 tests."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )
    return ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )


@pytest.mark.parametrize(
    "use_nvfp4",
    [False, True],
    ids=["bf16", "nvfp4"],
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16), (33, 40)],
    ids=["1to8", "5to8", "13to16", "33to40"],
)
def test_deepseek_mla_layer_cudagraph_nan_padding(
    default_vllm_config, dist_init, num_real, num_padded, use_nvfp4,
):
    """Full DeepSeek-R1 MLA attention layer under CUDA graph with NaN padding.

    Tests the complete projection + attention pipeline:
    hidden_states → fused_qkv_a_proj → q_a_layernorm → q_b_proj →
    kv_a_layernorm → RoPE → MLA attention → o_proj

    The first graph replay fills ALL of hidden_states with NaN to pollute
    every internal buffer (projection outputs, layernorm intermediates,
    Q/K/V, attention workspace). The second replay restores real data
    and verifies no NaN leaked from the polluted buffers.
    """
    if use_nvfp4 and not current_platform.has_device_capability(100):
        pytest.skip("NVFP4 requires sm100+")

    from vllm.forward_context import set_forward_context
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16
    model = "nvidia/DeepSeek-R1-0528-NVFP4-v2"

    quant_config = _get_nvfp4_quant_config() if use_nvfp4 else None

    batch_spec = BatchSpec(
        seq_lens=[128] * num_real,
        query_lens=[1] * num_real,
    )
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        num_gpu_blocks=8192,
        dtype="bfloat16",
    )
    block_size = vllm_config.cache_config.block_size
    config = vllm_config.model_config.hf_config

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            layer = _build_deepseek_mla_layer(
                vllm_config, device, dtype, quant_config=quant_config,
            )

            # Build attention metadata
            common_attn_metadata = create_common_attn_metadata(
                batch_spec, block_size, device, arange_block_indices=True,
            )

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
                    [common_attn_metadata.block_table_tensor, padding],
                    dim=1,
                )

            # Allocate KV cache
            head_size = config.kv_lora_rank + config.qk_rope_head_dim
            kv_cache = torch.zeros(
                8192, block_size, head_size, dtype=dtype, device=device,
            )
            mla_attn = layer.mla_attn.mla_attn
            mla_attn.kv_cache = kv_cache

            # Register in static forward context
            layer_name = mla_attn.layer_name
            vllm_config.compilation_config.static_forward_context[
                layer_name
            ] = mla_attn

            kv_cache_spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_size=head_size,
                dtype=vllm_config.model_config.dtype,
                sliding_window=None,
                cache_dtype_str="auto",
            )

            builder_cls = mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec, [layer_name], vllm_config, device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            # Create padded inputs
            num_actual = common_attn_metadata.num_actual_tokens
            hidden_states = torch.randn(
                num_padded, config.hidden_size, dtype=dtype, device=device,
            ) * 0.02
            hidden_states[num_actual:] = float('nan')

            positions = torch.zeros(
                num_padded, dtype=torch.long, device=device,
            )
            positions[:num_actual] = torch.arange(
                num_actual, device=device,
            )

            _poison_cuda_allocator(device)

            with set_forward_context(attn_metadata, vllm_config):
                # Warmup
                output = layer(positions, hidden_states, None)
                torch.cuda.synchronize()

                # Capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = layer(positions, hidden_states, None)

                # First replay: ALL NaN to pollute every internal
                # buffer (W_q/W_k/W_v projections, layernorms, RoPE,
                # attention workspace, o_proj)
                saved_hidden = hidden_states[:num_actual].clone()
                saved_slot_mapping = attn_metadata.slot_mapping.clone()
                hidden_states.fill_(float('nan'))
                attn_metadata.slot_mapping.fill_(-1)
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()

                # Second replay: restore real data + NaN padding
                hidden_states[:num_actual] = saved_hidden
                hidden_states[num_actual:] = float('nan')
                attn_metadata.slot_mapping.copy_(saved_slot_mapping)
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()

        real_output = output[:num_actual]
        assert not torch.isnan(real_output).any(), (
            f"DeepSeek MLA layer CUDA graph: NaN in real tokens "
            f"(real={num_actual}, padded={num_padded})"
        )
        assert not torch.isinf(real_output).any(), (
            f"DeepSeek MLA layer CUDA graph: Inf in real tokens"
        )

        # Check KV cache — real slots must not have NaN.
        # The slot_mapping tells us which cache slots were written.
        real_slots = saved_slot_mapping[:num_actual]
        real_slots = real_slots[real_slots >= 0]  # filter any -1
        if real_slots.numel() > 0:
            kv_cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
            real_cache_entries = kv_cache_flat[real_slots]
            assert not torch.isnan(real_cache_entries).any(), (
                f"DeepSeek MLA layer: KV cache has NaN in real slots "
                f"(real={num_actual}, padded={num_padded})"
            )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16)],
    ids=["1to8", "5to8", "13to16"],
)
def test_mla_layer_production_padding(
    default_vllm_config, dist_init, num_real, num_padded,
):
    """MLA layer with production-matching CUDA graph padding behavior.

    In production (gpu_model_runner.py:2188), num_actual_tokens is set
    to num_tokens_padded, NOT the real token count. This means attention
    backends don't slice — they process ALL tokens including padding.
    Padding requests have seq_lens=0 and block_table=NULL_BLOCK_ID (0).

    This test matches that exact contract:
    - num_actual_tokens = num_padded (not num_real)
    - Padding requests in query_start_loc, seq_lens, block_table
    - slot_mapping = -1 for padding tokens
    - FP8 KV cache (production uses --kv-cache-dtype fp8)
    """
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16
    model = "nvidia/DeepSeek-R1-0528-NVFP4-v2"

    vllm_config = create_vllm_config(
        model_name=model, max_model_len=128,
        num_gpu_blocks=8192, dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config
    block_size = vllm_config.cache_config.block_size

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            layer = _build_deepseek_mla_layer(vllm_config, device, dtype)

            # Build metadata matching production: num_actual_tokens =
            # num_padded, with padding requests having seq_lens=0.
            # Real requests: indices 0..num_real-1
            # Padding requests: indices num_real..num_padded-1
            query_start_loc = torch.arange(
                num_padded + 1, dtype=torch.int32, device=device,
            )  # each request has 1 token (decode)
            query_start_loc_cpu = query_start_loc.cpu()

            seq_lens = torch.zeros(
                num_padded, dtype=torch.int32, device=device,
            )
            seq_lens[:num_real] = 128  # real requests have context
            seq_lens_cpu = seq_lens.cpu()

            num_computed_tokens_cpu = torch.zeros(
                num_padded, dtype=torch.int32,
            )
            num_computed_tokens_cpu[:num_real] = 127  # context len

            max_blocks = (128 + block_size - 1) // block_size
            block_table = torch.zeros(
                num_padded, max_blocks, dtype=torch.int32, device=device,
            )
            # Real requests get valid blocks
            for i in range(num_real):
                for b in range(max_blocks):
                    block_table[i, b] = i * max_blocks + b + 1
            # Padding requests get NULL_BLOCK_ID (0) — already zero

            slot_mapping = torch.full(
                (num_padded,), -1, dtype=torch.int64, device=device,
            )
            # Real tokens get valid slots
            for i in range(num_real):
                pos = 127  # decode position
                blk_idx = pos // block_size
                blk_off = pos % block_size
                slot_mapping[i] = (
                    block_table[i, blk_idx] * block_size + blk_off
                )

            # KEY: num_actual_tokens = num_padded (production behavior)
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                _seq_lens_cpu=seq_lens_cpu,
                _num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_padded,
                num_actual_tokens=num_padded,  # PADDED, not real
                max_query_len=1,
                max_seq_len=128,
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                causal=True,
            )

            # Pad block table for MLA alignment
            required_divisor = max(1, int(128 / block_size))
            cols = common_attn_metadata.block_table_tensor.shape[1]
            if cols % required_divisor != 0:
                padded_cols = ((cols + required_divisor - 1)
                               // required_divisor) * required_divisor
                padding = torch.zeros(
                    (num_padded, padded_cols - cols),
                    dtype=torch.int32, device=device,
                )
                common_attn_metadata.block_table_tensor = torch.cat(
                    [common_attn_metadata.block_table_tensor, padding],
                    dim=1,
                )

            head_size = config.kv_lora_rank + config.qk_rope_head_dim
            kv_cache = torch.zeros(
                8192, block_size, head_size, dtype=dtype, device=device,
            )
            mla_attn = layer.mla_attn.mla_attn
            mla_attn.kv_cache = kv_cache
            layer_name = mla_attn.layer_name
            vllm_config.compilation_config.static_forward_context[
                layer_name
            ] = mla_attn

            kv_cache_spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_size=head_size,
                dtype=vllm_config.model_config.dtype,
                sliding_window=None,
                cache_dtype_str="auto",
            )
            builder_cls = mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec, [layer_name], vllm_config, device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            # Hidden states: real data + NaN padding
            hidden_states = torch.randn(
                num_padded, config.hidden_size, dtype=dtype, device=device,
            ) * 0.02
            hidden_states[num_real:] = float('nan')

            positions = torch.zeros(
                num_padded, dtype=torch.long, device=device,
            )
            positions[:num_real] = 127  # decode positions

            _poison_cuda_allocator(device)

            with set_forward_context(attn_metadata, vllm_config):
                # Warmup
                output = layer(positions, hidden_states, None)
                torch.cuda.synchronize()

                # Capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = layer(positions, hidden_states, None)

                # First replay: ALL NaN to pollute internal buffers
                saved_hidden = hidden_states[:num_real].clone()
                saved_slot_mapping = slot_mapping.clone()
                hidden_states.fill_(float('nan'))
                slot_mapping.fill_(-1)
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()

                # Second replay: restore real data + NaN padding
                hidden_states[:num_real] = saved_hidden
                hidden_states[num_real:] = float('nan')
                slot_mapping.copy_(saved_slot_mapping)
                output.fill_(float('nan'))
                graph.replay()
                torch.cuda.synchronize()

        real_output = output[:num_real]
        assert not torch.isnan(real_output).any(), (
            f"MLA layer (production padding): NaN in real tokens "
            f"(real={num_real}, padded={num_padded}, "
            f"num_actual_tokens={num_padded})"
        )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16)],
    ids=["1to8", "5to8", "13to16"],
)
def test_deepseek_mla_layer_cudagraph_sm_pressure(
    default_vllm_config, dist_init, sm_pressure, num_real, num_padded,
):
    """Full MLA layer with NVFP4 weights under CUDA graph with SM pressure.

    Same as test_deepseek_mla_layer_cudagraph_nan_padding but with NVFP4
    weights and a background kernel occupying SMs. This changes kernel
    scheduling and can expose PDL/timing bugs that only manifest under
    contention. Replays the graph many times to increase the chance of
    hitting a race.
    """
    from vllm.forward_context import set_forward_context
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16
    model = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
    quant_config = _get_nvfp4_quant_config()

    batch_spec = BatchSpec(
        seq_lens=[128] * num_real,
        query_lens=[1] * num_real,
    )

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        num_gpu_blocks=8192,
        dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            layer = _build_deepseek_mla_layer(
                vllm_config, device, dtype, quant_config=quant_config,
            )

            common_attn_metadata = create_common_attn_metadata(
                batch_spec,
                vllm_config.cache_config.block_size,
                device, arange_block_indices=True,
            )
            block_size = vllm_config.cache_config.block_size

            required_divisor = max(1, int(128 / block_size))
            cols = common_attn_metadata.block_table_tensor.shape[1]
            if cols % required_divisor != 0:
                padded_cols = ((cols + required_divisor - 1)
                               // required_divisor) * required_divisor
                padding = torch.zeros(
                    (common_attn_metadata.block_table_tensor.shape[0],
                     padded_cols - cols),
                    dtype=torch.int32, device=device,
                )
                common_attn_metadata.block_table_tensor = torch.cat(
                    [common_attn_metadata.block_table_tensor, padding],
                    dim=1,
                )

            head_size = config.kv_lora_rank + config.qk_rope_head_dim
            kv_cache = torch.zeros(
                8192, block_size, head_size, dtype=dtype, device=device,
            )
            mla_attn = layer.mla_attn.mla_attn
            mla_attn.kv_cache = kv_cache
            layer_name = mla_attn.layer_name
            vllm_config.compilation_config.static_forward_context[
                layer_name
            ] = mla_attn

            kv_cache_spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_size=head_size,
                dtype=vllm_config.model_config.dtype,
                sliding_window=None,
                cache_dtype_str="auto",
            )
            builder_cls = mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec, [layer_name], vllm_config, device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            num_actual = common_attn_metadata.num_actual_tokens
            hidden_states = torch.randn(
                num_padded, config.hidden_size, dtype=dtype, device=device,
            ) * 0.02
            hidden_states[num_actual:] = float('nan')
            positions = torch.zeros(
                num_padded, dtype=torch.long, device=device,
            )
            positions[:num_actual] = torch.arange(
                num_actual, device=device,
            )

            _poison_cuda_allocator(device)

            with set_forward_context(attn_metadata, vllm_config):
                # Warmup
                output = layer(positions, hidden_states, None)
                torch.cuda.synchronize()

                # Capture
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = layer(positions, hidden_states, None)

                # Replay many times under SM pressure to stress
                # kernel scheduling and expose timing bugs
                saved_hidden = hidden_states[:num_actual].clone()
                saved_slot_mapping = attn_metadata.slot_mapping.clone()

                for i in range(1000):
                    # Alternate: pollute → replay → restore → replay
                    hidden_states.fill_(float('nan'))
                    attn_metadata.slot_mapping.fill_(-1)
                    output.fill_(float('nan'))
                    graph.replay()

                    hidden_states[:num_actual] = saved_hidden
                    hidden_states[num_actual:] = float('nan')
                    attn_metadata.slot_mapping.copy_(saved_slot_mapping)
                    output.fill_(float('nan'))
                    graph.replay()

                    # Check every 100 iterations to avoid sync overhead
                    if i % 100 == 99:
                        torch.cuda.synchronize()
                        real_output = output[:num_actual]
                        assert not torch.isnan(real_output).any(), (
                            f"SM pressure MLA CUDA graph: NaN at "
                            f"iteration {i} (real={num_actual}, "
                            f"padded={num_padded})"
                        )
    finally:
        torch.set_default_dtype(old_dtype)


# ============================================================================
# Direct test for silu_and_mul_scaled_fp4_experts_quant padding bug
# ============================================================================


@pytest.mark.xfail(
    reason="Known bug: silu_mul_cvt_fp16_to_fp4 kernel padding rows "
           "default to expert_idx=0 and overwrite expert 0's scale. "
           "Fix: skip rows where rowIdx >= expert_offsets[n_experts].",
    strict=True,
)
@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_experts,num_covered,m_topk",
    [
        (4, 12, 16),    # 4 padding rows beyond last expert
        (4, 10, 16),    # 6 padding rows
        (8, 14, 16),    # 2 padding rows
        (4, 4, 8),      # half the rows are padding
        (8, 1, 8),      # 7 padding rows, only 1 real
        (2, 3, 8),      # minimal experts, lots of padding
    ],
    ids=["4pad", "6pad", "2pad", "half_pad", "7pad_1real", "minimal"],
)
def test_fp4_quant_kernel_padding_corruption(
    num_experts, num_covered, m_topk,
):
    """Direct test for the silu_mul_cvt_fp16_to_fp4 padding bug.

    When expert_offsets[-1] < m_topk, rows beyond the last expert
    default to expert_idx=0 and overwrite expert 0's scale factor.
    This is a known bug in the CUDA kernel.
    """
    import vllm._custom_ops as ops

    device = "cuda"
    k = 128  # intermediate size

    # Distribute tokens across experts evenly
    tokens_per_expert = num_covered // num_experts
    remainder = num_covered % num_experts
    offsets = [0]
    for i in range(num_experts):
        n = tokens_per_expert + (1 if i < remainder else 0)
        offsets.append(offsets[-1] + n)
    assert offsets[-1] == num_covered

    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    blockscale_offsets = torch.tensor(
        offsets, dtype=torch.int32, device=device,
    )
    input_global_scale = torch.ones(
        num_experts, dtype=torch.float32, device=device,
    )

    # Clean run: all rows have valid data
    c1_clean = torch.randn(
        m_topk, k * 2, dtype=torch.bfloat16, device=device,
    )
    _, scales_clean = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_clean, input_global_scale, expert_offsets,
        blockscale_offsets, 1,
    )

    # Dirty run: padding rows (beyond num_covered) filled with NaN
    c1_dirty = c1_clean.clone()
    c1_dirty[num_covered:] = float('nan')
    _, scales_dirty = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_dirty, input_global_scale, expert_offsets,
        blockscale_offsets, 1,
    )

    # Check that scales for real expert rows are NOT corrupted
    # by the NaN padding rows
    scales_clean_real = scales_clean[:num_covered]
    scales_dirty_real = scales_dirty[:num_covered]
    match = torch.equal(scales_clean_real, scales_dirty_real)
    assert match, (
        f"FP4 quant kernel: NaN padding rows (rows {num_covered}..{m_topk}) "
        f"corrupted real expert scales. "
        f"This is the silu_mul_cvt_fp16_to_fp4 padding bug where orphan "
        f"rows default to expert_idx=0 and overwrite expert 0's scale."
    )


# ============================================================================
# NVFP4 MoE with CUDA graph: router + experts
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

    Uses CutlassExpertsFp4 (standard format). The production
    FlashInferCuteDSLBatchedExperts kernel is tested through the
    multi-GPU DeepEP tests, as it requires DeepEP LL prepare/finalize.

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
        # Directly set topk_ids to route all tokens to specific experts
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

    _poison_cuda_allocator(device)

    with set_current_vllm_config(vllm_cfg):
        # Warmup (triton JIT compilation)
        output = kernel.apply(**apply_kwargs)
        torch.cuda.synchronize()

        # Capture in CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = kernel.apply(**apply_kwargs)

        # First replay: fill ALL inputs with NaN to maximally
        # pollute every internal graph buffer (workspaces, intermediates).
        saved_hidden = hidden_states[:num_real].clone()
        saved_topk_weights = topk_weights[:num_real].clone()
        saved_topk_ids = topk_ids[:num_real].clone()
        hidden_states.fill_(float('nan'))
        topk_weights.fill_(float('nan'))
        topk_ids.fill_(0)  # route everything to expert 0
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()

        # Second replay: restore real data + NaN padding.
        # Tests whether stale NaN in internal graph buffers from
        # the first replay corrupts real token output.
        hidden_states[:num_real] = saved_hidden
        hidden_states[num_real:] = float('nan')
        topk_weights[:num_real] = saved_topk_weights
        topk_ids[:num_real] = saved_topk_ids
        output.fill_(float('nan'))
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
    """NVFP4 MoE (CutlassExpertsFp4) with router + CUDA graph + NaN padding.
    The production FlashInferCuteDSLBatchedExperts kernel is tested through
    the multi-GPU DeepEP tests."""
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
    ],
    ids=["most_zero", "expert0_zero", "only_expert7", "30of32_zero",
         "only_expert0", "minimal_2e", "scattered_zeros"],
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
        pytest.skip("No active experts")
    topk = min(2, len(active))
    _run_moe_nan_padding_test(
        num_real, num_padded, E, topk, K=1024, N=512,
        active_experts=active,
    )


# ============================================================================
# Numerical edge cases: inputs that could induce NaN via computation
# ============================================================================


@pytest.mark.parametrize(
    "input_pattern",
    [
        "zeros",           # all zeros — degenerate attention scores
        "moderate",        # moderate values — tests scaling behavior
        "tiny",            # very small — underflow in softmax
        "identical",       # all tokens identical — uniform attention
    ],
)
def test_mla_layer_numerical_edge_cases(
    default_vllm_config, dist_init, input_pattern,
):
    """Test that the full MLA layer doesn't produce NaN from edge-case
    hidden_states. Covers Q/K/V projections, RoPE, softmax, attention,
    and o_proj."""
    from vllm.forward_context import set_forward_context
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16
    model = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
    num_tokens = 8

    batch_spec = BatchSpec(
        seq_lens=[128] * num_tokens,
        query_lens=[1] * num_tokens,
    )

    vllm_config = create_vllm_config(
        model_name=model, max_model_len=128,
        num_gpu_blocks=8192, dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            layer = _build_deepseek_mla_layer(vllm_config, device, dtype)

            block_size = vllm_config.cache_config.block_size
            common_attn_metadata = create_common_attn_metadata(
                batch_spec, block_size, device,
                arange_block_indices=True,
            )

            required_divisor = max(1, int(128 / block_size))
            cols = common_attn_metadata.block_table_tensor.shape[1]
            if cols % required_divisor != 0:
                padded_cols = ((cols + required_divisor - 1)
                               // required_divisor) * required_divisor
                padding = torch.zeros(
                    (common_attn_metadata.block_table_tensor.shape[0],
                     padded_cols - cols),
                    dtype=torch.int32, device=device,
                )
                common_attn_metadata.block_table_tensor = torch.cat(
                    [common_attn_metadata.block_table_tensor, padding],
                    dim=1,
                )

            head_size = config.kv_lora_rank + config.qk_rope_head_dim
            kv_cache = torch.zeros(
                8192, block_size, head_size, dtype=dtype, device=device,
            )
            mla_attn = layer.mla_attn.mla_attn
            mla_attn.kv_cache = kv_cache
            layer_name = mla_attn.layer_name
            vllm_config.compilation_config.static_forward_context[
                layer_name
            ] = mla_attn

            kv_cache_spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_size=head_size,
                dtype=vllm_config.model_config.dtype,
                sliding_window=None,
                cache_dtype_str="auto",
            )
            builder_cls = mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec, [layer_name], vllm_config, device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            # Create edge-case input
            if input_pattern == "zeros":
                hidden = torch.zeros(
                    num_tokens, config.hidden_size,
                    dtype=dtype, device=device)
            elif input_pattern == "moderate":
                hidden = torch.full(
                    (num_tokens, config.hidden_size), 10.0,
                    dtype=dtype, device=device)
            elif input_pattern == "tiny":
                hidden = torch.full(
                    (num_tokens, config.hidden_size), 1e-7,
                    dtype=dtype, device=device)
            elif input_pattern == "identical":
                row = torch.randn(
                    1, config.hidden_size,
                    dtype=dtype, device=device) * 0.02
                hidden = row.expand(num_tokens, -1).contiguous()

            positions = torch.arange(
                num_tokens, dtype=torch.long, device=device)

            with set_forward_context(attn_metadata, vllm_config):
                output = layer(positions, hidden, None)
            torch.cuda.synchronize()

            assert not torch.isnan(output).any(), (
                f"MLA layer produced NaN with "
                f"input_pattern='{input_pattern}'"
            )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "weight_pattern",
    [
        "zero_global_scale",
        "huge_global_scale",
        "tiny_global_scale",
        "max_fp4_weights",
        "zero_fp4_weights",
        "max_fp8_block_scales",
        "zero_fp8_block_scales",
        "nan_fp8_block_scales",
    ],
)
def test_mla_layer_nvfp4_weight_edge_cases(
    default_vllm_config, dist_init, weight_pattern,
):
    """Test full MLA layer with adversarial NVFP4 weight configurations.

    The Q/K/V projection weights (fused_qkv_a_proj, q_b_proj, kv_b_proj,
    o_proj) use NVFP4 quantization. Tests degenerate scale/weight values
    that could cause NaN through the projection → attention → output
    pipeline.
    """
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16
    model = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
    num_tokens = 8

    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None, exclude_modules=[],
    )

    batch_spec = BatchSpec(
        seq_lens=[128] * num_tokens,
        query_lens=[1] * num_tokens,
    )

    vllm_config = create_vllm_config(
        model_name=model, max_model_len=128,
        num_gpu_blocks=8192, dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            # Build layer WITHOUT process_weights_after_loading so we
            # can modify raw quant params before they're baked into
            # derived tensors (alpha, etc.)
            from vllm.model_executor.models.deepseek_v2 import (
                DeepseekV2MLAAttention,
            )
            layer = DeepseekV2MLAAttention(
                vllm_config=vllm_config, config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix="model.layers.3.self_attn",
            )
            # Init weights
            for name, param in layer.named_parameters():
                if param.dtype == torch.uint8:
                    param.data.copy_(torch.randint(
                        0, 256, param.shape, dtype=torch.uint8))
                elif param.dtype == torch.float8_e4m3fn:
                    param.data.copy_(torch.randint(
                        1, 127, param.shape,
                        dtype=torch.uint8).view(torch.float8_e4m3fn))
                elif param.dtype == torch.float32:
                    param.data.fill_(1.0)
                elif param.is_floating_point():
                    param.data.copy_(torch.randn_like(
                        param, dtype=torch.float32).mul_(0.02).to(
                            param.dtype))

            # Apply adversarial patterns BEFORE process_weights
            for module in layer.modules():
                for name, param in module.named_parameters(recurse=False):
                    if weight_pattern == "zero_global_scale":
                        if name in ("input_scale", "weight_scale_2"):
                            param.data.fill_(0.0)
                    elif weight_pattern == "huge_global_scale":
                        if name in ("input_scale", "weight_scale_2"):
                            param.data.fill_(1e30)
                    elif weight_pattern == "tiny_global_scale":
                        if name in ("input_scale", "weight_scale_2"):
                            param.data.fill_(1e-30)
                    elif weight_pattern == "max_fp4_weights":
                        if name == "weight" and param.dtype == torch.uint8:
                            param.data.fill_(0x77)
                    elif weight_pattern == "zero_fp4_weights":
                        if name == "weight" and param.dtype == torch.uint8:
                            param.data.fill_(0x00)
                    elif weight_pattern == "max_fp8_block_scales":
                        if name == "weight_scale":
                            param.data.view(torch.uint8).fill_(0x7E)
                    elif weight_pattern == "zero_fp8_block_scales":
                        if name == "weight_scale":
                            param.data.view(torch.uint8).fill_(0x00)
                    elif weight_pattern == "nan_fp8_block_scales":
                        if name == "weight_scale":
                            param.data.view(torch.uint8).fill_(0xFF)

            layer = layer.to(device=device)
            # NOW process weights — bakes adversarial values into alpha
            for module in layer.modules():
                if hasattr(module, 'quant_method') and hasattr(
                    module.quant_method, 'process_weights_after_loading'
                ):
                    module.quant_method.process_weights_after_loading(
                        module)
            layer.mla_attn.mla_attn.process_weights_after_loading(dtype)

            block_size = vllm_config.cache_config.block_size
            common_attn_metadata = create_common_attn_metadata(
                batch_spec, block_size, device,
                arange_block_indices=True,
            )
            required_divisor = max(1, int(128 / block_size))
            cols = common_attn_metadata.block_table_tensor.shape[1]
            if cols % required_divisor != 0:
                padded_cols = ((cols + required_divisor - 1)
                               // required_divisor) * required_divisor
                padding = torch.zeros(
                    (common_attn_metadata.block_table_tensor.shape[0],
                     padded_cols - cols),
                    dtype=torch.int32, device=device,
                )
                common_attn_metadata.block_table_tensor = torch.cat(
                    [common_attn_metadata.block_table_tensor, padding],
                    dim=1,
                )

            head_size = config.kv_lora_rank + config.qk_rope_head_dim
            kv_cache = torch.zeros(
                8192, block_size, head_size, dtype=dtype, device=device,
            )
            mla_attn = layer.mla_attn.mla_attn
            mla_attn.kv_cache = kv_cache
            layer_name = mla_attn.layer_name
            vllm_config.compilation_config.static_forward_context[
                layer_name
            ] = mla_attn

            kv_cache_spec = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_size=head_size,
                dtype=vllm_config.model_config.dtype,
                sliding_window=None, cache_dtype_str="auto",
            )
            builder_cls = mla_attn.attn_backend.get_builder_cls()
            builder = builder_cls(
                kv_cache_spec, [layer_name], vllm_config, device,
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            hidden = torch.randn(
                num_tokens, config.hidden_size,
                dtype=dtype, device=device) * 0.02
            positions = torch.arange(
                num_tokens, dtype=torch.long, device=device)

            with set_forward_context(attn_metadata, vllm_config):
                output = layer(positions, hidden, None)
            torch.cuda.synchronize()

            if weight_pattern in ("nan_fp8_block_scales",
                                  "zero_global_scale",
                                  "huge_global_scale"):
                # These produce NaN/Inf through legitimate overflow —
                # verify no crash, no hang
                pass
            else:
                assert not torch.isnan(output).any(), (
                    f"MLA layer produced NaN with NVFP4 "
                    f"weight_pattern='{weight_pattern}'"
                )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "input_pattern",
    [
        "zeros",           # all zeros — risk of 0/0 in scale computation
        "moderate",        # moderate values — risk of overflow in fp4 quant
        "tiny",            # very small values — risk of underflow
        "identical",       # all tokens identical — degenerate routing
    ],
)
def test_moe_numerical_edge_cases(workspace_init, input_pattern):
    """Test that specific numerical edge cases don't produce NaN in the
    NVFP4 MoE pipeline (router + CutlassExpertsFp4)."""
    from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
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
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernel,
        MoEActivation,
    )

    device = torch.device(f"{DEVICE_TYPE}:0")
    E, topk, K, N = 8, 2, 1024, 512
    num_tokens = 16

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

    # Create edge-case input
    if input_pattern == "zeros":
        hidden = torch.zeros(num_tokens, K, device=device,
                              dtype=torch.bfloat16)
    elif input_pattern == "moderate":
        hidden = torch.full((num_tokens, K), 10.0, device=device,
                             dtype=torch.bfloat16)
    elif input_pattern == "tiny":
        hidden = torch.full((num_tokens, K), 1e-7, device=device,
                             dtype=torch.bfloat16)
    elif input_pattern == "identical":
        row = torch.randn(1, K, device=device, dtype=torch.bfloat16)
        hidden = row.expand(num_tokens, -1).contiguous()
    else:
        raise ValueError(f"Unknown input_pattern: {input_pattern}")

    # Router — use fixed routing to avoid NaN from softmax(inf)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32,
                            device=device)
    for ki in range(topk):
        topk_ids[:, ki] = ki
    topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32,
                               device=device) / topk

    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(pipeline_parallel_size=1))

    with set_current_vllm_config(vllm_cfg):
        output = kernel.apply(
            hidden_states=hidden, w1=w1_q, w2=w2_q,
            topk_weights=topk_weights, topk_ids=topk_ids,
            global_num_experts=E, activation=MoEActivation.SILU,
            apply_router_weight_on_input=False, expert_map=None,
        )
    torch.cuda.synchronize()

    assert not torch.isnan(output).any(), (
        f"NVFP4 MoE produced NaN with input_pattern='{input_pattern}'"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "input_pattern",
    [
        "zeros",
        "moderate",
        "tiny",
        "identical",
    ],
)
def test_nvfp4_dense_mlp_numerical_edge_cases(
    default_vllm_config, dist_init, input_pattern,
):
    """Test that NVFP4 dense MLP doesn't produce NaN from edge-case inputs."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16

    vllm_config = create_vllm_config(
        model_name="nvidia/DeepSeek-R1-0528-NVFP4-v2",
        max_model_len=128, num_gpu_blocks=8192, dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config
    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None, exclude_modules=[],
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )
            for name, param in mlp.named_parameters():
                if param.dtype == torch.uint8:
                    param.data.copy_(torch.randint(
                        0, 256, param.shape, dtype=torch.uint8))
                elif param.dtype == torch.float8_e4m3fn:
                    param.data.copy_(torch.randint(
                        1, 127, param.shape,
                        dtype=torch.uint8).view(torch.float8_e4m3fn))
                elif param.dtype == torch.float32:
                    param.data.fill_(1.0)
                elif param.is_floating_point():
                    param.data.copy_(torch.randn_like(
                        param, dtype=torch.float32).mul_(0.02).to(
                            param.dtype))
            mlp = mlp.to(device=device)
            for module in mlp.modules():
                if hasattr(module, 'quant_method') and hasattr(
                    module.quant_method, 'process_weights_after_loading'
                ):
                    module.quant_method.process_weights_after_loading(
                        module)

            num_tokens = 16
            if input_pattern == "zeros":
                hidden = torch.zeros(num_tokens, config.hidden_size,
                                      dtype=dtype, device=device)
            elif input_pattern == "moderate":
                hidden = torch.full(
                    (num_tokens, config.hidden_size), 10.0,
                    dtype=dtype, device=device)
            elif input_pattern == "tiny":
                hidden = torch.full(
                    (num_tokens, config.hidden_size), 1e-7,
                    dtype=dtype, device=device)
            elif input_pattern == "identical":
                row = torch.randn(1, config.hidden_size, dtype=dtype,
                                   device=device) * 0.02
                hidden = row.expand(num_tokens, -1).contiguous()

            output = mlp(hidden)
            torch.cuda.synchronize()

            assert not torch.isnan(output).any(), (
                f"NVFP4 dense MLP produced NaN with "
                f"input_pattern='{input_pattern}'"
            )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "weight_pattern",
    [
        "zero_global_scale",        # global scale = 0 → division by zero
        "huge_global_scale",        # global scale = 1e30 → overflow
        "tiny_global_scale",        # global scale = 1e-30 → underflow
        "max_fp4_weights",          # all FP4 values maxed out (0x77)
        "zero_fp4_weights",         # all FP4 values zero (0x00)
        "max_fp8_block_scales",     # all block scales at fp8 max (448)
        "zero_fp8_block_scales",    # all block scales zero
        "nan_fp8_block_scales",     # NaN in block scales (0xFF = NaN in e4m3)
        "mixed_extreme_scales",     # some blocks max scale, some zero
        "tiny_act_global_scale",    # activation global scale near zero
        "huge_act_global_scale",    # activation global scale huge
        "zero_act_global_scale",    # activation global scale = 0
        "mismatched_act_scales",    # a1_gscale tiny, a2_gscale huge
        "subnormal_fp8_scales",     # FP8 block scales at subnormal (0x01)
        "min_normal_fp8_scales",    # FP8 block scales at min normal
        "all_scales_tiny",          # both global and block scales tiny
    ],
)
def test_nvfp4_moe_weight_edge_cases(workspace_init, weight_pattern):
    """Test NVFP4 MoE with adversarial weight/scale configurations.

    These test degenerate NVFP4 quantization states that could arise
    from checkpoint corruption, quantization edge cases, or padding
    in weight tensors.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
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
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernel,
        MoEActivation,
    )

    device = torch.device(f"{DEVICE_TYPE}:0")
    E, topk, K, N = 4, 2, 1024, 512
    num_tokens = 8

    # Start with valid weights
    (_, w1_q, w1_bs, w1_gs), (_, w2_q, w2_bs, w2_gs) = (
        make_test_weights(E, N, K, in_dtype=torch.bfloat16,
                          quant_dtype="nvfp4")
    )

    # Modify weights/scales based on pattern
    if weight_pattern == "zero_global_scale":
        w1_gs = torch.zeros_like(w1_gs)
        w2_gs = torch.zeros_like(w2_gs)
    elif weight_pattern == "huge_global_scale":
        w1_gs = torch.full_like(w1_gs, 1e30)
        w2_gs = torch.full_like(w2_gs, 1e30)
    elif weight_pattern == "tiny_global_scale":
        w1_gs = torch.full_like(w1_gs, 1e-30)
        w2_gs = torch.full_like(w2_gs, 1e-30)
    elif weight_pattern == "max_fp4_weights":
        # 0x77 = both nibbles 0111 = max positive FP4 E2M1 value
        w1_q.fill_(0x77)
        w2_q.fill_(0x77)
    elif weight_pattern == "zero_fp4_weights":
        w1_q.fill_(0x00)
        w2_q.fill_(0x00)
    elif weight_pattern == "max_fp8_block_scales":
        # 0x7E = 448.0 in fp8 e4m3fn (max finite value)
        w1_bs.view(torch.uint8).fill_(0x7E)
        w2_bs.view(torch.uint8).fill_(0x7E)
    elif weight_pattern == "zero_fp8_block_scales":
        w1_bs.view(torch.uint8).fill_(0x00)
        w2_bs.view(torch.uint8).fill_(0x00)
    elif weight_pattern == "nan_fp8_block_scales":
        # 0xFF = NaN in fp8 e4m3fn
        w1_bs.view(torch.uint8).fill_(0xFF)
        w2_bs.view(torch.uint8).fill_(0xFF)
    elif weight_pattern == "mixed_extreme_scales":
        # Alternate between max and zero block scales
        w1_bs_bytes = w1_bs.view(torch.uint8)
        w1_bs_bytes[::2] = 0x7E  # max
        w1_bs_bytes[1::2] = 0x00  # zero

    a1_gs = torch.ones((E,), device=device, dtype=torch.float32)
    a2_gs = torch.ones((E,), device=device, dtype=torch.float32)

    # Activation global scale patterns
    if weight_pattern == "tiny_act_global_scale":
        a1_gs.fill_(1e-20)
        a2_gs.fill_(1e-20)
    elif weight_pattern == "huge_act_global_scale":
        a1_gs.fill_(1e20)
        a2_gs.fill_(1e20)
    elif weight_pattern == "zero_act_global_scale":
        a1_gs.fill_(0.0)
        a2_gs.fill_(0.0)
    elif weight_pattern == "mismatched_act_scales":
        a1_gs.fill_(1e-20)
        a2_gs.fill_(1e20)
    elif weight_pattern == "subnormal_fp8_scales":
        # 0x01 in fp8 e4m3fn = smallest subnormal ≈ 2^-9 ≈ 0.001953
        w1_bs.view(torch.uint8).fill_(0x01)
        w2_bs.view(torch.uint8).fill_(0x01)
    elif weight_pattern == "min_normal_fp8_scales":
        # 0x08 in fp8 e4m3fn = smallest normal = 2^-6 = 0.015625
        w1_bs.view(torch.uint8).fill_(0x08)
        w2_bs.view(torch.uint8).fill_(0x08)
    elif weight_pattern == "all_scales_tiny":
        # Everything tiny: global weight scales, act scales, block scales
        w1_gs = torch.full_like(w1_gs, 1e-10)
        w2_gs = torch.full_like(w2_gs, 1e-10)
        w1_bs.view(torch.uint8).fill_(0x01)  # subnormal fp8
        w2_bs.view(torch.uint8).fill_(0x01)
        a1_gs.fill_(1e-10)
        a2_gs.fill_(1e-10)

    # Avoid division by zero in alpha computation
    safe_w1_gs = w1_gs.clamp(min=1e-10)
    safe_w2_gs = w2_gs.clamp(min=1e-10)

    quant_config = nvfp4_moe_quant_config(
        g1_alphas=(1 / safe_w1_gs), g2_alphas=(1 / safe_w2_gs),
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

    hidden = torch.randn(num_tokens, K, device=device,
                           dtype=torch.bfloat16) * 0.1

    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32,
                            device=device)
    for ki in range(topk):
        topk_ids[:, ki] = ki
    topk_weights = torch.ones(num_tokens, topk, dtype=torch.float32,
                               device=device) / topk

    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(pipeline_parallel_size=1))

    with set_current_vllm_config(vllm_cfg):
        output = kernel.apply(
            hidden_states=hidden, w1=w1_q, w2=w2_q,
            topk_weights=topk_weights, topk_ids=topk_ids,
            global_num_experts=E, activation=MoEActivation.SILU,
            apply_router_weight_on_input=False, expert_map=None,
        )
    torch.cuda.synchronize()

    # For some patterns (NaN scales, zero scales), NaN or Inf output
    # may be expected. The key check is: no crash, no hang.
    # For patterns that should produce finite output, check for NaN.
    # Patterns that legitimately produce NaN/Inf (invalid quant states)
    expect_nan = weight_pattern in (
        "nan_fp8_block_scales",
        "zero_act_global_scale",   # 0 scale → inf in dequant
        "huge_act_global_scale",   # huge scale → overflow
    )
    if expect_nan:
        # Verify no crash/hang — NaN output is expected
        pass
    elif weight_pattern in ("zero_global_scale", "zero_fp8_block_scales",
                            "zero_fp4_weights"):
        # Zero weights/scales → zero or near-zero output expected
        assert not torch.isnan(output).any(), (
            f"NVFP4 MoE: unexpected NaN with {weight_pattern}"
        )
    else:
        assert not torch.isnan(output).any(), (
            f"NVFP4 MoE: NaN with weight_pattern='{weight_pattern}'"
        )


# ============================================================================
# SM pressure tests for MoE and dense MLP
# ============================================================================


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16)],
    ids=["1to8", "5to8", "13to16"],
)
def test_nvfp4_moe_cudagraph_sm_pressure(
    workspace_init, sm_pressure, num_real, num_padded,
):
    """NVFP4 MoE (CutlassExpertsFp4) under CUDA graph with SM pressure.
    1000 replays with background kernel contention. The production
    FlashInferCuteDSLBatchedExperts kernel is tested through multi-GPU
    DeepEP tests."""
    from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
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
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernel,
        MoEActivation,
    )

    device = torch.device(f"{DEVICE_TYPE}:0")
    E, topk, K, N = 8, 2, 1024, 512

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

    gate_weight = torch.randn(E, K, device=device,
                               dtype=torch.bfloat16) * 0.02
    hidden_states = torch.randn(
        num_padded, K, device=device, dtype=torch.bfloat16) * 0.1
    hidden_states[num_real:] = float('nan')

    router_logits = hidden_states @ gate_weight.t()
    routing_weights = torch.softmax(
        router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)

    apply_kwargs = dict(
        hidden_states=hidden_states, w1=w1_q, w2=w2_q,
        topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=E, activation=MoEActivation.SILU,
        apply_router_weight_on_input=False, expert_map=None,
    )

    _poison_cuda_allocator(device)

    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(pipeline_parallel_size=1))

    with set_current_vllm_config(vllm_cfg):
        output = kernel.apply(**apply_kwargs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = kernel.apply(**apply_kwargs)

        saved_hidden = hidden_states[:num_real].clone()
        saved_tw = topk_weights[:num_real].clone()
        saved_ti = topk_ids[:num_real].clone()

        for i in range(1000):
            hidden_states.fill_(float('nan'))
            topk_weights.fill_(float('nan'))
            topk_ids.fill_(0)
            output.fill_(float('nan'))
            graph.replay()

            hidden_states[:num_real] = saved_hidden
            hidden_states[num_real:] = float('nan')
            topk_weights[:num_real] = saved_tw
            topk_ids[:num_real] = saved_ti
            output.fill_(float('nan'))
            graph.replay()

            if i % 100 == 99:
                torch.cuda.synchronize()
                assert not torch.isnan(output[:num_real]).any(), (
                    f"SM pressure NVFP4 MoE: NaN at iteration {i}")


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires sm100+",
)
@pytest.mark.parametrize(
    "num_real,num_padded",
    [(1, 8), (5, 8), (13, 16)],
    ids=["1to8", "5to8", "13to16"],
)
def test_nvfp4_dense_mlp_cudagraph_sm_pressure(
    default_vllm_config, dist_init, sm_pressure, num_real, num_padded,
):
    """NVFP4 dense MLP under CUDA graph with SM pressure.
    1000 replays with background kernel contention."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

    device = torch.device(f"{DEVICE_TYPE}:0")
    dtype = torch.bfloat16

    vllm_config = create_vllm_config(
        model_name="nvidia/DeepSeek-R1-0528-NVFP4-v2",
        max_model_len=128, num_gpu_blocks=8192, dtype="bfloat16",
    )
    config = vllm_config.model_config.hf_config
    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None, exclude_modules=[],
    )

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with set_current_vllm_config(vllm_config):
            mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )
            for name, param in mlp.named_parameters():
                if param.dtype == torch.uint8:
                    param.data.copy_(torch.randint(
                        0, 256, param.shape, dtype=torch.uint8))
                elif param.dtype == torch.float8_e4m3fn:
                    param.data.copy_(torch.randint(
                        1, 127, param.shape,
                        dtype=torch.uint8).view(torch.float8_e4m3fn))
                elif param.dtype == torch.float32:
                    param.data.fill_(1.0)
                elif param.is_floating_point():
                    param.data.copy_(torch.randn_like(
                        param, dtype=torch.float32).mul_(0.02).to(
                            param.dtype))
            mlp = mlp.to(device=device)
            for module in mlp.modules():
                if hasattr(module, 'quant_method') and hasattr(
                    module.quant_method, 'process_weights_after_loading'
                ):
                    module.quant_method.process_weights_after_loading(
                        module)

            hidden = torch.randn(
                num_padded, config.hidden_size, dtype=dtype,
                device=device) * 0.02
            hidden[num_real:] = float('nan')

            _poison_cuda_allocator(device)

            output = mlp(hidden)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = mlp(hidden)

            saved_hidden = hidden[:num_real].clone()

            for i in range(1000):
                hidden.fill_(float('nan'))
                output.fill_(float('nan'))
                graph.replay()

                hidden[:num_real] = saved_hidden
                hidden[num_real:] = float('nan')
                output.fill_(float('nan'))
                graph.replay()

                if i % 100 == 99:
                    torch.cuda.synchronize()
                    assert not torch.isnan(output[:num_real]).any(), (
                        f"SM pressure NVFP4 dense MLP: NaN at "
                        f"iteration {i}")
    finally:
        torch.set_default_dtype(old_dtype)


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

    try:
        if use_nvfp4:
            import vllm.envs as envs_mod
            _orig_nvfp4_dispatch = getattr(
                envs_mod, 'VLLM_DEEPEPLL_NVFP4_DISPATCH', False)
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
                                  in_dtype=torch.bfloat16,
                                  quant_dtype="nvfp4")
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
        topk_weights, topk_ids = torch.topk(
            routing_weights, topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True)
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
            prepare_finalize=a2a, fused_experts=fused_experts,
            inplace=False,
        )

        from tests.v1.attention.test_cudagraph_nan_padding import (
            _poison_cuda_allocator,
        )
        _poison_cuda_allocator(device)

        apply_kwargs = dict(
            hidden_states=hidden_states, w1=w1, w2=w2,
            topk_weights=topk_weights, topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )

        # SM pressure: background kernel on separate stream
        stop_event = threading.Event()
        hog_stream = torch.cuda.Stream(device=device)

        def _sm_hog():
            with torch.cuda.stream(hog_stream):
                a = torch.randn(2048, 2048, device=device,
                                 dtype=torch.bfloat16)
                b = torch.randn(2048, 2048, device=device,
                                 dtype=torch.bfloat16)
                while not stop_event.is_set():
                    torch.mm(a, b, out=a)

        hog_thread = threading.Thread(target=_sm_hog, daemon=True)
        hog_thread.start()

        with set_current_vllm_config(VllmConfig()):
            # Warmup
            output = kernel.apply(**apply_kwargs)
            torch.cuda.synchronize()

            # Capture in CUDA graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = kernel.apply(**apply_kwargs)

            # Multiple replays with pollution under SM pressure
            saved_hidden = hidden_states[:num_real].clone()
            saved_tw = topk_weights[:num_real].clone()
            saved_ti = topk_ids[:num_real].clone()

            for i in range(1000):
                # Pollute → replay
                hidden_states.fill_(float('nan'))
                topk_weights.fill_(float('nan'))
                topk_ids.fill_(0)
                output.fill_(float('nan'))
                graph.replay()

                # Restore → replay
                hidden_states[:num_real] = saved_hidden
                hidden_states[num_real:] = float('nan')
                topk_weights[:num_real] = saved_tw
                topk_ids[:num_real] = saved_ti
                output.fill_(float('nan'))
                graph.replay()

                if i % 100 == 99:
                    torch.cuda.synchronize()
                    mode = "NVFP4" if use_nvfp4 else "FP8"
                    real_output = output[:num_real]
                    assert not torch.isnan(real_output).any(), (
                        f"DeepEP LL {mode} MoE: NaN at iteration {i} "
                        f"on rank {pgi.rank} "
                        f"(real={num_real}, padded={num_padded})"
                    )

    finally:
        stop_event.set()
        hog_thread.join(timeout=5)
        if use_nvfp4:
            envs_mod.VLLM_DEEPEPLL_NVFP4_DISPATCH = _orig_nvfp4_dispatch


# Smaller configs than single-GPU tests to avoid OOM during CUDA graph
# capture in multi-GPU workers (each worker runs on a single GPU with
# workspace + graph memory overhead).
DEEPEP_PADDING_CONFIGS = [
    (1, 8),
    (5, 8),
    (13, 16),
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
