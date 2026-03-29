# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test MLA attention kernel for NaN-producing edge cases.

The FlashInfer trtllm_batch_decode_with_kv_cache_mla kernel does softmax
internally. This test checks whether specific input patterns can produce
NaN in the output for real tokens.
"""

import pytest
import torch


def has_flashinfer_mla():
    try:
        from flashinfer.decode import (  # noqa: F401
            trtllm_batch_decode_with_kv_cache_mla,
        )
        return True
    except ImportError:
        return False


requires_flashinfer_mla = pytest.mark.skipif(
    not has_flashinfer_mla(),
    reason="flashinfer MLA kernel not available",
)


# DeepSeek R1 MLA dims
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
# For the FlashInfer MLA kernel, query head_dim must equal
# kv_lora_rank + qk_rope_head_dim (the compressed KV entry size).
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
NUM_HEADS = 16  # per-rank heads (128 total / 8 TP)
BLOCK_SIZE = 64
DTYPE = torch.bfloat16


WORKSPACE_SIZE = 128 * 1024 * 1024  # 128 MB


def make_mla_decode_inputs(
    batch_size: int,
    seq_lens: list[int],
    num_blocks: int,
    device: str = "cuda",
):
    """Create inputs for trtllm_batch_decode_with_kv_cache_mla."""
    max_seq_len = max(seq_lens)
    max_blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    # FlashInfer requires block_num % (128 / block_size) == 0
    block_table_pad = int(128 // BLOCK_SIZE)
    max_blocks_per_seq = (
        (max_blocks_per_seq + block_table_pad - 1)
        // block_table_pad
        * block_table_pad
    )

    # Query: [batch, 1, num_heads, qk_head_dim]
    q = torch.randn(
        batch_size, 1, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=device
    )

    # KV cache: [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
    kv_entry_size = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    # Allocate enough blocks for padded block tables
    total_blocks = max(num_blocks, batch_size * max_blocks_per_seq)
    kv_cache = torch.randn(
        total_blocks, 1, BLOCK_SIZE, kv_entry_size, dtype=DTYPE, device=device
    )

    # Block tables: [batch, max_blocks_per_seq]
    block_tables = torch.zeros(
        batch_size, max_blocks_per_seq, dtype=torch.int32, device=device
    )
    block_idx = 0
    for i, sl in enumerate(seq_lens):
        n_blocks = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in range(n_blocks):
            block_tables[i, j] = block_idx
            block_idx += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    workspace = torch.zeros(
        WORKSPACE_SIZE, dtype=torch.uint8, device=device
    )

    return q, kv_cache, block_tables, seq_lens_t, max_seq_len, workspace


@requires_flashinfer_mla
@pytest.mark.parametrize("seq_len", [1, 16, 128, 1024])
@torch.inference_mode()
def test_mla_decode_clean_inputs(seq_len):
    """MLA decode with normal random inputs should never produce NaN."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    assert torch.isfinite(o).all(), (
        f"NaN/Inf in MLA decode output with clean inputs, seq_len={seq_len}"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_large_qk_values():
    """Large Q/K values that could cause QK dot product overflow."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 2
    seq_len = 64
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    # Scale up Q and K to near bf16 max to stress overflow
    q *= 100.0
    kv_cache *= 100.0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    assert torch.isfinite(o).all(), (
        "NaN/Inf in MLA decode output with large Q/K values"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_nan_in_kv_cache():
    """NaN in KV cache entries should not produce NaN at unrelated tokens."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 2
    seq_len = 64
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    # Poison one block that belongs to request 1 only
    req1_first_block = block_tables[1, 0].item()
    kv_cache[req1_first_block] = float("nan")

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    # Request 0 should be clean (its KV cache is fine)
    assert torch.isfinite(o[0]).all(), (
        "NaN leaked from request 1's poisoned KV cache to request 0"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_seq_len_1():
    """seq_len=1 means only one KV entry — softmax denominator could be tricky."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_lens = [1] * batch_size
    num_blocks = batch_size

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    assert torch.isfinite(o).all(), (
        "NaN/Inf in MLA decode with seq_len=1"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_mixed_seq_lens():
    """Mixed sequence lengths with padding in the batch."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    seq_lens = [1, 128, 3, 1024]
    batch_size = len(seq_lens)
    num_blocks = sum(
        (sl + BLOCK_SIZE - 1) // BLOCK_SIZE for sl in seq_lens
    )

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    assert torch.isfinite(o).all(), (
        "NaN/Inf in MLA decode with mixed seq_lens"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_zero_kv_cache():
    """All-zero KV cache — softmax gets uniform weights, output should be zero."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 2
    seq_len = 32
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(batch_size, seq_lens, num_blocks)
    )

    kv_cache.zero_()

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    assert torch.isfinite(o).all(), (
        "NaN/Inf in MLA decode with all-zero KV cache"
    )


# ---- CUDA graph simulation tests ----


@requires_flashinfer_mla
@pytest.mark.parametrize("num_real", [1, 4, 8])
@torch.inference_mode()
def test_mla_decode_cuda_graph_nan_padding_q(num_real):
    """Simulate CUDA graph: padding tokens have NaN Q values.

    In CUDA graph mode, the query buffer is sized for max_batch but only
    num_real tokens are real. Padding Q values may contain NaN from
    previous iterations. The kernel must not let padding NaN leak into
    real token outputs.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    max_batch = 16
    seq_len = 128
    seq_lens = [seq_len] * max_batch
    num_blocks = max_batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(max_batch, seq_lens, num_blocks)
    )

    # Poison padding Q values with NaN
    q[num_real:] = float("nan")
    # Set padding seq_lens to 0
    seq_lens_t[num_real:] = 0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    real_output = o[:num_real]
    assert torch.isfinite(real_output).all(), (
        f"NaN leaked from padding Q into real token outputs "
        f"(num_real={num_real}, max_batch={max_batch})"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_cuda_graph_inf_padding_q():
    """Padding Q with Inf — extreme QK dot products from padding."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    max_batch = 16
    num_real = 4
    seq_len = 128
    seq_lens = [seq_len] * max_batch
    num_blocks = max_batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(max_batch, seq_lens, num_blocks)
    )

    q[num_real:] = float("inf")
    seq_lens_t[num_real:] = 0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    real_output = o[:num_real]
    assert torch.isfinite(real_output).all(), (
        "NaN/Inf leaked from Inf padding Q into real token outputs"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_cuda_graph_garbage_padding_q():
    """Padding Q with random large values (stale buffer content)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    max_batch = 16
    num_real = 4
    seq_len = 128
    seq_lens = [seq_len] * max_batch
    num_blocks = max_batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(max_batch, seq_lens, num_blocks)
    )

    # Simulate stale buffer: padding has large random values
    q[num_real:] = torch.randn_like(q[num_real:]) * 1000.0
    seq_lens_t[num_real:] = 0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
        bmm2_scale=1.0,
    )

    real_output = o[:num_real]
    assert torch.isfinite(real_output).all(), (
        "NaN/Inf leaked from stale padding Q into real token outputs"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_cuda_graph_replay():
    """Simulate CUDA graph capture + replay with changing batch sizes."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    max_batch = 16
    seq_len = 128
    num_blocks = max_batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    # Allocate fixed buffers (as CUDA graph would)
    seq_lens_all = [seq_len] * max_batch
    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws = (
        make_mla_decode_inputs(max_batch, seq_lens_all, num_blocks)
    )

    def run_with_real_tokens(num_real):
        # Fresh Q for real tokens, NaN for padding
        q[:num_real].normal_()
        q[num_real:] = float("nan")
        seq_lens_t[:num_real] = seq_len
        seq_lens_t[num_real:] = 0

        o = trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=ws,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens_t,
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (QK_HEAD_DIM ** 0.5),
            bmm2_scale=1.0,
        )
        return o

    # Simulate multiple "replays" with different real token counts
    for num_real in [16, 8, 4, 1, 12]:
        o = run_with_real_tokens(num_real)
        real_output = o[:num_real]
        assert torch.isfinite(real_output).all(), (
            f"NaN leaked during simulated graph replay "
            f"(num_real={num_real})"
        )


# ---- FP8 KV cache tests ----


def make_mla_decode_inputs_fp8(
    batch_size: int,
    seq_lens: list[int],
    num_blocks: int,
    kv_scale: float = 1.0,
    device: str = "cuda",
):
    """Create inputs with FP8 KV cache for trtllm_batch_decode_with_kv_cache_mla."""
    max_seq_len = max(seq_lens)
    max_blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table_pad = int(128 // BLOCK_SIZE)
    max_blocks_per_seq = (
        (max_blocks_per_seq + block_table_pad - 1)
        // block_table_pad
        * block_table_pad
    )

    q = torch.randn(
        batch_size, 1, NUM_HEADS, QK_HEAD_DIM, dtype=DTYPE, device=device
    )

    kv_entry_size = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    total_blocks = max(num_blocks, batch_size * max_blocks_per_seq)

    # Create bf16 KV data, scale it, then quantize to FP8
    kv_bf16 = torch.randn(
        total_blocks, 1, BLOCK_SIZE, kv_entry_size, dtype=DTYPE, device=device
    )
    kv_cache = kv_bf16.to(torch.float8_e4m3fn)

    block_tables = torch.zeros(
        batch_size, max_blocks_per_seq, dtype=torch.int32, device=device
    )
    block_idx = 0
    for i, sl in enumerate(seq_lens):
        n_blocks = (sl + BLOCK_SIZE - 1) // BLOCK_SIZE
        for j in range(n_blocks):
            block_tables[i, j] = block_idx
            block_idx += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    workspace = torch.zeros(
        WORKSPACE_SIZE, dtype=torch.uint8, device=device
    )

    # bmm scales for FP8: scale = 1/qk_head_dim^0.5 * q_scale * k_scale
    # Use kv_scale as the combined q/k scale factor
    bmm1_scale = (1.0 / (QK_HEAD_DIM ** 0.5)) * kv_scale
    bmm2_scale = kv_scale

    return (
        q, kv_cache, block_tables, seq_lens_t, max_seq_len, workspace,
        bmm1_scale, bmm2_scale,
    )


@requires_flashinfer_mla
@pytest.mark.parametrize("seq_len", [1, 128, 1024])
@torch.inference_mode()
def test_mla_decode_fp8_clean(seq_len):
    """FP8 KV cache with clean inputs should not produce NaN."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws, s1, s2 = (
        make_mla_decode_inputs_fp8(batch_size, seq_lens, num_blocks)
    )

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=s1,
        bmm2_scale=s2,
    )

    assert torch.isfinite(o).all(), (
        f"NaN/Inf with FP8 KV cache, seq_len={seq_len}"
    )


@requires_flashinfer_mla
@pytest.mark.parametrize("kv_scale", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
@torch.inference_mode()
def test_mla_decode_fp8_extreme_scales(kv_scale):
    """FP8 KV cache with extreme dequant scales — can overflow during BMM."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_len = 128
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws, s1, s2 = (
        make_mla_decode_inputs_fp8(
            batch_size, seq_lens, num_blocks, kv_scale=kv_scale
        )
    )

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=s1,
        bmm2_scale=s2,
    )

    assert torch.isfinite(o).all(), (
        f"NaN/Inf with FP8 KV cache, kv_scale={kv_scale}"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_fp8_nan_in_cache():
    """NaN in FP8 KV cache block for one request must not leak to others."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_len = 128
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws, s1, s2 = (
        make_mla_decode_inputs_fp8(batch_size, seq_lens, num_blocks)
    )

    # Poison request 2's first block with FP8 NaN (0x7F)
    req2_block = block_tables[2, 0].item()
    kv_cache[req2_block] = torch.tensor(
        float("nan"), dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=s1,
        bmm2_scale=s2,
    )

    # Other requests should be clean
    for i in [0, 1, 3]:
        assert torch.isfinite(o[i]).all(), (
            f"FP8 NaN leaked from request 2 to request {i}"
        )


@requires_flashinfer_mla
@pytest.mark.parametrize("num_real", [1, 4, 8])
@torch.inference_mode()
def test_mla_decode_fp8_cuda_graph_nan_padding(num_real):
    """FP8 KV cache + CUDA graph padding with NaN Q values."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    max_batch = 16
    seq_len = 128
    seq_lens = [seq_len] * max_batch
    num_blocks = max_batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws, s1, s2 = (
        make_mla_decode_inputs_fp8(max_batch, seq_lens, num_blocks)
    )

    # NaN padding Q + zero seq_lens for padding
    q[num_real:] = float("nan")
    seq_lens_t[num_real:] = 0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=s1,
        bmm2_scale=s2,
    )

    real_output = o[:num_real]
    assert torch.isfinite(real_output).all(), (
        f"FP8: NaN leaked from padding Q to real tokens "
        f"(num_real={num_real})"
    )


@requires_flashinfer_mla
@torch.inference_mode()
def test_mla_decode_fp8_max_fp8_values():
    """FP8 KV cache filled with max representable FP8 value (448.0).

    When dequantized with a scale, QK dot products could overflow.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    batch_size = 4
    seq_len = 128
    seq_lens = [seq_len] * batch_size
    num_blocks = batch_size * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    q, kv_cache, block_tables, seq_lens_t, max_seq_len, ws, s1, s2 = (
        make_mla_decode_inputs_fp8(batch_size, seq_lens, num_blocks)
    )

    # Fill KV cache with max FP8 e4m3fn value
    max_fp8 = torch.tensor(448.0, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    kv_cache.fill_(max_fp8)

    # Also use large Q values
    q *= 10.0

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens_t,
        max_seq_len=max_seq_len,
        bmm1_scale=s1,
        bmm2_scale=s2,
    )

    assert torch.isfinite(o).all(), (
        "NaN/Inf with max FP8 KV values + large Q"
    )
