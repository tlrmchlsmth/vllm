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
