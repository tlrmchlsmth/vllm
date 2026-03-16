# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that CUDA graph padding uses seq_lens=1 instead of seq_lens=0.

With seq_lens=0, attention kernels compute softmax over zero elements,
producing NaN/Inf. These values contaminate real tokens via quantized
GEMM tiles (e.g., NVFP4 128-row tiles), causing progressive accuracy
degradation.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers



@pytest.mark.cpu_test
@pytest.mark.parametrize("num_reqs,max_num_reqs", [
    (1, 4),
    (3, 8),
    (7, 16),
    (15, 32),
])
def test_input_batch_padding_seq_lens_nonzero(num_reqs: int,
                                               max_num_reqs: int):
    """Verify InputBatch.prepare_pos_seq_lens sets seq_lens >= 1 for padding.

    Padding requests (indices num_reqs..max_num_reqs) must have seq_lens > 0
    to prevent NaN from empty softmax in attention kernels.
    """
    max_num_tokens = max_num_reqs * 4
    num_tokens = num_reqs * 2

    input_buffers = InputBuffers(
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        device=torch.device("cpu"),
    )

    num_scheduled_tokens = np.full(num_reqs,
                                   num_tokens // num_reqs,
                                   dtype=np.int32)
    num_scheduled_tokens[-1] += num_tokens % num_reqs

    InputBatch.prepare_pos_seq_lens(
        input_buffers=input_buffers,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    # Real requests should have their seq_lens set
    real_seq_lens = input_buffers.seq_lens[:num_reqs]
    assert torch.all(real_seq_lens > 0), (
        f"Real request seq_lens should be > 0, got {real_seq_lens}")

    # Padding requests should have seq_lens >= 1, NOT 0
    if max_num_reqs > num_reqs:
        padding_seq_lens = input_buffers.seq_lens[num_reqs:max_num_reqs]
        assert torch.all(padding_seq_lens >= 1), (
            f"Padding seq_lens should be >= 1 to prevent NaN from empty "
            f"softmax, but got {padding_seq_lens}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_attention_nan_with_zero_seq_lens():
    """Demonstrate that attention with seq_lens=0 produces NaN.

    This is the root cause of NVFP4 NaN corruption: CUDA graph padding
    requests with seq_lens=0 cause empty softmax in the attention kernel,
    producing NaN/Inf that contaminates real tokens via quantized GEMM tiles.

    The fix (seq_lens=1) gives the kernel a valid single-element softmax.
    """
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not available")

    device = "cuda"
    dtype = torch.bfloat16
    num_heads = 4
    head_dim = 64
    block_size = 16
    num_blocks = 32
    max_kv_len = 128
    num_real = 2
    num_padding = 2
    num_seqs = num_real + num_padding

    # Each request has 1 query token (decode mode)
    query_lens = [1] * num_seqs
    total_q = sum(query_lens)

    q = torch.randn(total_q, num_heads, head_dim,
                     device=device, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim,
                           device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim,
                           device=device, dtype=dtype)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    scale = head_dim ** -0.5

    # Case 1: padding requests have seq_lens=0 (the bug)
    kv_lens_with_zero = torch.tensor(
        [max_kv_len] * num_real + [0] * num_padding, dtype=torch.int32
    )
    output_zero = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_with_zero,
        max_seqlen_q=1,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
    )

    # Padding positions (indices 2,3) should have NaN from empty softmax
    padding_output = output_zero[num_real:]
    assert torch.any(torch.isnan(padding_output)), (
        "Expected NaN in attention output for seq_lens=0 padding requests, "
        "but got finite values. The attention kernel may handle empty "
        "sequences gracefully on this platform."
    )

    # Real positions should be finite
    real_output = output_zero[:num_real]
    assert not torch.any(torch.isnan(real_output)), (
        "Real request outputs should be finite"
    )

    # Case 2: padding requests have seq_lens=1 (the fix)
    kv_lens_with_one = torch.tensor(
        [max_kv_len] * num_real + [1] * num_padding, dtype=torch.int32
    )
    output_one = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_with_one,
        max_seqlen_q=1,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        block_table=block_tables,
    )

    # With seq_lens=1, ALL outputs should be finite (no NaN)
    assert not torch.any(torch.isnan(output_one)), (
        "With seq_lens=1 fix, all attention outputs should be finite"
    )
    assert not torch.any(torch.isinf(output_one)), (
        "With seq_lens=1 fix, all attention outputs should be finite"
    )
