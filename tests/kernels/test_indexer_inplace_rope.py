# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that in-place RoPE on tensor slices produces identical results to
the old split -> RoPE -> cat pattern used by the DeepSeek V3.2 Indexer.

The optimization eliminates two torch.cat calls per layer (CatArrayBatchedCopy
kernel, ~89us each) by relying on the fact that the CUDA RoPE implementation
modifies q and k in-place.
"""

import pytest
import torch

from vllm import _custom_ops as ops

N_HEAD = 64
HEAD_DIM = 128
ROPE_DIM = 64
MAX_POS = 4096
BASE = 10000.0

NUM_TOKENS = [1, 4, 16, 64, 256]


def _build_cos_sin_cache(
    rope_dim: int, max_pos: int, base: float, dtype: torch.dtype
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype=dtype, device="cuda")


def _old_split_cat_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original Indexer forward pattern: split -> in-place CUDA RoPE -> cat."""
    n_head = q.shape[1]
    q_pe, q_nope = torch.split(
        q, [rope_dim, q.shape[-1] - rope_dim], dim=-1
    )
    k_pe, k_nope = torch.split(
        k, [rope_dim, k.shape[-1] - rope_dim], dim=-1
    )

    # Use the same CUDA kernel as the new path for apples-to-apples comparison
    q_pe = q_pe.contiguous()
    k_pe_3d = k_pe.unsqueeze(1).contiguous()
    ops.rotary_embedding(
        positions, q_pe, k_pe_3d, rope_dim, cos_sin_cache, is_neox_style,
    )

    q_out = torch.cat([q_pe, q_nope], dim=-1)
    k_out = torch.cat([k_pe_3d.squeeze(1), k_nope], dim=-1)
    return q_out, k_out


def _new_inplace_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """New Indexer forward pattern: in-place CUDA RoPE on slices."""
    q_pe_slice = q[:, :, :rope_dim]
    k_pe_slice = k[:, :rope_dim].unsqueeze(1)
    ops.rotary_embedding(
        positions,
        q_pe_slice,
        k_pe_slice,
        rope_dim,
        cos_sin_cache,
        is_neox_style,
    )
    return q, k


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("is_neox_style", [True, False])
@torch.inference_mode()
def test_inplace_rope_matches_split_cat(num_tokens, is_neox_style):
    """The in-place slice approach must produce bit-identical results to
    the old split -> RoPE -> cat pattern."""
    dtype = torch.bfloat16
    cos_sin_cache = _build_cos_sin_cache(ROPE_DIM, MAX_POS, BASE, dtype)

    torch.manual_seed(42)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device="cuda")

    # q: [T, n_head, head_dim] with layout [pe(64), nope(64)]
    q_orig = torch.randn(
        num_tokens, N_HEAD, HEAD_DIM, dtype=dtype, device="cuda"
    )
    # k: [T, head_dim] with layout [pe(64), nope(64)]
    k_orig = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device="cuda")

    # Reference: split -> native RoPE -> cat
    q_ref, k_ref = _old_split_cat_rope(
        q_orig.clone(), k_orig.clone(), positions, cos_sin_cache,
        ROPE_DIM, is_neox_style,
    )

    # New: in-place CUDA RoPE on slices
    q_new = q_orig.clone()
    k_new = k_orig.clone()
    q_act, k_act = _new_inplace_rope(
        q_new, k_new, positions, cos_sin_cache,
        ROPE_DIM, is_neox_style,
    )

    torch.testing.assert_close(q_act, q_ref, atol=0, rtol=0)
    torch.testing.assert_close(k_act, k_ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_nope_dims_unchanged(num_tokens):
    """The nope portion of q and k must not be modified by in-place RoPE."""
    dtype = torch.bfloat16
    cos_sin_cache = _build_cos_sin_cache(ROPE_DIM, MAX_POS, BASE, dtype)

    torch.manual_seed(42)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device="cuda")

    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device="cuda")

    q_nope_before = q[:, :, ROPE_DIM:].clone()
    k_nope_before = k[:, ROPE_DIM:].clone()

    _new_inplace_rope(q, k, positions, cos_sin_cache, ROPE_DIM, True)

    torch.testing.assert_close(
        q[:, :, ROPE_DIM:], q_nope_before, atol=0, rtol=0
    )
    torch.testing.assert_close(k[:, ROPE_DIM:], k_nope_before, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_pe_dims_actually_change(num_tokens):
    """Sanity check: the pe portion IS modified (RoPE is not a no-op)."""
    dtype = torch.bfloat16
    cos_sin_cache = _build_cos_sin_cache(ROPE_DIM, MAX_POS, BASE, dtype)

    torch.manual_seed(42)
    # Use positions > 0 so cos/sin are non-trivial
    positions = torch.randint(1, MAX_POS, (num_tokens,), device="cuda")

    q = torch.randn(num_tokens, N_HEAD, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device="cuda")

    q_pe_before = q[:, :, :ROPE_DIM].clone()
    k_pe_before = k[:, :ROPE_DIM].clone()

    _new_inplace_rope(q, k, positions, cos_sin_cache, ROPE_DIM, True)

    assert not torch.equal(q[:, :, :ROPE_DIM], q_pe_before)
    assert not torch.equal(k[:, :ROPE_DIM], k_pe_before)


@torch.inference_mode()
def test_inplace_rope_from_gemm_output():
    """Simulate the actual Indexer data flow: q from GEMM -> view, k from
    GEMM slice -> LayerNorm. Verifies that in-place RoPE on slices of these
    tensors produces correct results."""
    dtype = torch.bfloat16
    cos_sin_cache = _build_cos_sin_cache(ROPE_DIM, MAX_POS, BASE, dtype)
    num_tokens = 32

    torch.manual_seed(42)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device="cuda")

    # Simulate: q comes from a GEMM and is reshaped
    q_flat = torch.randn(
        num_tokens, N_HEAD * HEAD_DIM, dtype=dtype, device="cuda"
    )
    q = q_flat.view(-1, N_HEAD, HEAD_DIM)

    # Simulate: k comes from a GEMM slice + LayerNorm
    kw = torch.randn(num_tokens, HEAD_DIM + N_HEAD, dtype=dtype, device="cuda")
    k_raw = kw[:, :HEAD_DIM].contiguous()
    k = torch.nn.functional.layer_norm(k_raw, [HEAD_DIM])

    # Reference
    q_ref, k_ref = _old_split_cat_rope(
        q.clone(), k.clone(), positions, cos_sin_cache, ROPE_DIM, True,
    )

    # In-place
    _new_inplace_rope(q, k, positions, cos_sin_cache, ROPE_DIM, True)

    torch.testing.assert_close(q, q_ref, atol=0, rtol=0)
    torch.testing.assert_close(k, k_ref, atol=0, rtol=0)


@torch.inference_mode()
def test_inplace_rope_zero_tokens():
    """Edge case: zero tokens should not crash."""
    dtype = torch.bfloat16
    cos_sin_cache = _build_cos_sin_cache(ROPE_DIM, MAX_POS, BASE, dtype)

    positions = torch.empty(0, dtype=torch.long, device="cuda")
    q = torch.empty(0, N_HEAD, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.empty(0, HEAD_DIM, dtype=dtype, device="cuda")

    _new_inplace_rope(q, k, positions, cos_sin_cache, ROPE_DIM, True)
    assert q.shape == (0, N_HEAD, HEAD_DIM)
    assert k.shape == (0, HEAD_DIM)
