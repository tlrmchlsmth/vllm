# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for zero-overhead NaN/Inf detection in RMSNorm kernels."""

import pytest
import torch

from vllm.model_executor.layers.nan_detector import NaNDetector


@pytest.fixture(autouse=True)
def reset_nan_detector():
    """Reset the singleton between tests."""
    NaNDetector.reset()
    yield
    NaNDetector.reset()


@pytest.fixture
def device():
    return "cuda:0"


@pytest.mark.parametrize("hidden_size", [64, 128, 256])
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@torch.inference_mode()
def test_nan_detection_rms_norm(default_vllm_config, device, hidden_size, num_tokens):
    """NaN in input should be detected at the correct token position."""
    from vllm import _custom_ops as ops

    num_layers = 3
    max_num_tokens = 32

    nan_flags = torch.zeros(num_layers, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    # Clean input — no flags should be set.
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags.sum().item() == 0, "False positive on clean input"

    # Inject NaN at token 1, layer 0.
    nan_flags.zero_()
    x_nan = x.clone()
    if num_tokens > 1:
        x_nan[1, 0] = float("nan")
        ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 0, max_num_tokens)
        assert nan_flags[0, 1].item() == 1, "NaN not detected at token 1"
        assert nan_flags[0, 0].item() == 0, "False positive at token 0"
    else:
        x_nan[0, 0] = float("nan")
        ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 0, max_num_tokens)
        assert nan_flags[0, 0].item() == 1, "NaN not detected at token 0"

    # Inject NaN at a different layer index.
    nan_flags.zero_()
    ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 2, max_num_tokens)
    assert nan_flags[0].sum().item() == 0, "Wrong layer got the flag"
    assert nan_flags[2].any().item(), "NaN not detected at layer 2"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_inf_detection_rms_norm(default_vllm_config, device, hidden_size):
    """Inf in input should be detected."""
    from vllm import _custom_ops as ops

    num_tokens = 4
    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    x[2, 0] = float("inf")
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags[0, 2].item() == 1, "Inf not detected at token 2"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_nan_detection_fused_add_rms_norm(default_vllm_config, device, hidden_size):
    """NaN detection works with the fused add+norm path."""
    from vllm import _custom_ops as ops

    num_tokens = 4
    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn_like(x)

    # Clean — no flags.
    ops.fused_add_rms_norm(
        x.clone(), residual.clone(), weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags.sum().item() == 0

    # Inject NaN in the input (not residual).
    nan_flags.zero_()
    x_nan = x.clone()
    x_nan[3, 0] = float("nan")
    ops.fused_add_rms_norm(
        x_nan, residual.clone(), weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 3].item() == 1, "NaN not detected at token 3"

    # Inject NaN in the residual.
    nan_flags.zero_()
    res_nan = residual.clone()
    res_nan[0, 0] = float("nan")
    ops.fused_add_rms_norm(
        x.clone(), res_nan, weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 0].item() == 1, "NaN in residual not detected"


@torch.inference_mode()
def test_no_detection_when_disabled(default_vllm_config, device):
    """When nan_flags is None, no detection occurs (null pointer path)."""
    from vllm import _custom_ops as ops

    hidden_size = 64
    num_tokens = 4
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    x[0, 0] = float("nan")
    out = torch.empty_like(x)

    # Should not crash — nan_flags=None means no detection.
    ops.rms_norm(out, x, weight, 1e-6)


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_padding_nan_does_not_leak_rms_norm(
    default_vllm_config, device, hidden_size, add_residual
):
    """NaN in padding positions must not leak into real tokens via RMSNorm."""
    from vllm import _custom_ops as ops

    num_real = 4
    num_padded = 8
    dtype = torch.float16

    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_padded, hidden_size, dtype=dtype, device=device)
    x[num_real:] = float("nan")

    if add_residual:
        residual = torch.randn(
            num_padded, hidden_size, dtype=dtype, device=device
        )
        residual[num_real:] = float("nan")
        ops.fused_add_rms_norm(x, residual, weight, 1e-6)
        assert torch.isfinite(x[:num_real]).all(), (
            "NaN leaked from padding into real tokens via fused_add_rms_norm"
        )
        assert torch.isfinite(residual[:num_real]).all(), (
            "NaN leaked from padding into real residual via fused_add_rms_norm"
        )
    else:
        out = torch.empty_like(x)
        ops.rms_norm(out, x, weight, 1e-6)
        assert torch.isfinite(out[:num_real]).all(), (
            "NaN leaked from padding into real tokens via rms_norm"
        )


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_padding_nan_does_not_leak_rms_norm_static_fp8_quant(
    default_vllm_config, device, hidden_size, add_residual
):
    """NaN in padding must not leak into real tokens via fused norm+FP8 quant."""
    from tests.kernels.quant_utils import FP8_DTYPE

    num_real = 4
    num_padded = 8
    dtype = torch.float16

    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    weight.normal_(mean=1.0, std=0.1)
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    x = torch.randn(num_padded, hidden_size, dtype=dtype, device=device)
    x[num_real:] = float("nan")
    out = torch.empty(num_padded, hidden_size, dtype=FP8_DTYPE, device=device)

    if add_residual:
        residual = torch.randn(
            num_padded, hidden_size, dtype=dtype, device=device
        )
        residual[num_real:] = float("nan")
        torch.ops._C.fused_add_rms_norm_static_fp8_quant(
            out, x, residual, weight, scale, 1e-6
        )
        real_out = out[:num_real].to(torch.float32)
        assert torch.isfinite(real_out).all(), (
            "NaN leaked via fused_add_rms_norm_static_fp8_quant"
        )
        assert torch.isfinite(residual[:num_real]).all(), (
            "NaN leaked into residual via fused_add_rms_norm_static_fp8_quant"
        )
    else:
        torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale, 1e-6)
        real_out = out[:num_real].to(torch.float32)
        assert torch.isfinite(real_out).all(), (
            "NaN leaked via rms_norm_static_fp8_quant"
        )


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@torch.inference_mode()
def test_padding_nan_does_not_leak_rms_norm_dynamic_quant(
    default_vllm_config, device, hidden_size
):
    """NaN in padding must not leak via dynamic per-token quant norm."""
    from tests.kernels.quant_utils import FP8_DTYPE
    from vllm import _custom_ops as ops

    num_real = 4
    num_padded = 8
    dtype = torch.float16

    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(num_padded, hidden_size, dtype=dtype, device=device)
    x[num_real:] = float("nan")

    out, scales = ops.rms_norm_dynamic_per_token_quant(
        x, weight, 1e-6, FP8_DTYPE
    )
    real_out = out[:num_real].to(torch.float32)
    assert torch.isfinite(real_out).all(), (
        "NaN leaked via rms_norm_dynamic_per_token_quant"
    )


@pytest.mark.parametrize("num_real", [1, 4])
@pytest.mark.parametrize("num_padded", [8, 16])
@torch.inference_mode()
def test_reshape_and_cache_skips_neg1_slots(
    default_vllm_config, device, num_real, num_padded
):
    """reshape_and_cache must not write to cache for slot_mapping=-1 (padding)."""
    from vllm import _custom_ops as ops

    num_heads = 4
    head_size = 64
    block_size = 16
    num_blocks = 4
    dtype = torch.float16
    x_val = head_size  # for reshape_and_cache, x = vector width

    key = torch.randn(
        num_padded, num_heads, head_size, dtype=dtype, device=device
    )
    value = torch.randn(
        num_padded, num_heads, head_size, dtype=dtype, device=device
    )
    # Padding tokens have NaN
    key[num_real:] = float("nan")
    value[num_real:] = float("nan")

    key_cache = torch.zeros(
        num_blocks, num_heads, head_size // x_val, block_size, x_val,
        dtype=dtype, device=device,
    )
    value_cache = torch.zeros(
        num_blocks, num_heads, head_size, block_size,
        dtype=dtype, device=device,
    )

    # Real tokens get valid slots, padding gets -1
    slot_mapping = torch.full(
        (num_padded,), -1, dtype=torch.int64, device=device
    )
    for i in range(num_real):
        slot_mapping[i] = i  # block 0, offsets 0..num_real-1

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ops.reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping, "auto",
        k_scale, v_scale,
    )

    # Cache should have no NaN — padding tokens were skipped
    assert torch.isfinite(key_cache).all(), (
        "NaN written to key_cache from padding token via reshape_and_cache"
    )
    assert torch.isfinite(value_cache).all(), (
        "NaN written to value_cache from padding token via reshape_and_cache"
    )


@pytest.mark.parametrize("num_real", [1, 4])
@pytest.mark.parametrize("num_padded", [8, 16])
@torch.inference_mode()
def test_concat_and_cache_mla_skips_padding(
    default_vllm_config, device, num_real, num_padded
):
    """concat_and_cache_mla only processes slot_mapping.size(0) tokens.

    When slot_mapping is shorter than kv_c/k_pe (V1 padding), the kernel
    must not touch padding tokens at all.
    """
    from vllm import _custom_ops as ops

    kv_lora_rank = 512
    pe_dim = 64
    block_size = 16
    num_blocks = 4
    dtype = torch.float16
    entry_size = kv_lora_rank + pe_dim

    kv_c = torch.randn(
        num_padded, kv_lora_rank, dtype=dtype, device=device
    )
    k_pe = torch.randn(num_padded, pe_dim, dtype=dtype, device=device)
    # Padding tokens have NaN
    kv_c[num_real:] = float("nan")
    k_pe[num_real:] = float("nan")

    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=dtype, device=device
    )

    # Only provide slot_mapping for real tokens (V1 behavior)
    slot_mapping = torch.arange(
        num_real, dtype=torch.int64, device=device
    )

    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, "auto", scale
    )

    # Only the first num_real slots should have data, and no NaN anywhere
    assert torch.isfinite(kv_cache).all(), (
        "NaN written to KV cache from padding via concat_and_cache_mla"
    )


@pytest.mark.parametrize("num_real", [1, 4])
@torch.inference_mode()
def test_reshape_and_cache_flash_skips_neg1_slots(
    default_vllm_config, device, num_real
):
    """reshape_and_cache_flash must skip slot_mapping=-1 entries."""
    from vllm import _custom_ops as ops

    num_padded = 8
    num_heads = 4
    head_size = 64
    block_size = 16
    num_blocks = 4
    dtype = torch.float16

    key = torch.randn(
        num_padded, num_heads, head_size, dtype=dtype, device=device
    )
    value = torch.randn(
        num_padded, num_heads, head_size, dtype=dtype, device=device
    )
    key[num_real:] = float("nan")
    value[num_real:] = float("nan")

    # Flash cache layout: [num_blocks, 2, block_size, num_heads, head_size]
    # (NHD layout)
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size,
        dtype=dtype, device=device,
    )
    value_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size,
        dtype=dtype, device=device,
    )

    slot_mapping = torch.full(
        (num_padded,), -1, dtype=torch.int64, device=device
    )
    for i in range(num_real):
        slot_mapping[i] = i

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto",
        k_scale, v_scale,
    )

    assert torch.isfinite(key_cache).all(), (
        "NaN written to key_cache from padding via reshape_and_cache_flash"
    )
    assert torch.isfinite(value_cache).all(), (
        "NaN written to value_cache from padding via reshape_and_cache_flash"
    )


@torch.inference_mode()
def test_nan_detector_class(default_vllm_config, device):
    """Test the NaNDetector singleton lifecycle."""
    detector = NaNDetector.get()

    # Register layers.
    idx0 = detector.register("layer_0")
    idx1 = detector.register("layer_1")
    assert idx0 == 0
    assert idx1 == 1

    # Finalize.
    max_tokens = 8
    detector.finalize(torch.device(device), max_tokens)
    assert detector.nan_flags is not None
    assert detector.nan_flags.shape == (2, max_tokens)
    assert detector.max_num_tokens == max_tokens

    # Clear + check with no NaN — should log nothing.
    detector.clear()
    detector.check(4)  # 4 real tokens

    # Manually set a flag and check.
    detector.nan_flags[0, 2] = 1
    detector.check(4)  # Should log ERROR for layer_0, token 2

    # Set a flag in padding region.
    detector.clear()
    detector.nan_flags[1, 6] = 1
    detector.check(4)  # Should log WARNING for layer_1 (padding)
