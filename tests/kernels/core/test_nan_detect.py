# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NaN/Inf detection AND clearing in RMSNorm kernels.

The RMSNorm kernel detects NaN/Inf via the variance reduction (zero cost)
and writes a per-token int8 flag. When nan_flags is provided and NaN is
detected, the kernel also zeroes the output (and residual in the fused
path) to prevent propagation.
"""

import pytest
import torch

from vllm.model_executor.layers.nan_detector import NaNDetector


@pytest.fixture(autouse=True)
def reset_nan_detector():
    NaNDetector.reset()
    yield
    NaNDetector.reset()


@pytest.fixture
def device():
    return "cuda:0"


# ---- Detection tests ----


@pytest.mark.parametrize("hidden_size", [64, 256])
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@torch.inference_mode()
def test_nan_detected_and_flagged(
    default_vllm_config, device, hidden_size, num_tokens
):
    """NaN in input should set the flag at the correct token position."""
    from vllm import _custom_ops as ops

    max_num_tokens = 32
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device)
    out = torch.empty_like(x)

    # Clean input — no flags.
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags.sum().item() == 0, "False positive on clean input"

    # Inject NaN at token 0.
    nan_flags.zero_()
    x_nan = x.clone()
    x_nan[0, 0] = float("nan")
    ops.rms_norm(out, x_nan, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags[0, 0].item() == 1, "NaN not flagged at token 0"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_inf_detected_and_flagged(default_vllm_config, device, hidden_size):
    """Inf in input should set the flag."""
    from vllm import _custom_ops as ops

    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(4, hidden_size, dtype=torch.float16, device=device)
    x[2, 0] = float("inf")
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    assert nan_flags[0, 2].item() == 1, "Inf not flagged at token 2"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_fused_nan_in_input_flagged(default_vllm_config, device, hidden_size):
    """NaN in input (not residual) detected in fused_add_rms_norm."""
    from vllm import _custom_ops as ops

    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(4, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn_like(x)
    x[3, 0] = float("nan")
    ops.fused_add_rms_norm(
        x, residual, weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 3].item() == 1, "NaN in input not flagged"


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_fused_nan_in_residual_flagged(
    default_vllm_config, device, hidden_size
):
    """NaN in residual detected in fused_add_rms_norm."""
    from vllm import _custom_ops as ops

    max_num_tokens = 8
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)

    x = torch.randn(4, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn_like(x)
    residual[1, 0] = float("nan")
    ops.fused_add_rms_norm(
        x, residual, weight, 1e-6, nan_flags, 0, max_num_tokens
    )
    assert nan_flags[0, 1].item() == 1, "NaN in residual not flagged"


# ---- Clearing tests (output zeroed on NaN) ----


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@torch.inference_mode()
def test_rms_norm_output_zeroed_on_nan(
    default_vllm_config, device, hidden_size
):
    """When NaN is detected, the output for that token should be all zeros."""
    from vllm import _custom_ops as ops

    num_tokens = 8
    max_num_tokens = 16
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    x[3, :] = float("nan")
    out = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)

    # Token 3 output should be zeroed.
    assert (out[3] == 0).all(), "NaN token output not zeroed by rms_norm"
    # Other tokens should be finite and non-zero.
    assert torch.isfinite(out[0]).all(), "Clean token output is not finite"
    assert out[0].abs().sum() > 0, "Clean token output is unexpectedly zero"


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@torch.inference_mode()
def test_fused_output_and_residual_zeroed_on_nan(
    default_vllm_config, device, hidden_size
):
    """Fused path: both output and residual zeroed for NaN tokens."""
    from vllm import _custom_ops as ops

    num_tokens = 8
    max_num_tokens = 16
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    residual = torch.randn_like(x)
    x[2, :] = float("nan")

    ops.fused_add_rms_norm(
        x, residual, weight, 1e-6, nan_flags, 0, max_num_tokens
    )

    # Token 2: both output (x) and residual should be zeroed.
    assert (x[2] == 0).all(), (
        "NaN token output not zeroed by fused_add_rms_norm"
    )
    assert (residual[2] == 0).all(), (
        "NaN token residual not zeroed by fused_add_rms_norm"
    )
    # Other tokens should be finite.
    assert torch.isfinite(x[0]).all(), "Clean token output is not finite"
    assert torch.isfinite(residual[0]).all(), "Clean token residual not finite"


@pytest.mark.parametrize("hidden_size", [64, 256, 5120])
@torch.inference_mode()
def test_fused_nan_in_residual_both_zeroed(
    default_vllm_config, device, hidden_size
):
    """NaN in residual (not input): both output and residual zeroed."""
    from vllm import _custom_ops as ops

    num_tokens = 8
    max_num_tokens = 16
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    residual = torch.randn_like(x)
    residual[5, :] = float("nan")

    ops.fused_add_rms_norm(
        x, residual, weight, 1e-6, nan_flags, 0, max_num_tokens
    )

    assert nan_flags[0, 5].item() == 1, "NaN in residual not flagged"
    assert (x[5] == 0).all(), "Output not zeroed when residual has NaN"
    assert (residual[5] == 0).all(), "Residual not zeroed when it had NaN"


# ---- Selective clearing (only NaN tokens affected) ----


@pytest.mark.parametrize("hidden_size", [64, 256])
@torch.inference_mode()
def test_only_nan_tokens_affected(default_vllm_config, device, hidden_size):
    """Clean tokens must produce normal output even when other tokens have NaN."""
    from vllm import _custom_ops as ops

    num_tokens = 8
    max_num_tokens = 16
    nan_flags = torch.zeros(1, max_num_tokens, dtype=torch.int8, device=device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    weight.normal_(mean=1.0, std=0.1)

    x = torch.randn(
        num_tokens, hidden_size, dtype=torch.float16, device=device
    )
    x_ref = x.clone()

    # Poison tokens 1 and 5.
    x[1, :] = float("nan")
    x[5, :] = float("nan")

    out = torch.empty_like(x)
    out_ref = torch.empty_like(x)
    ops.rms_norm(out, x, weight, 1e-6, nan_flags, 0, max_num_tokens)
    ops.rms_norm(out_ref, x_ref, weight, 1e-6)  # no nan_flags

    # Poisoned tokens should be zeroed.
    assert (out[1] == 0).all()
    assert (out[5] == 0).all()
    assert nan_flags[0, 1].item() == 1
    assert nan_flags[0, 5].item() == 1

    # Clean tokens should match the reference exactly.
    for i in [0, 2, 3, 4, 6, 7]:
        assert nan_flags[0, i].item() == 0, f"False positive at token {i}"
        torch.testing.assert_close(out[i], out_ref[i], atol=0, rtol=0)


# ---- No detection when disabled ----


@torch.inference_mode()
def test_no_detection_when_disabled(default_vllm_config, device):
    """When nan_flags is None, NaN propagates normally (no crash)."""
    from vllm import _custom_ops as ops

    hidden_size = 64
    weight = torch.ones(hidden_size, dtype=torch.float16, device=device)
    x = torch.randn(4, hidden_size, dtype=torch.float16, device=device)
    x[0, 0] = float("nan")
    out = torch.empty_like(x)

    ops.rms_norm(out, x, weight, 1e-6)
    assert not torch.isfinite(out[0]).all(), (
        "NaN should propagate when detection is disabled"
    )


# ---- NaNDetector class ----


@torch.inference_mode()
def test_nan_detector_class(default_vllm_config, device):
    detector = NaNDetector.get()

    idx0 = detector.register("layer_0")
    idx1 = detector.register("layer_1")
    assert idx0 == 0
    assert idx1 == 1

    detector.finalize(torch.device(device), 8)
    assert detector.nan_flags is not None
    assert detector.nan_flags.shape == (2, 8)

    detector.clear()
    detector.check(4)

    detector.nan_flags[0, 2] = 1
    detector.check(4)  # should log ERROR for layer_0, token 2
