# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that NVFP4 swizzled scale padding rows are zeroed.

The scaled_fp4_quant kernel only writes scale values for the actual m rows,
but the output tensor is padded to round_up(m, 128) rows. Previously the
padding used torch.empty, leaving uninitialized memory that the CUTLASS
mm_fp4 kernel reads in its 128-row tiles, potentially contaminating real
rows with NaN/garbage values.
"""

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.has_device_capability(100),
                    reason="NVFP4 requires SM100+")
@pytest.mark.parametrize("m", [1, 13, 65, 127, 129, 255])
def test_fp4_swizzled_scale_padding_is_zeroed(m: int):
    """Verify that scale padding rows (m..round_up(m,128)) are zero."""
    import vllm._custom_ops as ops

    n = 256  # must be multiple of 16
    input_tensor = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    _output, output_scale = ops.scaled_fp4_quant(
        input_tensor, global_scale, is_sf_swizzled_layout=True
    )

    # output_scale is viewed as float8_e4m3fn, shape (rounded_m, rounded_n//4)
    # Check that rows beyond m are all zero
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)

    if rounded_m > m:
        # View as raw bytes to check for zeros
        scale_raw = output_scale.view(torch.uint8)
        # scale shape is (rounded_m, cols) where cols = round_up(n//16, 4) // 4
        # viewed as uint8, each row has cols * element_size bytes
        num_cols = scale_raw.shape[1] if scale_raw.ndim > 1 else scale_raw.numel()
        scale_2d = scale_raw.reshape(rounded_m, -1)
        padding_rows = scale_2d[m:]
        assert torch.all(padding_rows == 0), (
            f"Scale padding rows {m}..{rounded_m} contain non-zero values. "
            f"Non-zero count: {(padding_rows != 0).sum().item()}"
        )
