# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for estimating decoder deployment memory footprints."""

from vllm.sizing.gpu import GPU_MEMORY_BYTES, parse_memory
from vllm.sizing.report import SizingReport


def estimate_model_size(*args, **kwargs):
    from vllm.sizing.runner import estimate_model_size as _estimate_model_size

    return _estimate_model_size(*args, **kwargs)

__all__ = [
    "GPU_MEMORY_BYTES",
    "SizingReport",
    "estimate_model_size",
    "parse_memory",
]
