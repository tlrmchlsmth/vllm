# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the flat GPT-OSS model definition."""

from vllm.models.gpt_oss.model import (
    GptOssForCausalLM,
    GptOssModel,
    MLPBlock,
    OAIAttention,
    TransformerBlock,
)

__all__ = [
    "GptOssForCausalLM",
    "GptOssModel",
    "MLPBlock",
    "OAIAttention",
    "TransformerBlock",
]
