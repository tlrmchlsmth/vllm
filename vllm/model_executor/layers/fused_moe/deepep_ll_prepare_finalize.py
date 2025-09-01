# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import deep_ep
import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input, normalize_batched_scales_shape)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]


def _shape_vector(t: torch.Tensor,
                  max_ndim: int = 8,
                  device=None,
                  dtype=torch.float64):
    """Encode shape into a fixed-length vector (padded with -1)."""
    shp = list(t.shape)
    vec = torch.full((max_ndim, ), -1, dtype=dtype, device=device)
    vec[:min(len(shp), max_ndim)] = torch.tensor(shp[:max_ndim],
                                                 dtype=dtype,
                                                 device=device)
    return vec


@torch.no_grad()
def tp_all_equal(
    x: torch.Tensor,
    tp_group,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    quick_only: bool = False,
    max_ndim_for_shape: int = 8,
):
    """
    Returns (all_equal: bool, details: dict).
    - First does a lightweight all_gather of fingerprints (shape + stats).
    - If fingerprints differ, returns False immediately.
    - If quick_only=False and fingerprints differ, it falls back to a full
      elementwise check by broadcasting rank-0's tensor to the group and
      comparing with torch.allclose.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"

    ws = dist.get_world_size(group=tp_group)

    # Build a compact fingerprint: [shape vector..., numel, mean, std, L1]
    xf = x.float()
    device = x.device
    shape_vec = _shape_vector(x, max_ndim_for_shape, device=device)
    stats = torch.tensor(
        [
            x.numel(),
            xf.mean().item(),
            xf.std(unbiased=False).item(),
            xf.abs().sum().item()
        ],
        dtype=torch.float64,
        device=device,
    )
    fp_local = torch.cat([shape_vec, stats])  # fixed-size
    # all_gather requires equal-sized tensors
    bufs = [torch.zeros_like(fp_local) for _ in range(ws)]
    dist.all_gather(bufs, fp_local, group=tp_group)
    fps = torch.stack(bufs, dim=0)  # [tp, F]

    # Quick decision: if all fingerprints match within tolerance, call it equal.
    same_fp = torch.all(torch.isclose(fps, fps[0], rtol=rtol, atol=atol))
    if same_fp:
        return True, {"mode": "fingerprint", "fingerprints": fps}

    if quick_only:
        return False, {"mode": "fingerprint_mismatch", "fingerprints": fps}

    # Full check: broadcast rank-0 tensor and compare elementwise.
    # Prepare a ref tensor that will be overwritten by broadcast
    ref = x.clone()
    dist.broadcast(ref, src=0, group=tp_group)
    # If shapes differ, this will fail fast:
    same_shape = (tuple(ref.shape) == tuple(x.shape))
    if not same_shape:
        return False, {
            "mode": "shape_mismatch",
            "ref_shape": tuple(ref.shape),
            "local_shape": tuple(x.shape)
        }

    equal = torch.allclose(x,
                           ref.to(dtype=x.dtype, device=x.device),
                           rtol=rtol,
                           atol=atol)
    return bool(equal), {"mode": "elementwise", "equal": bool(equal)}


def dequant_fp8(expert_x_fp8: torch.Tensor,
                expert_x_scales: torch.Tensor) -> torch.Tensor:
    """
    Return dequantized tensor in fp32
    """
    # TODO (varun) : Optimize leverage num_tokens_per_expert counts
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, DEEPEP_QUANT_BLOCK_SIZE)
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using DeepEP low-latency kernels.
    """

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 4096, 5120, 6144, 7168]

    def __init__(self,
                 buffer: deep_ep.Buffer,
                 max_tokens_per_rank: int,
                 num_dispatchers: int,
                 use_fp8_dispatch: bool = False):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handle = None
        self.num_dispatchers_ = num_dispatchers

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def _do_quant(
        self,
        x: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        a1_dtype: torch.dtype,
        quant_dtype: Union[torch.dtype, str, None],
        per_act_token_quant: bool,
        block_shape: Optional[list[int]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        block_k = block_shape[1] if block_shape is not None else None
        if self.use_fp8_dispatch:
            if block_k == DEEPEP_QUANT_BLOCK_SIZE:
                # DeepEP kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched in.
            x_fp8, x_scales = x
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, torch.Tensor)

        num_experts, max_tokens, hidden_dim = x.size()

        # TODO (varun): Optimization - Use a batched version of quant
        x = x.view((-1, hidden_dim))
        x, x_scales = moe_kernel_quantize_input(x, a1_scale, quant_dtype,
                                                per_act_token_quant,
                                                block_shape)
        x = x.view((num_experts, -1, hidden_dim))

        if quant_dtype is not None:
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[mk.ExpertTokensMetadata], Optional[torch.Tensor],
               Optional[torch.Tensor]]:

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, \
            (f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}")

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, \
            "DeepEP kernels quantize the inputs in blocks of shape 128"

        has_per_token_scales = a1_scale.numel(
        ) != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)
        assert not has_per_token_scales, (
            "low_latency kernels doesn't support dispatching per-token scales")

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        expert_x, expert_num_tokens, self.handle, event, hook = \
                self.buffer.low_latency_dispatch(a1,
                                                topk_ids,
                                                self.max_tokens_per_rank,
                                                num_experts,
                                                use_fp8=self.use_fp8_dispatch,
                                                async_finish=False,
                                                return_recv_hook=False)

        expert_x, expert_x_scale = self._do_quant(
            expert_x, a1_scale, a2_scale, a1.dtype, quant_config.quant_dtype,
            quant_config.per_act_token_quant, quant_config.block_shape)

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None)

        return (expert_x, expert_x_scale, expert_tokens_meta, None, None)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert isinstance(
            weight_and_reduce_impl, TopKWeightAndReduceDelegate
        ), ("Weight application and reduction happens in the combine kernel.")
        assert self.handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        all_equal, info = tp_all_equal(fused_expert_output,
                                       get_tp_group(),
                                       atol=1e-5,
                                       rtol=1e-4)
        print("TP activations equal?", all_equal, "| details:", info["mode"])

        # TODO (varun) : Enable zero copy mode
        _, event, hook = self.buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            combine_topk_weights,
            self.handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=False,
            out=output)
