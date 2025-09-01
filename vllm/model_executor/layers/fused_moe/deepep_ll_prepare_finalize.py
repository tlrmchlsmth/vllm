# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import deep_ep
import torch

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

    Uses vLLM's TP group collectives:
      - tp_group.all_gather(tensor) -> list[tensor per rank]
    Steps:
      1) All-gather compact fingerprints. If quick_only=True, decide from these.
      2) Confirm shapes match across ranks (early exit if not).
      3) All-gather full flattened tensors and compare every pair with allclose.
      4) Build a consensus boolean via a final all_gather of a 1-element flag.
    """
    device = x.device

    # ---- (1) Fingerprints via tp_group.all_gather ----
    xf = x.float()
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
    fp_list = tp_group.all_gather(fp_local)
    fps = torch.stack(fp_list, dim=0)  # [tp, F]

    same_fp = torch.all(torch.isclose(fps, fps[0], rtol=rtol, atol=atol))
    if quick_only:
        return bool(same_fp), {"mode": "fingerprint", "fingerprints": fps}

    # ---- (2) Shapes agreement (early out) ----
    local_shape = tuple(x.shape)
    shp_info_local = torch.tensor(
        [len(local_shape)] + list(local_shape)[:max_ndim_for_shape],
        dtype=torch.int64,
        device=device,
    )
    shp_len = 1 + max_ndim_for_shape
    if shp_info_local.numel() < shp_len:
        pad = torch.full((shp_len - shp_info_local.numel(), ),
                         -1,
                         dtype=torch.int64,
                         device=device)
        shp_info_local = torch.cat([shp_info_local, pad])

    shp_list = tp_group.all_gather(shp_info_local)
    gathered_shapes = []
    for t in shp_list:
        num_dim = int(t[0].item())
        dims = []
        for d in t[1:1 + num_dim]:
            if int(d.item()) == -1:
                break
            dims.append(int(d.item()))
        gathered_shapes.append(tuple(dims))

    if len(set(gathered_shapes)) != 1:
        return False, {"mode": "shape_mismatch", "shapes": gathered_shapes}

    # ---- (3) Gather full tensors and compare pairwise ----
    flat = x.to(dtype=torch.float32).contiguous().view(-1)
    flat_list = tp_group.all_gather(
        flat)  # list of [numel] float32; shapes are equal so numel matches
    all_flat = torch.stack(flat_list, dim=0)  # [tp, numel]

    tp = all_flat.size(0)
    mismatch_pairs = []
    for i in range(tp):
        for j in range(i + 1, tp):
            if not torch.allclose(
                    all_flat[i], all_flat[j], rtol=rtol, atol=atol):
                mismatch_pairs.append((i, j))

    all_equal_local = (len(mismatch_pairs) == 0)

    # ---- (4) Group-wide consensus via all_gather of a flag ----
    flag = torch.tensor([1 if all_equal_local else 0],
                        device=device,
                        dtype=torch.int32)
    flags = tp_group.all_gather(flag)
    all_equal = all(int(f.item()) == 1 for f in flags)

    details = {
        "mode": "all_gather_elementwise",
        "pairwise_mismatches": mismatch_pairs,
        "fingerprint_agreed": bool(same_fp),
    }
    return bool(all_equal), details


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
