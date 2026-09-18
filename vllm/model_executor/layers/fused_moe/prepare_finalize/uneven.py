# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


@dataclass(frozen=True)
class ExpertTransportLayout:
    """Map model experts to uniform transport blocks with unused tail slots."""

    num_experts: int
    ep_size: int
    round_robin: bool = False

    @property
    def experts_per_rank(self) -> int:
        return (self.num_experts + self.ep_size - 1) // self.ep_size

    @property
    def num_slots(self) -> int:
        return self.experts_per_rank * self.ep_size

    def to_transport(self, ids: torch.Tensor) -> torch.Tensor:
        if self.round_robin:
            owner = ids.remainder(self.ep_size)
            local = ids.div(self.ep_size, rounding_mode="floor")
        else:
            base, remainder = divmod(self.num_experts, self.ep_size)
            split = remainder * (base + 1)
            owner = torch.where(
                ids < split,
                ids.div(base + 1, rounding_mode="floor"),
                remainder + (ids - split).div(max(base, 1), rounding_mode="floor"),
            )
            start = owner * base + owner.clamp(max=remainder)
            local = ids - start
        return torch.where(
            (ids >= 0) & (ids < self.num_experts),
            owner * self.experts_per_rank + local,
            -1,
        )

    def from_transport(self, ids: torch.Tensor) -> torch.Tensor:
        owner = ids.div(self.experts_per_rank, rounding_mode="floor")
        local = ids.remainder(self.experts_per_rank)
        base, remainder = divmod(self.num_experts, self.ep_size)
        count = base + (owner < remainder).to(ids.dtype)
        if self.round_robin:
            logical = local * self.ep_size + owner
        else:
            logical = owner * base + owner.clamp(max=remainder) + local
        return torch.where(
            (ids >= 0) & (ids < self.num_slots) & (local < count), logical, -1
        )


class UnevenExpertPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Adapt uniform-block transports to the model's uneven expert ownership."""

    def __init__(
        self,
        inner: mk.FusedMoEPrepareAndFinalizeModular,
        layout: ExpertTransportLayout,
        num_local_experts: int,
        ep_rank: int,
    ):
        self.inner = inner
        self.layout = layout
        self.num_local_experts = num_local_experts
        self.ep_rank = ep_rank

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return self.inner.activation_format

    def topk_indices_dtype(self) -> torch.dtype | None:
        return self.inner.topk_indices_dtype()

    def max_num_tokens_per_rank(self) -> int | None:
        return self.inner.max_num_tokens_per_rank()

    def num_dispatchers(self) -> int:
        return self.inner.num_dispatchers()

    def output_is_reduced(self) -> bool:
        return self.inner.output_is_reduced()

    def supports_async(self) -> bool:
        return self.inner.supports_async()

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        self.inner.post_init_setup(fused_experts)

    def _restore(self, result: mk.PrepareResultType) -> mk.PrepareResultType:
        x, scales, metadata, ids, weights = result
        if ids is not None:
            logical_ids = self.layout.from_transport(ids)
            # HT substitutes a remote expert for -1. Its final transport slot
            # may be padding, so preserve that substitution in the model space.
            remote_expert = self.layout.num_experts - 1 if self.ep_rank == 0 else 0
            ids = torch.where(
                (ids >= 0) & (logical_ids < 0), remote_expert, logical_ids
            )
        if self.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            x = x[: self.num_local_experts]
            if scales is not None:
                scales = scales[: self.num_local_experts]
        if metadata is not None:
            metadata = replace(
                metadata,
                expert_num_tokens=(
                    metadata.expert_num_tokens[: self.num_local_experts]
                    if metadata.expert_num_tokens is not None
                    else None
                ),
                expert_num_tokens_cpu=(
                    metadata.expert_num_tokens_cpu[: self.num_local_experts]
                    if metadata.expert_num_tokens_cpu is not None
                    else None
                ),
            )
        return x, scales, metadata, ids, weights

    def _pad_output(self, output: torch.Tensor) -> torch.Tensor:
        if (
            self.activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and self.num_local_experts < self.layout.experts_per_rank
        ):
            padding = output.new_zeros(
                (
                    self.layout.experts_per_rank - self.num_local_experts,
                    *output.shape[1:],
                )
            )
            return torch.cat((output, padding), dim=0)
        return output

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        return self._restore(
            self.inner.prepare(
                a1,
                topk_weights,
                self.layout.to_transport(topk_ids),
                self.layout.num_slots,
                None,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )
        )

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType] | mk.ReceiverType:
        dispatch_ids = self.layout.to_transport(topk_ids)
        result = self.inner.prepare_async(
            a1,
            topk_weights,
            dispatch_ids,
            self.layout.num_slots,
            None,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        receiver = result[1] if isinstance(result, tuple) else result

        def restore():
            # Keep the mapped IDs alive until the asynchronous dispatch completes.
            _ = dispatch_ids
            return self._restore(receiver())

        return (result[0], restore) if isinstance(result, tuple) else restore

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self.inner.finalize(
            output,
            self._pad_output(fused_expert_output),
            topk_weights,
            self.layout.to_transport(topk_ids),
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
        combine_ids = self.layout.to_transport(topk_ids)
        padded_output = self._pad_output(fused_expert_output)
        result = self.inner.finalize_async(
            output,
            padded_output,
            topk_weights,
            combine_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver = result[1] if isinstance(result, tuple) else result

        def finish():
            _ = combine_ids, padded_output
            return receiver()

        return (result[0], finish) if isinstance(result, tuple) else finish
