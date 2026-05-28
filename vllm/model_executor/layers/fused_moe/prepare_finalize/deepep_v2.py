# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)

import os
import threading

_deepep_stats_lock = threading.Lock()
_deepep_stats: dict[str, list[float]] = {
    "dispatch_duration": [],
    "combine_duration": [],
}
_deepep_events: dict[str, tuple] = {}
_deepep_collector_running = False


def get_deepep_stats() -> "dict[str, list[float]] | None":
    if _deepep_events and not _deepep_collector_running:
        _start_deepep_collector()
    with _deepep_stats_lock:
        if (not _deepep_stats["dispatch_duration"]
                and not _deepep_stats["combine_duration"]):
            return None
        result = {k: list(v) for k, v in _deepep_stats.items()}
        _deepep_stats["dispatch_duration"].clear()
        _deepep_stats["combine_duration"].clear()
        return result


def _start_deepep_collector():
    global _deepep_collector_running
    if _deepep_collector_running:
        return
    _deepep_collector_running = True

    import time

    def _loop():
        while True:
            time.sleep(0.5)
            for name, (start_evt, end_evt) in _deepep_events.items():
                try:
                    if not start_evt.query() or not end_evt.query():
                        continue
                    elapsed_ms = start_evt.elapsed_time(end_evt)
                    if elapsed_ms > 0:
                        with _deepep_stats_lock:
                            _deepep_stats[name].append(elapsed_ms / 1000.0)
                except (RuntimeError, ValueError):
                    pass

    t = threading.Thread(target=_loop, daemon=True,
                         name="deepep-stats-collector")
    t.start()


class DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP v2 ElasticBuffer (unified API).

    Supports two modes controlled by the `use_cudagraph` constructor arg:

    **Decode mode (use_cudagraph=True):**
      - do_expand=False, do_cpu_sync=False
      - Tokens returned in original order with recv_topk_idx (global IDs)
      - Worst-case tensor allocation; padding rows zeroed via
        handle.psum_num_recv_tokens_per_scaleup_rank
      - Fully cudagraph-capturable
      - Expert kernel sorts internally (expert_tokens_meta=None)

    **Prefill mode (use_cudagraph=False):**
      - do_expand=True, do_cpu_sync=True
      - Per-expert-contiguous layout; exact memory allocation
      - Saves GPU memory (no worst-case allocation)
      - Not cudagraph-capturable (CPU polling), but prefill doesn't
        use cudagraphs anyway
      - Provides expert_tokens_meta for efficient batched expert kernels

    Both modes use async_with_compute_stream=False (synchronous from
    caller's perspective). The ElasticBuffer handles comm internally.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer: deep_ep.ElasticBuffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        use_cudagraph: bool = False,
        dedup_topk: bool = False,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch
        self.use_cudagraph = use_cudagraph
        self.dedup_topk = dedup_topk
        self._dedup_neg_one: torch.Tensor | None = None

        # DBO microbatching: one handle slot per micro-batch.
        self.handles: list[deep_ep.EPHandle | None] = [None, None]

        self._timing = os.environ.get("DEEPEP_METRICS_ENABLED") == "1"
        if self._timing:
            self._dispatch_start = torch.cuda.Event(enable_timing=True)
            self._dispatch_end = torch.cuda.Event(enable_timing=True)
            self._combine_start = torch.cuda.Event(enable_timing=True)
            self._combine_end = torch.cuda.Event(enable_timing=True)
            _deepep_events["dispatch_duration"] = (
                self._dispatch_start, self._dispatch_end)
            _deepep_events["combine_duration"] = (
                self._combine_start, self._combine_end)

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _dedup_topk(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge duplicate expert IDs per token (from hash-based MoE routing).

        For each token, if two topk slots select the same expert, sum their
        weights into the first occurrence and set the duplicate to -1 / 0.
        All ops are unconditional torch.where so this is CUDA-graph safe.
        """
        topk_ids = topk_ids.clone()
        topk_weights = topk_weights.clone()
        if self._dedup_neg_one is None:
            self._dedup_neg_one = torch.tensor(
                -1, dtype=topk_ids.dtype, device=topk_ids.device)
        neg_one = self._dedup_neg_one
        K = topk_ids.size(1)
        for j in range(1, K):
            for i in range(j):
                dup = (topk_ids[:, j] == topk_ids[:, i]) & (topk_ids[:, j] >= 0)
                topk_weights[:, i] = torch.where(
                    dup, topk_weights[:, i] + topk_weights[:, j],
                    topk_weights[:, i])
                topk_weights[:, j] = torch.where(
                    dup, 0.0, topk_weights[:, j])
                topk_ids[:, j] = torch.where(
                    dup, neg_one, topk_ids[:, j])
        return topk_ids, topk_weights

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        has_scales = token_scales is not None

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        # Decode: do_expand=False + do_cpu_sync=False (cudagraph-safe)
        # Prefill: do_expand=True + do_cpu_sync=True (memory-efficient)
        do_expand = not self.use_cudagraph
        do_cpu_sync = not self.use_cudagraph

        if self._timing:
            self._dispatch_start.record()

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            event,
        ) = self.buffer.dispatch(
            x=token_data,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            num_experts=num_experts,
            do_expand=do_expand,
            do_cpu_sync=do_cpu_sync,
            async_with_compute_stream=False,
        )

        if self._timing:
            self._dispatch_end.record()

        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        return lambda: self._receiver(
            event,
            has_scales,
            recv_x,
            recv_topk_idx,
            num_experts,
            handle.num_recv_tokens_per_expert_list,
            recv_topk_weights,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        recv_x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        recv_topk_idx: torch.Tensor | None,
        num_experts: int,
        recv_expert_num_tokens: list[int],
        recv_topk_weights: torch.Tensor | None,
        psum_recv_per_rank: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        if recv_topk_idx is None:
            # do_expand=True (prefill mode): build topk_ids from
            # per-expert token counts.
            total_tokens = sum(recv_expert_num_tokens)
            if total_tokens > 0:
                recv_topk_idx = torch.empty(
                    total_tokens,
                    dtype=torch.int64,
                    device=expert_x.device,
                )
                offset = 0
                for i, count in enumerate(recv_expert_num_tokens):
                    if count > 0:
                        recv_topk_idx[offset:offset + count].fill_(
                            i + self.rank_expert_offset)
                        offset += count
            else:
                recv_topk_idx = torch.empty(
                    0, dtype=torch.int64, device=expert_x.device,
                )
            recv_topk_idx = recv_topk_idx.unsqueeze(1)
        else:
            # do_expand=False (decode/cudagraph mode): recv_topk_idx has
            # LOCAL expert IDs (-1 for non-local and padding rows).
            # Convert valid local IDs to global. Rows with -1 are
            # skipped by expert kernels (TrtLLM tile-level skipping,
            # DeepGemm is_computation_valid), so no need to zero
            # hidden states, scales, or weights for padding rows.
            valid_mask = recv_topk_idx >= 0
            recv_topk_idx = torch.where(
                valid_mask,
                recv_topk_idx + self.rank_expert_offset,
                recv_topk_idx,
            )

        # Reshape recv_topk_weights to match recv_topk_idx shape [N, 1]
        if recv_topk_weights is not None and recv_topk_weights.ndim == 1:
            recv_topk_weights = recv_topk_weights.unsqueeze(1)

        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            recv_expert_num_tokens, device=expert_x.device,
        )

        if not quant_config.is_block_quantized and not defer_input_quant:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

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
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        if self.dedup_topk:
            topk_ids, topk_weights = self._dedup_topk(topk_ids, topk_weights)

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

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
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        assert handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        if fused_expert_output.dtype != torch.bfloat16:
            raise ValueError(
                f"DeepEP v2 combine requires bfloat16 input, "
                f"got {fused_expert_output.dtype}"
            )

        if self._timing:
            self._combine_start.record()

        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=handle,
            topk_weights=None,
            async_with_compute_stream=False,
        )

        if self._timing:
            self._combine_end.record()

        output.copy_(combined_x, non_blocking=True)
        return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
        return lambda: None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
