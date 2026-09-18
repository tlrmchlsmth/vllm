# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.uneven import (
    ExpertTransportLayout,
    UnevenExpertPrepareAndFinalize,
)


@pytest.mark.parametrize(
    "backend,constructor",
    [
        ("deepep_high_throughput", "DeepEPHTPrepareAndFinalize"),
        ("deepep_low_latency", "DeepEPLLPrepareAndFinalize"),
        ("deepep_v2", "DeepEPV2PrepareAndFinalize"),
        ("mori_high_throughput", "MoriPrepareAndFinalize"),
        ("mori_low_latency", "MoriPrepareAndFinalize"),
        ("nixl_ep", "NixlEPPrepareAndFinalize"),
        ("flashinfer_nvlink_two_sided", "FlashInferNVLinkTwoSidedPrepareAndFinalize"),
        ("flashinfer_all2allv", "FlashInferNVLinkTwoSidedPrepareAndFinalize"),
    ],
)
@pytest.mark.parametrize("num_experts", [10, 12])
def test_backend_factory_uses_uniform_transport_capacity(
    backend, constructor, num_experts, monkeypatch
):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe import all2all_utils
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        make_moe_prepare_and_finalize_no_dp_ep,
    )

    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False
    )
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        dp_size=3,
        ep_size=3,
        dp_rank=2,
        ep_rank=2,
        use_ep=True,
        all2all_backend=backend,
    )
    moe = FusedMoEConfig(
        num_experts=num_experts,
        num_logical_experts=num_experts,
        num_local_experts=num_experts // 3,
        experts_per_token=2,
        hidden_dim=2048,
        intermediate_size=128,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Unspecified,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    inner = make_moe_prepare_and_finalize_no_dp_ep(False)
    recorded = {}

    def get_handle(args):
        recorded.update(args)
        return object()

    def make_backend(*args, **kwargs):
        recorded.update(kwargs)
        return inner

    monkeypatch.setattr(all2all_utils, constructor, make_backend, raising=False)
    monkeypatch.setattr(
        all2all_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True)),
    )
    manager = SimpleNamespace(
        world_size=3, dp_world_size=3, rank=2, max_num_ep_ranks=3, get_handle=get_handle
    )
    adapter = all2all_utils.maybe_make_prepare_finalize(
        moe, FusedMoEQuantConfig.make(), all2all_manager=manager
    )
    if num_experts == 10:
        assert isinstance(adapter, UnevenExpertPrepareAndFinalize)
        assert adapter.layout.num_slots == 12
        assert adapter.num_local_experts == 3
    else:
        assert adapter is inner
    for field in ("num_experts", "num_global_experts", "expert_capacity"):
        if field in recorded:
            assert recorded[field] == 12
    if "num_local_experts" in recorded:
        assert recorded["num_local_experts"] == 4
    if "rank_expert_offset" in recorded:
        assert recorded["rank_expert_offset"] == 8


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("async_mode", ["sync", "receiver", "hook"])
def test_uneven_dispatch_combine_preserves_rows_and_async_order(batched, async_mode):
    """Padding is transport-only, and receive work stays behind the backend hook."""
    layout = ExpertTransportLayout(10, 3)
    ids = torch.tensor([[7], [8], [9], [-1]])
    transport_ids = torch.tensor([[8], [9], [10], [-1]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    counts = torch.tensor([2, 2, 2, 0], dtype=torch.int32)
    x = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4)
    scales = torch.ones(4, 2, 1)
    events: list[str] = []

    def prepare(a1, topk_weights, topk_ids, num_experts, *args):
        torch.testing.assert_close(topk_ids, transport_ids)
        assert num_experts == 12

        def receive():
            if async_mode == "hook":
                assert events[-1] == "dispatch_hook"
            events.append("receive")
            return (
                x if batched else x.flatten(0, 1),
                scales if batched else None,
                ExpertTokensMetadata(counts, counts),
                None if batched else transport_ids,
                None if batched else weights,
            )

        if async_mode == "sync":
            return receive()
        if async_mode == "hook":
            return lambda: events.append("dispatch_hook"), receive
        return receive

    def finalize(output, expert_output, topk_weights, topk_ids, *args):
        torch.testing.assert_close(topk_ids, transport_ids)
        if batched:
            assert expert_output.shape == x.shape
            assert (expert_output[3] == 0).all()

        def finish():
            if async_mode == "hook":
                assert events[-1] == "combine_hook"
            output.copy_(expert_output.sum(dim=0))

        if async_mode == "sync":
            return finish()
        if async_mode == "hook":
            return lambda: events.append("combine_hook"), finish
        return finish

    inner = SimpleNamespace(
        activation_format=(
            FusedMoEActivationFormat.BatchedExperts
            if batched
            else FusedMoEActivationFormat.Standard
        ),
        prepare=prepare,
        prepare_async=prepare,
        finalize=finalize,
        finalize_async=finalize,
    )
    adapter = UnevenExpertPrepareAndFinalize(inner, layout, 3, ep_rank=2)
    prepare_fn = adapter.prepare if async_mode == "sync" else adapter.prepare_async
    result = prepare_fn(x, weights, ids, 10, None, False, None)
    if async_mode != "sync":
        assert not events
        if isinstance(result, tuple):
            hook, receive = result
            hook()
            result = receive()
        else:
            result = result()
    prepared, prepared_scales, metadata, prepared_ids, _ = result
    torch.testing.assert_close(metadata.expert_num_tokens, counts[:3])
    torch.testing.assert_close(metadata.expert_num_tokens_cpu, counts[:3])
    if batched:
        torch.testing.assert_close(prepared, x[:3])
        torch.testing.assert_close(prepared_scales, scales[:3])
        assert prepared_ids is None
    else:
        torch.testing.assert_close(prepared_ids, ids)

    output = torch.full_like(prepared.sum(dim=0), -123)
    finalize_fn = adapter.finalize if async_mode == "sync" else adapter.finalize_async
    result = finalize_fn(output, prepared, weights, ids, False, None)
    if async_mode != "sync":
        assert (output == -123).all()
        if isinstance(result, tuple):
            hook, finish = result
            hook()
            finish()
        else:
            result()
    torch.testing.assert_close(output, prepared.sum(dim=0))


def test_ht_remote_sentinel_does_not_become_a_local_expert():
    """HT's last-slot sentinel can land in padding on the final rank."""
    layout = ExpertTransportLayout(10, 3)
    inner = SimpleNamespace(activation_format=FusedMoEActivationFormat.Standard)
    adapter = UnevenExpertPrepareAndFinalize(inner, layout, 4, ep_rank=0)
    result = adapter._restore(
        (torch.empty(1, 2), None, None, torch.tensor([[3, 11, -1]]), None)
    )
    torch.testing.assert_close(result[3], torch.tensor([[3, 9, -1]]))
