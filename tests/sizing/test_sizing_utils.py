# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from types import SimpleNamespace

import pytest

from vllm.entrypoints.cli.sizer import SizerSubcommand
from vllm.sizing.activation import estimate_activation_range_from_config
from vllm.sizing.gpu import GiB, parse_memory
from vllm.sizing.mapping import _extract_parallel_module_names
from vllm.sizing.report import SizingReport, Strategy, StrategyReport
from vllm.sizing.runner import estimate_model_size
from vllm.sizing.strategy import default_strategies, parse_strategy_spec
from vllm.sizing.weight_loader import _estimate_local_tensor_bytes

pytestmark = pytest.mark.skip_global_cleanup


class ColumnParallelLinear:
    pass


class ReplicatedLinear:
    pass


class FusedMoE:
    pass


class FakeAstModel:
    def __init__(self) -> None:
        self.q_a_proj = ColumnParallelLinear()
        self.router = ReplicatedLinear()
        self.experts = FusedMoE()


def test_parse_memory_units() -> None:
    assert parse_memory("1GiB") == GiB
    assert parse_memory("1.5GiB") == int(1.5 * GiB)
    assert parse_memory("80GB") == 80 * 1000**3


def test_parse_strategy_spec_expands_cartesian_product() -> None:
    strategies = parse_strategy_spec("dp=1,2 tp=1,4 dcp=1")

    assert strategies == [
        Strategy(dp=1, tp=1, dcp=1),
        Strategy(dp=1, tp=4, dcp=1),
        Strategy(dp=2, tp=1, dcp=1),
        Strategy(dp=2, tp=4, dcp=1),
    ]


def test_parse_strategy_spec_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="dp/tp/ep/pp/dcp"):
        parse_strategy_spec("tp=1 foo=2")


def test_default_strategies_add_ep_only_for_moe() -> None:
    dense = default_strategies(4, is_moe=False)
    moe = default_strategies(4, is_moe=True)

    assert all(strategy.ep == 1 for strategy in dense)
    assert any(strategy.ep > 1 for strategy in moe)
    assert all(strategy.ep == strategy.dp * strategy.tp for strategy in moe)


def test_expert_weights_are_not_sharded_by_tp_and_ep() -> None:
    tensor = SimpleNamespace(bytes=1024)
    mapping = SimpleNamespace(
        packed_modules_mapping={"gate_up_proj": []},
        stacked_param_names=(),
        expert_weight_names=frozenset({"experts.gate_up_proj"}),
    )

    local_bytes = _estimate_local_tensor_bytes(
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        tensor,
        Strategy(tp=8, ep=8),
        mapping,
    )

    assert local_bytes == 128


def test_expert_weights_use_flattened_dp_tp_when_ep_is_disabled() -> None:
    tensor = SimpleNamespace(bytes=1024)
    mapping = SimpleNamespace(
        packed_modules_mapping={},
        stacked_param_names=(),
        expert_weight_names=frozenset(),
    )

    local_bytes = _estimate_local_tensor_bytes(
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        tensor,
        Strategy(dp=2, tp=4, ep=1),
        mapping,
    )

    assert local_bytes == 128


def test_ast_mapping_extracts_parallel_module_assignments() -> None:
    mapping = _extract_parallel_module_names(FakeAstModel)

    assert "q_a_proj" in mapping.tp
    assert "router" in mapping.replicated
    assert "experts" in mapping.expert


def test_ast_tp_module_names_drive_tensor_sharding() -> None:
    tensor = SimpleNamespace(bytes=1024)
    mapping = SimpleNamespace(
        packed_modules_mapping={},
        stacked_param_names=(),
        expert_weight_names=frozenset(),
        tp_module_names=frozenset({"q_a_proj"}),
        expert_module_names=frozenset(),
    )

    local_bytes = _estimate_local_tensor_bytes(
        "model.layers.0.self_attn.q_a_proj.weight",
        tensor,
        Strategy(tp=4),
        mapping,
    )

    assert local_bytes == 256


def test_sizer_cli_defaults_max_gpus_to_8() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    SizerSubcommand().subparser_init(subparsers)

    args = parser.parse_args(["sizer", "model", "--gpu-type", "H200-141GB"])

    assert args.max_gpus == 8


def test_report_table_includes_kv_tokens() -> None:
    report = SizingReport(
        model="tiny",
        gpu_type=None,
        gpu_memory_bytes=GiB,
        rows=[
            StrategyReport(
                strategy=Strategy(dp=1, tp=2),
                weight_bytes=10,
                activation_min_bytes=20,
                activation_max_bytes=40,
                kv_cache_bytes=30,
                kv_cache_tokens=4096,
                max_concurrency=2.0,
                total_min_bytes=60,
                total_max_bytes=80,
                fits=True,
            )
        ],
    )

    table = report.to_table()

    assert "kv tokens" in table
    assert "4,096" in table
    assert "20 B - 40 B" in table


def test_report_table_uses_report_notes_for_provenance() -> None:
    report = SizingReport(
        model="tiny",
        gpu_type=None,
        gpu_memory_bytes=GiB,
        rows=[
            StrategyReport(
                strategy=Strategy(),
                weight_bytes=10,
                activation_min_bytes=20,
                activation_max_bytes=40,
                kv_cache_bytes=30,
                kv_cache_tokens=4096,
                max_concurrency=2.0,
                total_min_bytes=60,
                total_max_bytes=80,
                fits=True,
            )
        ],
        notes=["weights: safetensors metadata", "activations: config-derived range"],
    )

    table = report.to_table()

    assert "notes" not in table.splitlines()[0]
    assert table.splitlines()[2].endswith("yes")
    assert "note: weights: safetensors metadata" in table
    assert "note: activations: config-derived range" in table


def test_activation_range_uses_config_only() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype="float16",
            hf_text_config=SimpleNamespace(
                hidden_size=128,
                intermediate_size=512,
                num_attention_heads=8,
            ),
            is_moe=False,
            max_model_len=1024,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )

    estimate = estimate_activation_range_from_config(config)

    assert estimate.min_bytes > 0
    assert estimate.max_bytes > estimate.min_bytes
    assert "config-only" in estimate.note


def test_sizer_preserves_auto_dtype(monkeypatch) -> None:
    seen_dtypes: list[str] = []

    class FakeEngineArgs:
        dtype = "auto"
        skip_tokenizer_init = False

        def __deepcopy__(self, memo):
            del memo
            copy = FakeEngineArgs()
            copy.dtype = self.dtype
            copy.skip_tokenizer_init = self.skip_tokenizer_init
            return copy

        def create_engine_config(self):
            seen_dtypes.append(self.dtype)
            return SimpleNamespace(
                model_config=SimpleNamespace(model="fake", is_moe=False)
            )

    def fake_estimate_strategy(vllm_config, gpu_memory_bytes, strategy, report_notes):
        del vllm_config, gpu_memory_bytes, report_notes
        return StrategyReport(
            strategy=strategy,
            weight_bytes=0,
            activation_min_bytes=0,
            activation_max_bytes=0,
            kv_cache_bytes=0,
            kv_cache_tokens=0,
            max_concurrency=0,
            total_min_bytes=0,
            total_max_bytes=0,
            fits=True,
        )

    monkeypatch.setattr("vllm.sizing.runner._estimate_strategy", fake_estimate_strategy)

    estimate_model_size(
        FakeEngineArgs(),
        gpu_memory_bytes=GiB,
        gpu_type=None,
        max_gpus=1,
        strategy_spec="dp=1 tp=1",
    )

    assert seen_dtypes == ["auto"]
