# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import OptimizationLevel
from vllm.engine.arg_utils import EngineArgs
from vllm.sizing.activation import estimate_activation_range_from_config
from vllm.sizing.report import SizingReport, Strategy, StrategyReport
from vllm.sizing.strategy import default_strategies, parse_strategy_spec
from vllm.sizing.weight_loader import estimate_weights_from_checkpoint
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity, get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)


def estimate_model_size(
    engine_args: EngineArgs,
    *,
    gpu_memory_bytes: int,
    gpu_type: str | None,
    max_gpus: int,
    strategy_spec: str | None = None,
) -> SizingReport:
    base_args = copy.deepcopy(engine_args)
    base_args.skip_tokenizer_init = True
    base_args.enforce_eager = True
    base_args.optimization_level = OptimizationLevel.O0
    if hasattr(base_args, "compilation_config"):
        base_args.compilation_config.mode = CompilationMode.NONE
    base_config = base_args.create_engine_config()
    strategies = (
        parse_strategy_spec(strategy_spec)
        if strategy_spec is not None
        else default_strategies(max_gpus, is_moe=base_config.model_config.is_moe)
    )

    rows: list[StrategyReport] = []
    notes: list[str] = []
    if base_config.model_config.is_moe:
        notes.append(
            "MoE parallelism: attention uses TP and is replicated over DP; "
            "experts use EP=DP*TP when ep>1, otherwise FFN TP=DP*TP"
        )
    for strategy in strategies:
        try:
            vllm_config = _config_for_strategy(base_config, strategy)
            row = _estimate_strategy(
                vllm_config, gpu_memory_bytes, strategy, report_notes=notes
            )
        except Exception as exc:
            row = StrategyReport(
                strategy=strategy,
                weight_bytes=0,
                activation_min_bytes=0,
                activation_max_bytes=0,
                kv_cache_bytes=0,
                kv_cache_tokens=0,
                max_concurrency=0.0,
                total_min_bytes=0,
                total_max_bytes=0,
                fits=False,
                notes=[f"unsupported strategy: {type(exc).__name__}: {exc}"],
            )
        rows.append(row)
        gc.collect()

    return SizingReport(
        model=base_config.model_config.model,
        gpu_type=gpu_type,
        gpu_memory_bytes=gpu_memory_bytes,
        rows=rows,
        notes=notes,
    )


def _args_for_strategy(engine_args: EngineArgs, strategy: Strategy) -> EngineArgs:
    args = copy.deepcopy(engine_args)
    args.data_parallel_size = strategy.dp
    args.tensor_parallel_size = strategy.tp
    args.pipeline_parallel_size = strategy.pp
    args.decode_context_parallel_size = strategy.dcp
    args.enable_expert_parallel = strategy.ep > 1
    args.enable_ep_weight_filter = strategy.ep > 1
    args.enforce_eager = True
    args.optimization_level = OptimizationLevel.O0
    if hasattr(args, "compilation_config"):
        args.compilation_config.mode = CompilationMode.NONE
    return args


def _config_for_strategy(
    base_config: VllmConfig,
    strategy: Strategy,
) -> VllmConfig:
    if strategy.ep > 1 and strategy.ep != strategy.dp * strategy.tp:
        raise ValueError(
            "vLLM derives expert parallel size from dp * tp; got "
            f"ep={strategy.ep} for dp={strategy.dp}, tp={strategy.tp}"
        )
    config = copy.deepcopy(base_config)
    parallel_config = config.parallel_config
    parallel_config.data_parallel_size = strategy.dp
    parallel_config.data_parallel_size_local = strategy.dp
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_rank_local = 0
    parallel_config.tensor_parallel_size = strategy.tp
    parallel_config.pipeline_parallel_size = strategy.pp
    parallel_config.decode_context_parallel_size = strategy.dcp
    parallel_config.enable_expert_parallel = strategy.ep > 1
    parallel_config.enable_ep_weight_filter = strategy.ep > 1
    parallel_config.world_size = strategy.tp * strategy.pp
    config.compilation_config.mode = CompilationMode.NONE
    config.optimization_level = OptimizationLevel.O0
    return config


def _estimate_strategy(
    vllm_config: VllmConfig,
    gpu_memory_bytes: int,
    strategy: Strategy,
    *,
    report_notes: list[str],
) -> StrategyReport:
    notes: list[str] = []
    weight_result = estimate_weights_from_checkpoint(
        vllm_config, strategy, include_records=False
    )
    _extend_unique(report_notes, weight_result.notes)

    activation_estimate = estimate_activation_range_from_config(vllm_config)
    _extend_unique(report_notes, [activation_estimate.note])

    kv_cache_bytes = 0
    kv_cache_tokens = 0
    max_concurrency = 0.0
    available_for_kv = max(
        gpu_memory_bytes
        - weight_result.total_bytes
        - activation_estimate.max_bytes,
        0,
    )
    try:
        kv_specs = _collect_kv_cache_specs(vllm_config)
        kv_configs = get_kv_cache_configs(vllm_config, [kv_specs], [available_for_kv])
        kv_config = kv_configs[0]
        kv_cache_bytes = sum(tensor.size for tensor in kv_config.kv_cache_tensors)
        if kv_config.kv_cache_groups:
            kv_cache_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, kv_config
            )
    except Exception as exc:
        notes.append(f"kv cache sizing unavailable: {type(exc).__name__}: {exc}")

    total_min_bytes = (
        weight_result.total_bytes + activation_estimate.min_bytes + kv_cache_bytes
    )
    total_max_bytes = (
        weight_result.total_bytes + activation_estimate.max_bytes + kv_cache_bytes
    )
    return StrategyReport(
        strategy=strategy,
        weight_bytes=weight_result.total_bytes,
        activation_min_bytes=activation_estimate.min_bytes,
        activation_max_bytes=activation_estimate.max_bytes,
        kv_cache_bytes=kv_cache_bytes,
        kv_cache_tokens=kv_cache_tokens,
        max_concurrency=max_concurrency,
        total_min_bytes=total_min_bytes,
        total_max_bytes=total_max_bytes,
        fits=total_max_bytes <= gpu_memory_bytes,
        notes=notes,
    )


def _collect_kv_cache_specs(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    if model_config.use_mla:
        raise NotImplementedError("config-only KV sizing does not support MLA yet")

    kv_cache_dtype = cache_config.cache_dtype
    kv_dtype = model_config.dtype if kv_cache_dtype == "auto" else _kv_dtype(
        kv_cache_dtype
    )
    spec = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_dtype,
        kv_quant_mode=get_kv_quant_mode(kv_cache_dtype),
        sliding_window=cache_config.sliding_window,
    )
    return {
        f"layers.{idx}.self_attn": spec
        for idx in range(model_config.get_num_layers(parallel_config))
    }


def _kv_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype == "float16":
        return torch.float16
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype == "fp8_e5m2":
        return getattr(torch, "float8_e5m2", torch.uint8)
    if kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "nvfp4":
        return getattr(torch, "float8_e4m3fn", torch.uint8)
    if kv_cache_dtype == "int8_per_token_head":
        return torch.int8
    if kv_cache_dtype == "int4_per_token_head":
        return torch.uint8
    return torch.float16


def _extend_unique(destination: list[str], values: list[str]) -> None:
    for value in values:
        if value not in destination:
            destination.append(value)
