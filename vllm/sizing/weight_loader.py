# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import torch
from safetensors import safe_open

from vllm.config import VllmConfig
from vllm.sizing.mapping import ModelWeightMapping, get_model_weight_mapping
from vllm.sizing.model_loader import estimate_weights_from_model_definition
from vllm.sizing.report import Strategy, WeightRecord
from vllm.transformers_utils.config import get_safetensors_params_metadata

_SAFETENSORS_DTYPE_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.uint8),
}

_TP_SHARDED_HINTS = (
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".o_proj.",
    ".gate_proj.",
    ".up_proj.",
    ".down_proj.",
    ".qkv_proj.",
    ".gate_up_proj.",
    ".w1.",
    ".w2.",
    ".w3.",
    ".w13.",
)
_EXPERT_HINTS = (
    ".experts.",
    ".mlp.experts.",
    ".block_sparse_moe.",
    ".moe.",
)


@dataclass
class WeightSizingResult:
    total_bytes: int
    records: list[WeightRecord]
    notes: list[str]


@dataclass(frozen=True)
class _TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def bytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * self.dtype.itemsize


def estimate_weights_from_checkpoint(
    vllm_config: VllmConfig,
    strategy: Strategy,
    *,
    include_records: bool = True,
) -> WeightSizingResult:
    notes: list[str] = []
    tensors = _load_safetensors_metadata(vllm_config, notes)
    mapping = get_model_weight_mapping(vllm_config)
    records: list[WeightRecord] = []
    total = 0

    for tensor in tensors:
        mapped_name = mapping.remap_checkpoint_name(tensor.name)
        local_bytes = _estimate_local_tensor_bytes(
            mapped_name, tensor, strategy, mapping
        )
        total += local_bytes
        if include_records:
            records.append(
                WeightRecord(
                    name=tensor.name,
                    param_name=mapped_name,
                    loaded_shape=tensor.shape,
                    loaded_dtype=str(tensor.dtype),
                    local_shape=tensor.shape,
                    local_dtype=str(tensor.dtype),
                    local_bytes=local_bytes,
                    note="checkpoint metadata estimate",
                )
            )

    if not tensors:
        total, records = estimate_weights_from_model_definition(
            vllm_config, include_records=include_records
        )
        notes.append("weights: vLLM meta model definition")
    else:
        notes.append(f"weights: safetensors metadata + {mapping.source}")

    return WeightSizingResult(total, records, notes)


def _load_safetensors_metadata(
    vllm_config: VllmConfig,
    notes: list[str],
) -> list[_TensorMetadata]:
    model_config = vllm_config.model_config
    model = model_config.model
    if os.path.isdir(model):
        hf_weights_files = glob.glob(os.path.join(model, "*.safetensors"))
    elif os.path.isfile(model) and model.endswith(".safetensors"):
        hf_weights_files = [model]
    else:
        tensors = _load_hf_safetensors_metadata(
            model, revision=model_config.revision
        )
        if tensors:
            return tensors
        notes.append("checkpoint: safetensors metadata unavailable")
        return []

    return _iter_safetensors_metadata(hf_weights_files, prefix="")


@cache
def _load_hf_safetensors_metadata(
    model: str,
    *,
    revision: str | None,
) -> tuple[_TensorMetadata, ...]:
    params_metadata = get_safetensors_params_metadata(model, revision=revision)
    tensors: list[_TensorMetadata] = []
    for name, info in params_metadata.items():
        if name == "__metadata__":
            continue
        dtype_name = info.get("dtype")
        shape = info.get("shape")
        if dtype_name is None or shape is None:
            continue
        dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(dtype_name)
        if dtype is None:
            continue
        tensors.append(
            _TensorMetadata(
                name=name,
                shape=tuple(shape),
                dtype=dtype,
            )
        )
    return tuple(tensors)


def _iter_safetensors_metadata(
    hf_weights_files: list[str],
    *,
    prefix: str,
) -> list[_TensorMetadata]:
    files = _expand_safetensors_files(hf_weights_files)
    tensors: list[_TensorMetadata] = []
    for st_file in sorted(files):
        if not os.path.exists(st_file):
            continue
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                weight_slice = f.get_slice(name)
                dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(weight_slice.get_dtype())
                if dtype is None:
                    continue
                tensors.append(
                    _TensorMetadata(
                        name=prefix + name,
                        shape=tuple(weight_slice.get_shape()),
                        dtype=dtype,
                    )
                )
    return tensors


def _expand_safetensors_files(hf_weights_files: list[str]) -> list[str]:
    files: list[str] = []
    for path in hf_weights_files:
        if any(char in path for char in "*?[]"):
            files.extend(glob.glob(path))
        else:
            files.append(path)
    return files


def _estimate_local_tensor_bytes(
    mapped_name: str,
    tensor: _TensorMetadata,
    strategy: Strategy,
    mapping: ModelWeightMapping,
) -> int:
    shard_factor = 1
    is_expert = _is_expert_tensor(mapped_name, mapping)
    if is_expert:
        shard_factor *= _expert_shard_factor(strategy)
    elif _is_tp_sharded_tensor(mapped_name, mapping):
        shard_factor *= max(strategy.tp, 1)
    if strategy.pp > 1 and _is_layer_tensor(mapped_name):
        shard_factor *= max(strategy.pp, 1)
    return (tensor.bytes + shard_factor - 1) // shard_factor


def _expert_shard_factor(strategy: Strategy) -> int:
    if strategy.ep > 1:
        return strategy.ep
    return max(strategy.dp * strategy.tp, 1)


def _is_tp_sharded_tensor(name: str, mapping: ModelWeightMapping) -> bool:
    if _matches_module_name(name, getattr(mapping, "tp_module_names", ())):
        return True
    if any(f".{param}." in name for param in mapping.packed_modules_mapping):
        return True
    if any(param in name for param in mapping.stacked_param_names):
        return True
    return any(hint in name for hint in _TP_SHARDED_HINTS)


def _is_expert_tensor(name: str, mapping: ModelWeightMapping) -> bool:
    if _matches_module_name(name, getattr(mapping, "expert_module_names", ())):
        return True
    if any(weight_name in name for weight_name in mapping.expert_weight_names):
        return True
    return any(hint in name for hint in _EXPERT_HINTS)


def _matches_module_name(name: str, module_names: Iterable[str]) -> bool:
    path_segments = name.split(".")
    return any(module_name in path_segments for module_name in module_names)


def _is_layer_tensor(name: str) -> bool:
    return ".layers." in name or ".h." in name or ".blocks." in name
