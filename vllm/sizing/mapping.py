# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import sys
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.model_executor.models import ModelRegistry

_TP_MODULE_TYPES = frozenset(
    {
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "QKVParallelLinear",
        "RowParallelLinear",
        "ParallelLMHead",
        "VocabParallelEmbedding",
    }
)
_REPLICATED_MODULE_TYPES = frozenset({"ReplicatedLinear"})
_EXPERT_MODULE_TYPES = frozenset({"FusedMoE"})


@dataclass(frozen=True)
class ModelWeightMapping:
    packed_modules_mapping: dict[str, list[str]]
    stacked_tuples: tuple[tuple[str, str, object], ...]
    stacked_param_names: frozenset[str]
    stacked_weight_names: frozenset[str]
    expert_weight_names: frozenset[str]
    tp_module_names: frozenset[str]
    replicated_module_names: frozenset[str]
    expert_module_names: frozenset[str]
    source: str

    def remap_checkpoint_name(self, name: str) -> str:
        for packed_name, weight_names in self.packed_modules_mapping.items():
            for weight_name in weight_names:
                token = f".{weight_name}."
                if token in name:
                    return name.replace(token, f".{packed_name}.")
        for packed_name, weight_name, _shard_id in self.stacked_tuples:
            if weight_name in name:
                return name.replace(weight_name, packed_name)
        return name


def get_model_weight_mapping(vllm_config: VllmConfig) -> ModelWeightMapping:
    try:
        model_cls, arch = ModelRegistry.resolve_model_cls(
            vllm_config.model_config.architectures,
            model_config=vllm_config.model_config,
        )
    except Exception:
        return ModelWeightMapping(
            {},
            tuple(),
            frozenset(),
            frozenset(),
            frozenset(),
            frozenset(),
            frozenset(),
            frozenset(),
            "none",
        )

    packed = getattr(model_cls, "packed_modules_mapping", None) or {}
    stacked = _extract_literal_tuple_list(model_cls, "stacked_params_mapping")
    experts = _extract_literal_tuple_list(model_cls, "expert_params_mapping")
    module_mapping = _extract_parallel_module_names(model_cls)
    expert_weight_names = frozenset(
        item[1] for item in experts if len(item) >= 2 and isinstance(item[1], str)
    )
    source = arch
    if stacked or experts:
        source += " load_weights source"
    if packed:
        source += " packed_modules_mapping"
    if module_mapping.has_values():
        source += " AST module mapping"
    return ModelWeightMapping(
        packed_modules_mapping=dict(packed),
        stacked_tuples=tuple(
            item
            for item in stacked
            if len(item) >= 3 and isinstance(item[0], str) and isinstance(item[1], str)
        ),
        stacked_param_names=frozenset(
            item[0] for item in stacked if item and isinstance(item[0], str)
        ),
        stacked_weight_names=frozenset(
            item[1] for item in stacked if len(item) >= 2 and isinstance(item[1], str)
        ),
        expert_weight_names=expert_weight_names,
        tp_module_names=module_mapping.tp,
        replicated_module_names=module_mapping.replicated,
        expert_module_names=module_mapping.expert,
        source=source,
    )


def _extract_literal_tuple_list(
    model_cls: type,
    variable_name: str,
) -> list[tuple]:
    try:
        source = inspect.getsource(model_cls)
    except (OSError, TypeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    values: list[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        try:
            literal = ast.literal_eval(node.value)
        except (SyntaxError, ValueError):
            continue
        if isinstance(literal, list):
            values.extend(item for item in literal if isinstance(item, tuple))
    return values


@dataclass(frozen=True)
class _ModuleMapping:
    tp: frozenset[str]
    replicated: frozenset[str]
    expert: frozenset[str]

    def has_values(self) -> bool:
        return bool(self.tp or self.replicated or self.expert)


def _extract_parallel_module_names(model_cls: type) -> _ModuleMapping:
    module = sys.modules.get(model_cls.__module__)
    if module is None:
        return _empty_module_mapping()
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return _empty_module_mapping()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _empty_module_mapping()

    tp: set[str] = set()
    replicated: set[str] = set()
    expert: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        attr_name = _single_self_attr_name(node.targets)
        if attr_name is None:
            continue
        call_name = _call_name(node.value)
        if call_name in _TP_MODULE_TYPES:
            tp.add(attr_name)
        elif call_name in _REPLICATED_MODULE_TYPES:
            replicated.add(attr_name)
        elif call_name in _EXPERT_MODULE_TYPES:
            expert.add(attr_name)

    return _ModuleMapping(
        tp=frozenset(tp),
        replicated=frozenset(replicated),
        expert=frozenset(expert),
    )


def _empty_module_mapping() -> _ModuleMapping:
    return _ModuleMapping(frozenset(), frozenset(), frozenset())


def _single_self_attr_name(targets: list[ast.expr]) -> str | None:
    if len(targets) != 1:
        return None
    target = targets[0]
    if not isinstance(target, ast.Attribute):
        return None
    if not isinstance(target.value, ast.Name) or target.value.id != "self":
        return None
    return target.attr


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None
