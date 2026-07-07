# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack, contextmanager
import gc
import sys
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.tensorizer import meta_tensor_mode
from vllm.model_executor.model_loader.utils import (
    get_model_architecture,
    initialize_model,
)
from vllm.sizing.report import Strategy, WeightRecord
from vllm.utils.torch_utils import set_default_torch_dtype


class SizingModelLoader(BaseModelLoader):
    """Model loader that records model-defined tensor sizes on meta tensors."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        model_class, _ = get_model_architecture(model_config)
        with (
            _fake_parallel_context(
                Strategy(
                    dp=vllm_config.parallel_config.data_parallel_size,
                    tp=vllm_config.parallel_config.tensor_parallel_size,
                    ep=(
                        vllm_config.parallel_config.data_parallel_size
                        * vllm_config.parallel_config.tensor_parallel_size
                        if vllm_config.parallel_config.enable_expert_parallel
                        else 1
                    ),
                    pp=vllm_config.parallel_config.pipeline_parallel_size,
                    dcp=vllm_config.parallel_config.decode_context_parallel_size,
                ),
                model_class=model_class,
            ),
            set_default_torch_dtype(model_config.dtype),
            meta_tensor_mode(),
        ):
            return initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                model_class=model_class,
                prefix=prefix,
            )


def estimate_weights_from_model_definition(
    vllm_config: VllmConfig,
    *,
    include_records: bool = True,
) -> tuple[int, list[WeightRecord]]:
    model = SizingModelLoader(vllm_config.load_config).load_model(
        vllm_config=vllm_config,
        model_config=vllm_config.model_config,
    )
    records: list[WeightRecord] = []
    total = 0
    for name, tensor in _named_state_tensors(model):
        local_bytes = tensor.numel() * tensor.element_size()
        total += local_bytes
        if include_records:
            records.append(
                WeightRecord(
                    name=name,
                    param_name=name,
                    loaded_shape=tuple(tensor.shape),
                    loaded_dtype=str(tensor.dtype),
                    local_shape=tuple(tensor.shape),
                    local_dtype=str(tensor.dtype),
                    local_bytes=local_bytes,
                    note="model-definition tensor",
                )
            )
    del model
    gc.collect()
    return total, records


def _named_state_tensors(model: nn.Module):
    yield from model.named_parameters()
    for module_prefix, module in model.named_modules():
        persistent = {
            name
            for name in module._buffers
            if name not in module._non_persistent_buffers_set
        }
        for name, tensor in module.named_buffers(recurse=False):
            if name in persistent:
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                yield full_name, tensor


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0):
        self.world_size = size
        self.rank_in_group = rank
        self.rank = rank
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == size - 1

    def size(self) -> int:
        return self.world_size

    @property
    def device_group(self):
        return None

    @property
    def cpu_group(self):
        return None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def first_rank(self) -> int:
        return 0

    @property
    def last_rank(self) -> int:
        return self.world_size - 1

    @property
    def ranks(self) -> list[int]:
        return list(range(self.world_size))

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return tensor

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return tensor

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj


@contextmanager
def _fake_parallel_context(strategy: Strategy, model_class: type[nn.Module]):
    ep_size = strategy.ep if strategy.ep > 1 else strategy.dp * strategy.tp
    tp_group = _FakeGroup(strategy.tp)
    pp_group = _FakeGroup(strategy.pp)
    dp_group = _FakeGroup(strategy.dp)
    ep_group = _FakeGroup(ep_size)
    pcp_group = _FakeGroup(1)
    patches = [
        patch(
            "vllm.distributed.get_tensor_model_parallel_world_size",
            return_value=strategy.tp,
        ),
        patch("vllm.distributed.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.distributed.get_tp_group", return_value=tp_group),
        patch("vllm.distributed.get_pp_group", return_value=pp_group),
        patch("vllm.distributed.get_dp_group", return_value=dp_group),
        patch("vllm.distributed.get_ep_group", return_value=ep_group),
        patch("vllm.distributed.get_pcp_group", return_value=pcp_group),
        patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
            return_value=strategy.tp,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp_group),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=pp_group),
        patch("vllm.distributed.parallel_state.get_dp_group", return_value=dp_group),
        patch("vllm.distributed.parallel_state.get_ep_group", return_value=ep_group),
        patch(
            "vllm.distributed.parallel_state.get_pcp_group", return_value=pcp_group
        ),
        patch("torch.distributed.is_initialized", return_value=False),
    ]
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        _patch_imported_distributed_helpers(
            stack,
            model_class=model_class,
            tp_group=tp_group,
            pp_group=pp_group,
            dp_group=dp_group,
            ep_group=ep_group,
            pcp_group=pcp_group,
        )
        yield


def _patch_imported_distributed_helpers(
    stack: ExitStack,
    model_class: type[nn.Module],
    tp_group: _FakeGroup,
    pp_group: _FakeGroup,
    dp_group: _FakeGroup,
    ep_group: _FakeGroup,
    pcp_group: _FakeGroup,
) -> None:
    module_names = {
        name
        for name in sys.modules
        if name == model_class.__module__
        or name.startswith("vllm.model_executor.layers")
        or name.startswith("vllm.model_executor.models")
    }
    replacements = {
        "get_tensor_model_parallel_world_size": lambda: tp_group.world_size,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_pp_group": lambda: pp_group,
        "get_dp_group": lambda: dp_group,
        "get_ep_group": lambda: ep_group,
        "get_pcp_group": lambda: pcp_group,
        "tensor_model_parallel_all_gather": lambda tensor, dim=0: tensor,
    }
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attr, replacement in replacements.items():
            if hasattr(module, attr):
                stack.enter_context(patch.object(module, attr, replacement))
