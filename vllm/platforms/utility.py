# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig


class UtilityPlatform(Platform):
    _enum = PlatformEnum.CUDA
    device_name: str = "NVIDIA H200"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    simple_compile_backend: str = "eager"
    _device_capability: DeviceCapability = DeviceCapability(9, 0)

    @classmethod
    def configure_target(cls, gpu_type: str | None) -> None:
        if gpu_type is None:
            return
        cls.device_name = _GPU_TYPE_TO_DEVICE_NAME.get(gpu_type, gpu_type)
        cls._device_capability = _GPU_TYPE_TO_CAPABILITY.get(
            gpu_type, DeviceCapability(9, 0)
        )

    @classmethod
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def import_ir_kernels(cls) -> None:
        pass

    @classmethod
    def is_utility_platform(cls) -> bool:
        return True

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return cls._device_capability

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return 0

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        return 0, 0

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if attn_selector_config.use_mla:
            raise NotImplementedError(
                "MLA attention is not supported on the utility platform yet."
            )
        if attn_selector_config.use_sparse:
            raise NotImplementedError(
                "Sparse attention is not supported on the utility platform."
            )
        return "vllm.v1.attention.backends.sizing.SizingAttentionBackend"

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        custom_ops = vllm_config.compilation_config.custom_ops
        if all(entry not in custom_ops for entry in ("all", "none")):
            custom_ops.append("none")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        pass

    @classmethod
    def device_count(cls) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability(89)

    @classmethod
    def supports_mx(cls) -> bool:
        return cls.has_device_capability(100)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True


_GPU_TYPE_TO_DEVICE_NAME = {
    "A10": "NVIDIA A10",
    "A10G": "NVIDIA A10G",
    "A40": "NVIDIA A40",
    "A100-40GB": "NVIDIA A100",
    "A100-80GB": "NVIDIA A100",
    "H100-80GB": "NVIDIA H100",
    "H200-141GB": "NVIDIA H200",
    "B100-192GB": "NVIDIA B100",
    "B200-180GB": "NVIDIA B200",
    "GB200-186GB": "NVIDIA GB200",
    "L4": "NVIDIA L4",
    "L40S": "NVIDIA L40S",
}

_GPU_TYPE_TO_CAPABILITY = {
    "A10": DeviceCapability(8, 6),
    "A10G": DeviceCapability(8, 6),
    "A40": DeviceCapability(8, 6),
    "A100-40GB": DeviceCapability(8, 0),
    "A100-80GB": DeviceCapability(8, 0),
    "H100-80GB": DeviceCapability(9, 0),
    "H200-141GB": DeviceCapability(9, 0),
    "B100-192GB": DeviceCapability(10, 0),
    "B200-180GB": DeviceCapability(10, 0),
    "GB200-186GB": DeviceCapability(10, 0),
    "L4": DeviceCapability(8, 9),
    "L40S": DeviceCapability(8, 9),
}
