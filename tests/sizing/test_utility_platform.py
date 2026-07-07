# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.envs as envs
import vllm.platforms as platforms
from vllm.platforms.interface import PlatformEnum
from vllm.platforms.utility import UtilityPlatform

pytestmark = pytest.mark.skip_global_cleanup


def test_utility_platform_activates_for_empty_target(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_TARGET_DEVICE", "empty")

    assert (
        platforms.utility_platform_plugin()
        == "vllm.platforms.utility.UtilityPlatform"
    )
    assert (
        platforms.resolve_current_platform_cls_qualname()
        == "vllm.platforms.utility.UtilityPlatform"
    )


def test_utility_platform_is_not_unspecified() -> None:
    platform = UtilityPlatform()

    assert platform._enum == PlatformEnum.CUDA
    assert platform.is_cuda()
    assert not platform.is_cpu()
    assert not platform.is_unspecified()
    assert platform.device_type == "cuda"


def test_utility_platform_configures_target_gpu() -> None:
    UtilityPlatform.configure_target("H200-141GB")

    assert UtilityPlatform.get_device_name() == "NVIDIA H200"
    assert UtilityPlatform.get_device_capability().as_version_str() == "9.0"


def test_utility_platform_uses_sizing_attention_backend() -> None:
    backend_path = UtilityPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=SimpleNamespace(use_mla=False, use_sparse=False),
    )

    assert (
        backend_path
        == "vllm.v1.attention.backends.sizing.SizingAttentionBackend"
    )
