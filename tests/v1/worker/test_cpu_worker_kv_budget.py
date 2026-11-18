# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU worker's automatic KV cache sizing must report the right cause.

A worker whose own footprint exceeds its reservation and a node that is short
of free memory need opposite remedies, so they get separate errors.
"""

import sys
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.cpu_resource_utils import MemoryNodeInfo

# Importing the CPU worker patches torch and the GPU buffer utilities for the
# whole process (vllm/v1/worker/cpu/shm.py), so the import is deferred into
# the tests and they run only on the CPU platform.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU worker test; needs the CPU platform"
)

GiB = 1 << 30


def _worker(monkeypatch, *, reserved, rss, total=8 * GiB, available=4 * GiB):
    import vllm.v1.worker.cpu_worker as cpu_worker_mod
    from vllm.v1.worker.cpu_worker import CPUWorker

    worker = CPUWorker.__new__(CPUWorker)
    worker.requested_cpu_memory = reserved
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None, enable_extensible_kv_cache=False
    )
    monkeypatch.setattr(CPUWorker, "_should_warm_up_model", lambda self: False)
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_allowed_cpu_list",
        lambda: [SimpleNamespace(numa_node=0)],
    )
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_memory_node_info",
        lambda node: MemoryNodeInfo(total_memory=total, available_memory=available),
    )
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss))
    monkeypatch.setattr(cpu_worker_mod.psutil, "Process", lambda pid: process)
    return worker


def test_footprint_above_reservation_asks_for_a_larger_reservation(monkeypatch):
    # 0.76 GiB already in use against a 0.16 GiB reservation (2% of 8 GiB).
    worker = _worker(monkeypatch, reserved=int(0.16 * GiB), rss=int(0.76 * GiB))
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    msg = str(exc.value)
    assert "increase `--gpu-memory-utilization` to more than 0.10" in msg
    assert "--kv-cache-memory-bytes" in msg
    # Neither the misleading remedy nor a negative KV size.
    assert "other processes" not in msg
    assert "-0." not in msg


def test_short_node_memory_still_asks_to_free_memory(monkeypatch):
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=2 * GiB)
    with pytest.raises(ValueError, match="Reduce CPU memory used by other processes"):
        worker.determine_available_memory()


def test_budget_that_fits_sizes_the_kv_cache(monkeypatch):
    worker = _worker(monkeypatch, reserved=3 * GiB, rss=1 * GiB)
    assert worker.determine_available_memory() == 2 * GiB


@pytest.mark.skipif(
    sys.platform != "linux", reason="CPU extensible KV cache requires Linux"
)
@pytest.mark.parametrize("mixed_attention", [False, True], ids=["full", "mixed"])
def test_automatic_cpu_kv_cache_generates_after_warmup(
    monkeypatch, tmp_path, vllm_runner, mixed_attention
):
    from transformers import Gemma2Config, OPTConfig

    from vllm import SamplingParams
    from vllm.v1.worker import cpu_worker

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.delenv("VLLM_CPU_KVCACHE_SPACE", raising=False)
    monkeypatch.setattr(
        cpu_worker, "get_memory_node_info", lambda _: MemoryNodeInfo(8 * GiB, GiB)
    )
    monkeypatch.setattr(cpu_worker, "get_cgroup_memory_headroom", lambda: GiB)

    common = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    if mixed_attention:
        config = Gemma2Config(
            **common,
            architectures=["Gemma2ForCausalLM"],
            intermediate_size=256,
            num_key_value_heads=2,
            head_dim=64,
            sliding_window=32,
            query_pre_attn_scalar=64,
            attn_logit_softcapping=None,
        )
    else:
        config = OPTConfig(**common, architectures=["OPTForCausalLM"], ffn_dim=256)
    config.save_pretrained(tmp_path)

    kwargs = dict(
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        kernel_config={"enable_jit_warmup": True},
    )
    prompts = [{"prompt_token_ids": [1, *range(3, 54)]}]
    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
    )
    with vllm_runner(
        str(tmp_path), kv_cache_memory_bytes=128 << 20, **kwargs
    ) as runner:
        expected = runner.llm.generate(prompts, sampling)[0].outputs[0].token_ids
    assert len(expected) == sampling.max_tokens
    with vllm_runner(str(tmp_path), **kwargs) as runner:
        cache_config = runner.llm.llm_engine.vllm_config.cache_config
        assert cache_config.enable_extensible_kv_cache
        assert cache_config.num_gpu_blocks > 1
        assert (
            runner.llm.generate(prompts, sampling)[0].outputs[0].token_ids == expected
        )
