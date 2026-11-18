# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Must be imported firstly
import vllm.v1.worker.cpu.shm  # noqa # isort: skip

import math
import mmap
import os
import resource
import sys

import psutil
import torch

from vllm import envs
from vllm.config import CompilationMode, VllmConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.cpu_resource_utils import (
    get_allowed_cpu_list,
    get_cgroup_memory_headroom,
    get_memory_node_info,
    get_worker_memory_nodes,
)
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.cpu.model_runner import CPUModelRunner
from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)

_CPU_KV_CACHE_MIN_HEADROOM = 256 << 20


class CPUWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        # TODO: use numactl for process setup
        # TODO: optimize for `interleaved` policy
        # Bind memory node
        allowed_memory_nodes = get_worker_memory_nodes(
            vllm_config.parallel_config.assigned_physical_gpu_ids
        )
        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        # TODO: some CI hosts are not correctly set, change to assertion
        # after fix
        if cpu_core.numa_node not in allowed_memory_nodes:
            logger.warning(
                "Node %s is not in available memory nodes %s.",
                cpu_core.numa_node,
                allowed_memory_nodes,
            )

        # On s390x, numa_node may be a synthetic book ID that doesn't
        # correspond to a real memory node. Fall back to first visible node.
        if cpu_core.numa_node in allowed_memory_nodes:
            memory_node = cpu_core.numa_node
        else:
            logger.warning(
                "CPU group key %s is not a valid memory node. "
                "Falling back to memory node %s.",
                cpu_core.numa_node,
                allowed_memory_nodes[0],
            )
            memory_node = allowed_memory_nodes[0]

        torch.ops._C.init_cpu_memory_env([memory_node])

        self.memory_node = memory_node
        memory_status = get_memory_node_info(memory_node)
        memory_fraction = vllm_config.cache_config.resolved_gpu_memory_utilization
        self.requested_cpu_memory = math.ceil(
            memory_status.total_memory * memory_fraction
        )

        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self):
        self.device = torch.device("cpu")

        # Check whether critical libraries are loaded
        def check_preloaded_libs(name: str) -> bool:
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )
                return False
            return True

        if sys.platform.startswith("linux"):
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                iomp_loaded = check_preloaded_libs("libiomp")
                if not iomp_loaded and self.vllm_config.speculative_config is not None:
                    logger.warning(
                        "Speculative decoding on CPU without Intel OpenMP in "
                        "LD_PRELOAD will cause significant performance loss. "
                        "Please follow the section `set LD_PRELOAD` in "
                        "https://docs.vllm.ai/en/latest/getting_started/"
                        "installation/cpu/ "
                        "to setup libiomp5.",
                    )

        def skip_set_num_threads(x: int):
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

        init_method = self.distributed_init_method
        os.environ["VLLM_DIST_IDENT"] = (
            os.path.basename(init_method.removeprefix("file://"))
            if init_method.startswith("file://")
            else init_method.split(":")[-1]
        )
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        if current_platform.uses_standalone_cpu_model_runner():
            from vllm.v1.worker.cpu.standalone_model_runner import (
                StandaloneCPUModelRunner,
            )

            self.model_runner: CPUModelRunner = StandaloneCPUModelRunner(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            self.model_runner = CPUModelRunner(self.vllm_config, self.device)

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def discard(self, tags: tuple[str, ...]) -> None:
        raise RuntimeError("Selective discard is not supported on CPU.")

    def _should_warm_up_model(self) -> bool:
        # VLLM_CPU_CI_ENV always skips warmup to save CI time.
        if envs.VLLM_CPU_CI_ENV:
            return False
        # With eager (CompilationMode.NONE) execution and an explicit KV
        # cache size, warmup serves no purpose: there's no compiled graph to
        # prewarm, and the auto KV-cache-size calc below -- which relies on
        # a warmup forward pass having already bumped RSS to a realistic
        # steady-state value -- is bypassed whenever the size is explicit.
        return not (
            self.compilation_config.mode == CompilationMode.NONE
            and self.cache_config.kv_cache_memory_bytes is not None
        )

    def determine_available_memory(self) -> int:
        if self.cache_config.enable_extensible_kv_cache:
            return self._profile_extensible_kv_cache_memory()
        if self._should_warm_up_model():
            self.model_runner.warming_up_model()

        allowed_cpu_list = get_allowed_cpu_list()
        cpu_core = allowed_cpu_list[0]

        memory_status = get_memory_node_info(cpu_core.numa_node)
        available_memory = memory_status.available_memory
        explicit_kv_cache_size = self.cache_config.kv_cache_memory_bytes

        kv_cache_size = None
        msg = None
        if explicit_kv_cache_size is not None:
            if explicit_kv_cache_size > available_memory:
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(explicit_kv_cache_size)} GiB). "
                    "Decrease --kv-cache-memory-bytes, VLLM_CPU_KVCACHE_SPACE, "
                    "or reduce CPU memory used by other processes."
                )
            kv_cache_size = explicit_kv_cache_size
            msg = (
                f"Explicitly set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}."
            )
        else:
            consumed_memory = psutil.Process(os.getpid()).memory_info().rss
            requested_memory_for_kv = int(self.requested_cpu_memory - consumed_memory)
            if requested_memory_for_kv <= 0:
                # The worker's own footprint already fills its reservation, so
                # freeing memory elsewhere cannot help: only a larger
                # reservation or an explicit KV cache size can.
                in_use = math.ceil(consumed_memory / memory_status.total_memory * 100)
                raise ValueError(
                    f"The worker already uses {format_gib(consumed_memory)} GiB "
                    "(model weights and runtime), more than the "
                    f"{format_gib(self.requested_cpu_memory)} GiB of CPU memory "
                    f"reserved for it on node {cpu_core.numa_node}, leaving none "
                    "for the KV cache. On the CPU backend, the "
                    "`--gpu-memory-utilization` flag controls the fraction of CPU "
                    "memory reserved (despite its name). To resolve: increase "
                    f"`--gpu-memory-utilization` to more than {in_use / 100:.2f}, "
                    "or set the KV cache size explicitly with "
                    "`--kv-cache-memory-bytes` or VLLM_CPU_KVCACHE_SPACE."
                )
            if requested_memory_for_kv > available_memory:
                raise ValueError(
                    f"Available memory on node {cpu_core.numa_node} "
                    f"({format_gib(available_memory)}/"
                    f"{format_gib(memory_status.total_memory)} GiB) on kv cache"
                    f" allocation is less than requested memory for kv "
                    f"({format_gib(requested_memory_for_kv)}/"
                    f"{format_gib(self.requested_cpu_memory)} GiB). "
                    "Reduce CPU memory used by other processes."
                )
            kv_cache_size = requested_memory_for_kv
            msg = (
                f"Auto set ({format_gib(kv_cache_size)}/"
                f"{format_gib(memory_status.total_memory)}) GiB for KV cache "
                f"on node {cpu_core.numa_node}, with "
                f"{format_gib(self.requested_cpu_memory)} GiB requested memory"
                f" for the worker. {format_gib(consumed_memory)} GiB"
                f" memory was consumed by non-kv usages."
            )

        logger.info(msg)

        return kv_cache_size

    def extensible_kv_cache_unsupported_reason(self) -> str | None:
        config = self.vllm_config
        parallel = self.parallel_config
        if (
            parallel.nnodes != 1
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            return "CPU extensible KV cache supports single-node tensor parallelism"
        if (
            config.kv_transfer_config is not None
            or config.speculative_config is not None
            or config.model_config.supports_multimodal_inputs
            or config.model_config.is_encoder_decoder
            or config.aux_output_config.enabled
            or config.weight_transfer_config is not None
        ):
            return "CPU extensible KV cache requires a decoder-only text model"
        return None

    def extend_kv_cache(self, num_blocks: int) -> None:
        super().extend_kv_cache(num_blocks)
        if self.model_runner.kv_cache_config.needs_kv_cache_zeroing:
            self.model_runner._init_kv_zero_meta()

    def disable_extensible_kv_cache(self) -> None:
        self.cache_config.enable_extensible_kv_cache = False
        self.requested_cpu_memory = math.ceil(
            get_memory_node_info(self.memory_node).total_memory
            * self.cache_config.resolved_gpu_memory_utilization
        )

    def _extensible_kv_cache_granule(self) -> int:
        return mmap.PAGESIZE

    def _extensible_kv_cache_warmup_limit(self, kv_cache: ExtensibleKVCache) -> int:
        bytes_available = self._extensible_kv_cache_budget() + kv_cache.physical_bytes
        return max(0, kv_cache.blocks_within(bytes_available))

    def _extensible_kv_cache_budget(self) -> int:
        # Divide the shared headroom before adding back a worker's own
        # committed cache pages. All workers can then grow concurrently.
        available = get_memory_node_info(self.memory_node).available_memory
        available //= self._ranks_on_memory_node
        cgroup_headroom = get_cgroup_memory_headroom()
        if cgroup_headroom is not None:
            available = min(
                available, cgroup_headroom // self.parallel_config.world_size
            )
        reserve = max(
            _CPU_KV_CACHE_MIN_HEADROOM, available // 10, self._cpu_warmup_peak
        )
        return max(0, available - reserve)

    def _profile_extensible_kv_cache_memory(self) -> int:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if self._should_warm_up_model():
            self.model_runner.warming_up_model()
        # Linux reports ru_maxrss in KiB. The high-water increase is a
        # lower bound when model loading had a higher peak than profiling.
        self._cpu_warmup_peak = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak
        ) * 1024
        group = get_world_group().cpu_group
        memory_nodes: list[int | None] = [None] * group.size()
        torch.distributed.all_gather_object(memory_nodes, self.memory_node, group)
        self._ranks_on_memory_node = memory_nodes.count(self.memory_node)
        return self._extensible_kv_cache_budget()

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # Note: the model has been compiled in determine_available_memory(),
        # Only compile here for models without kv cache
        if len(self.model_runner.kv_caches) == 0 and self._should_warm_up_model():
            self.model_runner.warming_up_model()
        kv_cache = getattr(self.model_runner, "extensible_kv_cache", None)
        if kv_cache is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            backed_bytes = kv_cache.physical_bytes
        # profile_run skips the scheduler-driven kernels, so without this
        # their Triton JIT lands in the first request's TTFT.
        if self.vllm_config.kernel_config.enable_jit_warmup:
            if current_platform.uses_standalone_cpu_model_runner():
                from vllm.v1.worker.cpu.warmup import warmup_kernels
            else:
                from vllm.v1.worker.gpu.warmup import warmup_kernels

            warmup_kernels(
                self.model_runner,
                self.execute_model,
                self.sample_tokens,
            )
        num_kv_blocks = None
        if kv_cache is not None:
            peak_growth = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - peak
            ) * 1024
            self._cpu_warmup_peak = max(
                self._cpu_warmup_peak,
                peak_growth - (kv_cache.physical_bytes - backed_bytes),
            )
            group = get_world_group().cpu_group
            torch.distributed.barrier(group=group)
            num_kv_blocks = self._extensible_kv_cache_warmup_limit(kv_cache)
            logger.info(
                "CPU KV cache budget after warmup fits %d blocks.", num_kv_blocks
            )
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(
            self.model_config.seed, self.parallel_config.data_parallel_index
        )
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
            num_kv_blocks=num_kv_blocks,
        )
