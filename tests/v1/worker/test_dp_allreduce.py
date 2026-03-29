# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DP scheduler all_reduce backends.

Tests the GlooDPAllReduce and NCCLSideStreamDPAllReduce backends
using real multi-process torch.distributed groups.
"""

import os
import traceback

import pytest
import torch
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]

from vllm.utils.network_utils import get_open_port


def _worker_test_gloo_allreduce(
    local_rank: int,
    world_size: int,
    init_method: str,
):
    """Worker function that tests GlooDPAllReduce."""
    from vllm.distributed.dp_allreduce import GlooDPAllReduce

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
    )

    try:
        group = torch.distributed.group.WORLD
        backend = GlooDPAllReduce(group)

        # Each rank fills its own column, all others zero
        tensor = torch.zeros(4, world_size, dtype=torch.int32, device="cpu")
        tensor[0][local_rank] = 100 + local_rank  # orig_tokens
        tensor[1][local_rank] = 200 + local_rank  # padded_tokens
        tensor[2][local_rank] = 1  # should_ubatch
        tensor[3][local_rank] = 2  # cudagraph_mode

        backend.all_reduce(tensor)

        # After all_reduce (sum), each column should have the sender's values
        for r in range(world_size):
            assert tensor[0][r].item() == 100 + r, (
                f"rank {local_rank}: expected orig_tokens[{r}]={100 + r}, "
                f"got {tensor[0][r].item()}"
            )
            assert tensor[1][r].item() == 200 + r
            assert tensor[2][r].item() == 1
            assert tensor[3][r].item() == 2

        backend.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def _worker_test_nccl_side_stream_allreduce(
    local_rank: int,
    world_size: int,
    init_method: str,
):
    """Worker function that tests NCCLSideStreamDPAllReduce."""
    from vllm.distributed.dp_allreduce import NCCLSideStreamDPAllReduce

    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        group = torch.distributed.group.WORLD
        backend = NCCLSideStreamDPAllReduce(group, world_size, device)

        # Each rank fills its own column
        tensor = torch.zeros(4, world_size, dtype=torch.int32, device="cpu")
        tensor[0][local_rank] = 10 + local_rank
        tensor[1][local_rank] = 20 + local_rank
        tensor[2][local_rank] = 1
        tensor[3][local_rank] = 2

        backend.all_reduce(tensor)

        # Verify all columns are filled correctly
        for r in range(world_size):
            assert tensor[0][r].item() == 10 + r, (
                f"rank {local_rank}: expected [0][{r}]={10 + r}, "
                f"got {tensor[0][r].item()}"
            )
            assert tensor[1][r].item() == 20 + r
            assert tensor[2][r].item() == 1
            assert tensor[3][r].item() == 2

        # Run multiple iterations to test stability
        for i in range(100):
            tensor.zero_()
            tensor[0][local_rank] = i
            tensor[1][local_rank] = i * 2
            tensor[2][local_rank] = 1
            tensor[3][local_rank] = 0
            backend.all_reduce(tensor)
            for r in range(world_size):
                assert tensor[0][r].item() == i
                assert tensor[1][r].item() == i * 2

        backend.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


def _worker_test_gloo_matches_nccl(
    local_rank: int,
    world_size: int,
    init_method: str,
):
    """Worker: verify GlooDPAllReduce and NCCLSideStreamDPAllReduce
    produce identical results."""
    from vllm.distributed.dp_allreduce import (
        GlooDPAllReduce,
        NCCLSideStreamDPAllReduce,
    )

    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)

    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        # Create both backends — gloo uses CPU group, NCCL uses device group
        gloo_group = torch.distributed.new_group(
            list(range(world_size)), backend="gloo"
        )
        nccl_group = torch.distributed.new_group(
            list(range(world_size)), backend="nccl"
        )

        gloo_backend = GlooDPAllReduce(gloo_group)
        nccl_backend = NCCLSideStreamDPAllReduce(
            nccl_group, world_size, device
        )

        torch.manual_seed(42 + local_rank)

        for _ in range(50):
            vals = torch.randint(0, 1000, (4,), dtype=torch.int32)

            gloo_tensor = torch.zeros(
                4, world_size, dtype=torch.int32, device="cpu"
            )
            nccl_tensor = torch.zeros(
                4, world_size, dtype=torch.int32, device="cpu"
            )

            for row in range(4):
                gloo_tensor[row][local_rank] = vals[row]
                nccl_tensor[row][local_rank] = vals[row]

            gloo_backend.all_reduce(gloo_tensor)
            nccl_backend.all_reduce(nccl_tensor)

            assert torch.equal(gloo_tensor, nccl_tensor), (
                f"rank {local_rank}: gloo and nccl results differ\n"
                f"gloo: {gloo_tensor}\nnccl: {nccl_tensor}"
            )

        gloo_backend.close()
        nccl_backend.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skip_global_cleanup
class TestGlooDPAllReduce:
    """Test GlooDPAllReduce backend (CPU-only, no GPU needed)."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_basic_allreduce(self, world_size):
        port = get_open_port()
        init_method = (
            f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{port}"
        )
        spawn(
            _worker_test_gloo_allreduce,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.skip_global_cleanup
class TestNCCLSideStreamDPAllReduce:
    """Test NCCLSideStreamDPAllReduce backend (requires GPUs)."""

    @pytest.mark.parametrize("world_size", [2])
    def test_basic_allreduce(self, world_size, num_gpus_available):
        if num_gpus_available < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")
        port = get_open_port()
        init_method = (
            f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{port}"
        )
        spawn(
            _worker_test_nccl_side_stream_allreduce,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.parametrize("world_size", [2])
    def test_matches_gloo(self, world_size, num_gpus_available):
        """Verify NCCL side-stream produces same results as gloo."""
        if num_gpus_available < world_size:
            pytest.skip(f"Need at least {world_size} GPUs")
        port = get_open_port()
        init_method = (
            f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{port}"
        )
        spawn(
            _worker_test_gloo_matches_nccl,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.skip_global_cleanup
class TestCreateDPAllReduceBackend:
    """Test the factory function."""

    def test_unknown_backend_raises(self):
        from vllm.distributed.dp_allreduce import create_dp_allreduce_backend

        with pytest.raises(ValueError, match="Unknown dp_cpu_backend"):
            create_dp_allreduce_backend(
                backend_name="invalid",
                dp_rank=0,
                dp_size=2,
                cpu_group=None,
                device_group=None,
                cuda_device=None,
            )

    def test_ucx_import_error(self):
        """UCX backend should give a clear error if ucx-py is missing."""
        from vllm.distributed.dp_allreduce import UCXDPAllReduce

        with pytest.raises(ImportError, match="ucx-py"):
            UCXDPAllReduce(rank=0, world_size=2, cpu_group=None)
