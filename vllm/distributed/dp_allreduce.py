# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side all_reduce backends for DP scheduler synchronization.

The default gloo backend uses TCP, which is susceptible to kernel-level
jitter and retransmission timeouts that can cause multi-second stalls
when one rank is a straggler at the per-iteration collective barrier.

Alternative backends:
- nccl-side-stream: Uses NCCL on a dedicated CUDA stream. No new deps,
  avoids TCP entirely. The side stream does not block the main compute
  stream, preserving async scheduling benefits.
- ucx: Uses UCX RDMA for true CPU-side RDMA. Requires ucx-py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class DPAllReduceBackend(ABC):
    """Abstract interface for DP scheduler all_reduce."""

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> None:
        """In-place sum all_reduce. Tensor is CPU, int32, shape (4, dp_size).
        Each rank populates only its own column; all other columns are zero.
        So sum all_reduce == allgather."""
        ...

    def close(self) -> None:
        pass


class GlooDPAllReduce(DPAllReduceBackend):
    """Default: gloo TCP all_reduce via torch.distributed."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group

    def all_reduce(self, tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, group=self.group)


class NCCLSideStreamDPAllReduce(DPAllReduceBackend):
    """NCCL all_reduce on a dedicated CUDA stream.

    Avoids gloo TCP entirely by routing the small all_reduce through NCCL
    (which uses NVLink/IB). A dedicated side stream ensures the main
    compute stream is not blocked — only the CPU thread waits for the
    side stream to finish, which is the same blocking behavior as gloo
    but with RDMA-level latency (~50us vs TCP's 100us+ with tail spikes).
    """

    def __init__(
        self,
        device_group: dist.ProcessGroup,
        dp_size: int,
        cuda_device: torch.device,
    ):
        self.device_group = device_group
        self.side_stream = torch.cuda.Stream(device=cuda_device)
        # Pre-allocate pinned CPU buffer and GPU buffer
        self.gpu_buf = torch.zeros(
            4, dp_size, dtype=torch.int32, device=cuda_device
        )
        self.pinned_buf = torch.zeros(
            4, dp_size, dtype=torch.int32, device="cpu"
        ).pin_memory()

    def all_reduce(self, tensor: torch.Tensor) -> None:
        # Copy input to pinned buffer, then to GPU on the side stream
        self.pinned_buf.copy_(tensor)
        with torch.cuda.stream(self.side_stream):
            self.gpu_buf.copy_(self.pinned_buf, non_blocking=True)
            dist.all_reduce(self.gpu_buf, group=self.device_group)
            self.pinned_buf.copy_(self.gpu_buf, non_blocking=True)
        self.side_stream.synchronize()
        # Copy result back to the caller's tensor
        tensor.copy_(self.pinned_buf)


class UCXDPAllReduce(DPAllReduceBackend):
    """UCX RDMA all_reduce for true CPU-side RDMA.

    Uses the low-level UCX API (ucp._libs.ucx_api) to create endpoints
    directly from exchanged worker addresses, bypassing ucx-py's
    built-in TCP handshake which can fail in Kubernetes environments.

    Worker addresses are exchanged via gloo (one-time at init).
    Per-iteration allgather uses UCX tag send/recv over RDMA.

    Requires ucx-py: pip install ucx-py (or build from source)
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        cpu_group: dist.ProcessGroup,
    ):
        try:
            from ucp._libs import ucx_api  # noqa: F401
        except ImportError:
            raise ImportError(
                "UCX DP backend requires ucx-py. "
                "Install with: pip install ucx-py-cu12"
            ) from None

        self.rank = rank
        self.world_size = world_size
        self._chunk_bytes = 4 * 4  # 4 int32s per column = 16 bytes
        self._init_endpoints(cpu_group)

        logger.info("UCX DP all_reduce initialized: rank %d/%d", rank, world_size)

    def _init_endpoints(self, cpu_group: dist.ProcessGroup):
        """Create UCX endpoints using low-level API.

        1. Create UCX context + worker
        2. Get worker address (serializable bytes)
        3. Exchange addresses via gloo all_reduce (one-time)
        4. Create endpoints directly from addresses (no TCP handshake)
        """
        from ucp._libs.ucx_api import (
            UCXAddress,
            UCXContext,
            UCXEndpoint,
            UCXWorker,
        )

        right_rank = (self.rank + 1) % self.world_size

        # Create UCX context and worker
        self._ctx = UCXContext()
        self._worker = UCXWorker(self._ctx)

        # Get our worker address as bytes
        my_addr = self._worker.get_address()
        my_addr_bytes = bytes(my_addr)
        addr_len = len(my_addr_bytes)

        # Exchange worker addresses via gloo all_reduce
        # Pad to fixed size. GB200 NVL72 with many IB devices has ~728 byte
        # addresses, so use 1024 to be safe.
        MAX_ADDR_LEN = 1024
        assert addr_len <= MAX_ADDR_LEN, (
            f"UCX address too long: {addr_len} > {MAX_ADDR_LEN}"
        )

        # Pack: 4 bytes length + address bytes, padded to MAX_ADDR_LEN + 4
        import struct
        my_info = struct.pack(">I", addr_len) + my_addr_bytes.ljust(
            MAX_ADDR_LEN, b"\0"
        )
        total_bytes = MAX_ADDR_LEN + 4  # 516 bytes
        n_ints = total_bytes // 4  # 129 int32s per rank

        info_tensor = torch.zeros(
            self.world_size, n_ints, dtype=torch.int32, device="cpu"
        )
        my_ints = struct.unpack(f"{n_ints}i", my_info)
        for i, v in enumerate(my_ints):
            info_tensor[self.rank, i] = v
        dist.all_reduce(info_tensor, group=cpu_group)

        # Barrier: ensure all ranks have exchanged addresses
        dist.barrier(group=cpu_group)

        # Decode right neighbor's worker address
        right_info = struct.pack(
            f"{n_ints}i",
            *[int(info_tensor[right_rank, i]) for i in range(n_ints)],
        )
        right_addr_len = struct.unpack(">I", right_info[:4])[0]
        right_addr_bytes = right_info[4 : 4 + right_addr_len]

        # Create endpoint to right neighbor directly from address
        right_addr = UCXAddress.from_buffer(right_addr_bytes)
        self._right_ep = UCXEndpoint.create_from_worker_address(
            self._worker, right_addr
        )

        logger.info(
            "UCX endpoint: rank %d → right=%d (addr %d bytes)",
            self.rank, right_rank, right_addr_len,
        )

        # For the ring, we also need the left neighbor to connect to us.
        # With create_from_worker_address, connections are one-sided —
        # we just need left neighbor's endpoint to us, which they create
        # on their side. We create our endpoint to the left neighbor too.
        left_rank = (self.rank - 1) % self.world_size
        left_info = struct.pack(
            f"{n_ints}i",
            *[int(info_tensor[left_rank, i]) for i in range(n_ints)],
        )
        left_addr_len = struct.unpack(">I", left_info[:4])[0]
        left_addr_bytes = left_info[4 : 4 + left_addr_len]
        left_addr = UCXAddress.from_buffer(left_addr_bytes)
        self._left_ep = UCXEndpoint.create_from_worker_address(
            self._worker, left_addr
        )

        logger.info(
            "UCX ring ready: rank %d, left=%d, right=%d",
            self.rank, left_rank, right_rank,
        )

    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Ring allgather using UCX tag send/recv."""
        from ucp._libs.ucx_api import tag_recv_nb, tag_send_nb

        N = self.world_size
        data = tensor.numpy()

        for step in range(N - 1):
            send_col = (self.rank - step) % N
            recv_col = (self.rank - step - 1) % N

            send_data = data[:, send_col].tobytes()
            recv_buf = bytearray(self._chunk_bytes)

            tag = step & 0xFFFFFFFF

            # Post send and recv
            send_req = tag_send_nb(
                self._right_ep, send_data, len(send_data), tag,
            )
            recv_req = tag_recv_nb(
                self._worker, recv_buf, len(recv_buf), tag,
            )

            # Progress until both complete
            while not (send_req.is_completed and recv_req.is_completed):
                self._worker.progress()

            # Unpack received column
            import struct
            vals = struct.unpack("4i", recv_buf)
            for row in range(4):
                data[row, recv_col] = vals[row]

    def close(self):
        pass


def create_dp_allreduce_backend(
    backend_name: str,
    dp_rank: int,
    dp_size: int,
    cpu_group: dist.ProcessGroup,
    device_group: dist.ProcessGroup,
    cuda_device: torch.device,
) -> DPAllReduceBackend:
    """Factory function to create the configured DP all_reduce backend."""
    if backend_name == "gloo":
        return GlooDPAllReduce(cpu_group)
    elif backend_name == "nccl-side-stream":
        logger.info(
            "Using NCCL side-stream DP all_reduce (rank %d)", dp_rank
        )
        return NCCLSideStreamDPAllReduce(device_group, dp_size, cuda_device)
    elif backend_name == "ucx":
        return UCXDPAllReduce(dp_rank, dp_size, cpu_group)
    else:
        raise ValueError(
            f"Unknown dp_cpu_backend: {backend_name!r}. "
            f"Options: gloo, nccl-side-stream, ucx"
        )
