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

    Uses UCX endpoints in a ring topology. Since each rank only populates
    its own column, the sum all_reduce is an allgather. Ring allgather
    in N-1 steps of 16 bytes each completes in microseconds over RDMA.

    Requires ucx-py: pip install ucx-py-cu12 (or ucx-py-cu11)
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        cpu_group: dist.ProcessGroup,
    ):
        try:
            import ucp  # noqa: F401
        except ImportError:
            raise ImportError(
                "UCX DP backend requires ucx-py. "
                "Install with: pip install ucx-py-cu12"
            ) from None

        import asyncio
        import threading

        self.rank = rank
        self.world_size = world_size

        # Dedicated event loop for UCX async operations
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="ucx-dp-loop"
        )
        self._thread.start()

        # Initialize ring topology
        self._right_ep = None
        self._left_ep = None
        self._chunk_bytes = 4 * 4  # 4 int32s per column = 16 bytes
        self._init_ring(cpu_group)

        logger.info("UCX DP all_reduce initialized: rank %d/%d", rank, world_size)

    def _run_coro(self, coro, timeout=30):
        """Run async coroutine on the background loop, block until done."""
        import asyncio

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def _init_ring(self, cpu_group: dist.ProcessGroup):
        """Set up UCX ring: connect to right neighbor, accept from left.
        Uses gloo for one-time address exchange (fine over TCP)."""
        import socket as socket_mod
        import threading

        import ucp

        right_rank = (self.rank + 1) % self.world_size
        left_rank = (self.rank - 1) % self.world_size

        # Create listener on the background loop
        accepted_event = threading.Event()
        accepted_ep_holder = [None]

        async def _start_listener():
            def _on_accept(ep):
                accepted_ep_holder[0] = ep
                accepted_event.set()

            lsnr = ucp.create_listener(_on_accept)
            return lsnr

        listener = self._run_coro(_start_listener())

        # Exchange (hostname, port) via gloo all_reduce (one-time)
        hostname = socket_mod.gethostname()
        hostname_bytes = hostname.encode()[:60].ljust(60, b"\0")
        # Pack: 60 bytes hostname + 4 bytes port (big endian)
        import struct

        my_info = hostname_bytes + struct.pack(">I", listener.port)
        assert len(my_info) == 64

        # Use a 2D tensor: (world_size, 16) of int32 = 64 bytes per rank
        info_tensor = torch.zeros(
            self.world_size, 16, dtype=torch.int32, device="cpu"
        )
        my_ints = struct.unpack(f"16i", my_info)
        for i, v in enumerate(my_ints):
            info_tensor[self.rank, i] = v
        dist.all_reduce(info_tensor, group=cpu_group)

        # Decode right neighbor's address
        right_info = struct.pack(
            "16i", *[int(info_tensor[right_rank, i]) for i in range(16)]
        )
        right_host = right_info[:60].rstrip(b"\0").decode()
        right_port = struct.unpack(">I", right_info[60:64])[0]

        # Connect to right neighbor
        async def _connect():
            return await ucp.create_endpoint(right_host, right_port)

        self._right_ep = self._run_coro(_connect())

        # Wait for left neighbor to connect to us
        if not accepted_event.wait(timeout=30):
            raise TimeoutError(
                f"UCX ring init: rank {self.rank} timed out waiting "
                f"for connection from rank {left_rank}"
            )
        self._left_ep = accepted_ep_holder[0]

        logger.info(
            "UCX ring ready: rank %d → right=%d (%s:%d), left=%d → us",
            self.rank, right_rank, right_host, right_port, left_rank,
        )

    def all_reduce(self, tensor: torch.Tensor) -> None:
        self._run_coro(self._ring_allgather(tensor))

    async def _ring_allgather(self, tensor: torch.Tensor):
        """Ring allgather: N-1 steps, each step passes one column around."""
        import asyncio

        N = self.world_size
        nbytes = self._chunk_bytes
        # Work with raw bytes for UCX
        data = tensor.numpy()

        for step in range(N - 1):
            send_col = (self.rank - step) % N
            recv_col = (self.rank - step - 1) % N

            send_bytes = data[:, send_col].tobytes()
            recv_buf = bytearray(nbytes)

            await asyncio.gather(
                self._right_ep.send(send_bytes),
                self._left_ep.recv(recv_buf),
            )

            # Unpack received column into the tensor
            import struct

            vals = struct.unpack("4i", recv_buf)
            for row in range(4):
                data[row, recv_col] = vals[row]

    def close(self):
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=5)


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
