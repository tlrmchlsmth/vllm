# SPDX-License-Identifier: Apache-2.0
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

import msgspec
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.sampling_params import KVTransferParams
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: list[bytes]
    # Base addr for each rank, each layer for KVs
    kv_caches_base_addr: list[list[tuple[int, int]]]
    num_blocks: int


class ReqMeta:

    def __init__(
        self,
        block_ids: list[int],
        remote_block_ids: list[int],
        remote_engine_id: list[int],
    ):
        self.block_ids = block_ids
        self.remote_block_ids = remote_block_ids
        self.remote_engine_id = remote_engine_id


class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        req_id: str,
        block_ids: list[int],
        kv_transfer_params: KVTransferParams,
    ):
        assert req_id not in self.requests
        self.requests[req_id] = ReqMeta(
            block_ids,
            remote_block_ids=kv_transfer_params.remote_block_ids,
            remote_engine_id=kv_transfer_params.remote_engine_id)


class NixlConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        self.engine_id = uuid.uuid4()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorScheduler(
                vllm_config, self.engine_id)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(self.engine_id)

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> int:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: torch.Tensor):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """NixlConnector does not save explicitly."""
        return

    def wait_for_save(self):
        """NixlConnector does not save explicitly."""
        return


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id = engine_id

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, Request] = {}

    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> int:
        """For remote prefill, allocate for all tokens."""
        if request.do_remote_prefill:
            return len(request.prompt_token_ids) - num_computed_tokens

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        if request.do_remote_decode:
            pass
        if request.do_remote_prefill and num_external_tokens > 0:
            self._reqs_need_recv[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for new_req in scheduler_output.scheduled_new_reqs:
            req = self._reqs_need_recv.pop(new_req.req_id, None)
            if req is not None:
                meta.add_new_req(
                    request_id=new_req.req_id,
                    local_block_ids=new_req.block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )

        # Invariant: only new requests should need load
        # and we should get all new requests each step().
        assert len(self._reqs_need_recv) == 0
        return meta


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")

        # Agent.
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)
        # Map of engine_id -> list[agent_names] (1 per rank).
        self._remote_agents: dict[str, list[str]] = {}

        # Metadata.
        self.engine_id = engine_id
        self.rank = 0

        # KV Caches and nixl tracking data.
        self.num_layers: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.kv_caches: tuple[torch.Tensor, torch.Tensor] = None

        # Map of engine_id -> kv_caches_base_addr
        # For Local: base addr for *this* rank, each layer for K,V
        # For Remote: base addr for *each* rank, each layer for K,V
        # KV_CACHES_ADDR_TYPE = Union[list[tuple[int, int]],
        #                             list[list[tuple[int, int]]]]
        self.kv_caches_base_addr: dict[str, any] = {}

        # Map of tp_mult -> nixl_prepped_dlist_handle (int).
        self.src_xfer_side_handles: dict[int, int] = {}
        # Map of engine_id -> map[tp_mult -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: defaultdict[str,
                                                dict[int,
                                                     int]] = defaultdict(dict)
        # Map of engine_id -> num_blocks.
        self.dst_num_blocks: dict[str, int] = {}
        self._registered_descs: list[any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_transfers = defaultdict(list[any])

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        first_layer_name = next(iter(kv_caches))
        first_kv_cache = kv_caches[first_layer_name]

        # [2 (k and v), num_blocks, ...]
        _, num_blocks, block_size, num_heads, head_dim = first_kv_cache.shape
        self.block_len = block_size * num_heads * head_dim * first_kv_cache[
            0].element_size()
        logger.debug("Per layer kv cache size: %s", first_kv_cache[0].shape)
        self.num_layers = len(kv_caches)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []
        for layer_name in kv_caches:
            kv_cache = kv_caches[layer_name]
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            base_addr = key_cache.data_ptr()
            region_len = 2 * num_blocks * self.block_len
            caches_data.append((base_addr, region_len, self.rank, ""))
            kv_caches_base_addr.append(
                (key_cache.data_ptr(), value_cache.data_ptr()))

        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr

        descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        self._registered_descs.append(descs)

        # THIS IS FOR DEBUG and INSECURE
        import os
        _ctx = zmq.Context()  # type: ignore
        _side_channel = _ctx.socket(zmq.PAIR)  # type: ignore
        NIXL_ROLE = os.getenv("NIXL_ROLE")
        if NIXL_ROLE == "SENDER":
            _side_channel.bind("tcp://localhost:5555")
            _side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
            metadata = NixlAgentMetadata(
                self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                kv_caches_base_addr=self.v_)
            encoder = msgspec.msgpack.Encoder()
            _side_channel.send(encoder.encode(metadata))

        elif NIXL_ROLE == "RECVER":
            _side_channel.bind("tcp://localhost:5555")
            _side_channel.setsockopt(zmq.LINGER, 0)  # type: ignore
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata_bytes = _side_channel.recv()
            metadata = decoder.decode(metadata_bytes)
            self.add_remote_agent(metadata)

        else:
            raise Exception("SET NIXL_ROLE to SENDER OR RECVER")

    def add_remote_agent(self, nixl_agent_meta: NixlAgentMetadata):
        engine_id = nixl_agent_meta.engine_id
        if engine_id in self._remote_agents:
            return

        num_blocks = nixl_agent_meta.num_blocks

        agent_names = []
        for agent_meta in nixl_agent_meta.agent_metadata:
            agent_name = self.nixl_wrapper.add_remote_agent(agent_meta)
            agent_names.append(agent_name)

        self._remote_agents[engine_id] = agent_names
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr

        # NOTE: once we support heterogeneous TP, we will need maintain the
        # src for each TP multiplier.
        # NOTE(rob): Dynamo only supports D TP size > P TP size.
        # https://github.com/vllm-project/vllm/pull/16124/files#diff-876efa5533f5dcff3fba850e8684a47d53c112e287988957c115b11691374f4bR331 # noqa: E501
        # Create descs and xfer side handles.
        tp_multiplier = 1
        dst_block_len = self.block_len // tp_multiplier
        if tp_multiplier not in self.src_xfer_side_handles:
            # Create descs and xfer side handles.
            blocks_data = []
            for layer_id in range(self.num_layers):
                # Both K and V.
                for base_addr in self.kv_caches_base_addr[
                        self.engine_id][layer_id]:
                    for block_id in range(self.num_blocks):
                        block_offset = block_id * self.block_len
                        for i in range(tp_multiplier):
                            tp_multiplier_offset = i * dst_block_len
                            blocks_data.append((base_addr + block_offset +
                                                tp_multiplier_offset,
                                                dst_block_len, self.rank))
            logger.debug("Created %s blocks for src engine %s and rank %s",
                         len(blocks_data), self.engine_id,
                         self.rank * tp_multiplier + i)

            # Register with NIXL.
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.src_xfer_side_handles[tp_multiplier] = (
                self.nixl_wrapper.prep_xfer_dlist("", descs))

        # create dst xfer side handles
        self.dst_num_blocks[engine_id] = num_blocks
        for i in range(tp_multiplier):
            blocks_data = []
            for layer_id in range(self.num_layers):
                for base_addr in self.kv_caches_base_addr[engine_id][
                        self.rank * tp_multiplier + i][layer_id]:
                    for block_id in range(num_blocks):
                        block_offset = block_id * dst_block_len
                        blocks_data.append(
                            (base_addr + block_offset, dst_block_len,
                             self.rank * tp_multiplier + i))
            logger.debug("Created %s blocks for dst engine %s and rank %s",
                         len(blocks_data), engine_id,
                         self.rank * tp_multiplier + i)
            # Register with NIXL.
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
            self.dst_xfer_side_handles[engine_id][i] = (
                self.nixl_wrapper.prep_xfer_dlist(
                    self._remote_agents[engine_id][self.rank * tp_multiplier +
                                                   i], descs))

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Get requests that are done sending or recving."""
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer message."""

        notified_req_ids: set[str] = set()
        # TODO: handle the TP case (N notifies for TP=N).
        # See: vllm/worker/worker_base.py L476 in DynamoPR.
        for req_ids in self.nixl_wrapper.get_new_notifs().values():
            for req_id in req_ids:
                assert req_id not in notified_req_ids
                notified_req_ids.add(req_id)
        return notified_req_ids

    def _pop_done_transfers(self, transfers: dict[str, list[str]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: str[str] = set()
        for req_id, handles in transfers.items():
            running_reqs = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # TODO ptarasiewicz: why abort is throwing errors?
                    # self.nixl_wrapper.release_xfer_handle(handle)
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if len(running_reqs) == 0:
                done_req_ids.add(req_id)
            else:
                transfers[req_id] = running_reqs
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.requests.items():
            # NOTE: this is non-blocking
            self._read_blocks(
                local_block_ids=meta.block_ids,
                # TODO: support staging once we do heterogeneous TP
                staging_block_ids=meta.block_ids,
                remote_block_ids=meta.remote_block_ids,
                dst_engine_id=meta.remote_engine_id,
                request_id=req_id,
            )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        staging_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
    ):
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # NOTE(rob): we could potentially do the rearranging during the load_kv!

        assert len(local_block_ids) == len(staging_block_ids) == len(
            remote_block_ids)
        if len(local_block_ids) == 0:
            return

        # TODO(rob): understand ranges code.
        local_ranges = self._get_ranges(local_block_ids)
        staging_ranges = self._get_ranges(staging_block_ids)
        _, staging_rearranging_ranges = self._get_same_length_ranges(
            local_ranges, staging_ranges)

        # TODO: support TP multipliers.
        tp_multiplier = 1
        remote_block_descs_ids = self._get_block_descs_ids(
            dst_engine_id, "all", remote_block_ids)
        local_xfer_side_handle = self.src_xfer_side_handles[tp_multiplier]

        # Read the data from the remote.
        for i in range(tp_multiplier):
            staging_block_descs_ids = self._get_block_descs_ids(
                self.engine_id,
                "all",
                staging_block_ids,
                i=i,
                tp_multiplier=tp_multiplier,
                staging_ranges=staging_rearranging_ranges)
            assert len(staging_block_descs_ids) == len(remote_block_descs_ids)
            remote_xfer_side_handle = self.dst_xfer_side_handles[
                dst_engine_id][i]

            # NOTE(rob): we use the request_id as the notify msg, so we
            # must use the same request_id in both the p and d workers.
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                staging_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=request_id,
            )
            # NOTE(rob): we will check this is done in the next forward pass.
            self._recving_transfers[request_id].append(handle)

        # NOTE(rob): this is actually pretty serious problem.
        # We need to figure out if we can put the staging blocks on the P worker side. # noqa: E501
        # The staging blocks need to be on the side that sends.

        # for local_range, staging_range in zip(local_rearranging_ranges, staging_rearranging_ranges): # noqa: E501
        #     logger.debug("Rearranging tensors for cache: %s, local_range: %s, staging_range: %s", # noqa: E501
        #                  self.kv_caches[0].shape, local_range, staging_range)
        #     for kv_cache in self.kv_caches:
        #         for cache in kv_cache:
        #             rearrange_tensors(cache[local_range[0]:local_range[1] + 1], # noqa: E501
        #                               cache[staging_range[0]:staging_range[1] + 1], # noqa: E501
        #                               tp_multiplier, "read")

    def _get_ranges(self, block_ids):
        # This function should return a list of ranges of block ids that are contiguous # noqa: E501
        # For example, if block_ids is [0, 1, 2, 4, 5, 6], the function should return [[0, 2], [4, 6]] # noqa: E501
        # The ranges are sorted by the starting block id
        # The function should also make sure that the block ids are contiguous
        # If the block ids are not contiguous, the function should raise an error # noqa: E501
        ranges = []
        for i in range(len(block_ids)):
            if i == 0 or block_ids[i] != block_ids[i - 1] + 1:
                ranges.append([block_ids[i], block_ids[i]])
            else:
                ranges[-1][1] = block_ids[i]
        return ranges

    def _get_block_descs_ids(self,
                             engine_id,
                             layer_ids,
                             block_ids,
                             i=None,
                             tp_multiplier=1,
                             staging_ranges=None):

        if layer_ids == "all":
            layer_ids = list(range(self.num_layers))
        if block_ids == "all":
            block_ids = list(range(self.num_blocks))

        descs_ids = []

        if i is not None:
            num_blocks = self.num_blocks
            for layer_id in layer_ids:
                for is_value in [0, 1]:
                    staging_range_idx = 0
                    for block_id in block_ids:
                        if block_id > staging_ranges[staging_range_idx][
                                1] or block_id < staging_ranges[
                                    staging_range_idx][0]:
                            staging_range_idx += 1
                        start_offset = staging_ranges[staging_range_idx][0]
                        i_offset = i * (staging_ranges[staging_range_idx][-1] -
                                        start_offset + 1)
                        descs_ids.append(
                            layer_id * 2 * num_blocks * tp_multiplier +
                            is_value * num_blocks * tp_multiplier +
                            start_offset * tp_multiplier + i_offset +
                            (block_id - start_offset))
        else:
            num_blocks = self.dst_num_blocks[engine_id]
            for layer_id in layer_ids:
                for is_value in [0, 1]:
                    for block_id in block_ids:
                        descs_ids.append(layer_id * 2 * num_blocks +
                                         is_value * num_blocks + block_id)
        return descs_ids
