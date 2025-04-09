# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import zmq
from lmcache.experimental.cache_engine import LMCacheEngine
from lmcache.integration.vllm.vllm_adapter import init_lmcache_engine

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.utils import make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)


# FIXME: Use a different way to generate the rpc path in order
# to avoid cross-vllm instance conflict
def get_zmq_rpc_path_lmcache(kv_role: str):
    base_url = envs.VLLM_RPC_BASE_PATH
    logger.debug("Base URL: %s", base_url)
    return f"ipc://{base_url}/lmcache_rpc_port_{kv_role}"


# TODO: move this to LMCache so that we can gracefully close it
class LMCacheLookupClient:

    def __init__(self, kv_role: str):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        socket_path = get_zmq_rpc_path_lmcache(kv_role)
        self.socket = make_zmq_socket(self.ctx,
                                      socket_path,
                                      zmq.REQ,
                                      bind=False)

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send(request)

        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:

    def __init__(self, lmcache_engine: LMCacheEngine, kv_role: str):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()
        socket_path = get_zmq_rpc_path_lmcache(kv_role)
        self.socket = make_zmq_socket(self.ctx,
                                      socket_path,
                                      zmq.REP,
                                      bind=True)

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                try:
                    request = self.socket.recv()
                    token_ids = self.decoder.decode(request)
                    result = self.lmcache_engine.lookup(token_ids)
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                except Exception as e:
                    logger.error("Error in LMCache lookup server: %s", e)
                    break
                #continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!


@dataclass
class LoadSpec:
    vllm_cached_tokens: int
    lmcache_cached_tokens: int


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request(request: "Request",
                     block_size: int,
                     load_spec: Optional[LoadSpec] = None) -> "ReqMeta":
        # NOTE: be careful! The scheduler may not schedule all of the tokens
        # in the request. And the allocated blocks could also be more than
        # the number of tokens in the request.
        # Therefore, we need to use min(tokens_in_request, tokens_in_blocks)
        # to determine the number of tokens for connector to use.
        token_ids = torch.tensor(request.prompt_token_ids)
        num_blocks = len(request.block_ids)
        valid_num_tokens = min(len(token_ids), num_blocks * block_size)
        token_ids = token_ids[:valid_num_tokens]

        # Extract slot mapping from block ids
        block_ids = torch.Tensor(request.block_ids)
        num_blocks = block_ids.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens].long()
        if len(slot_mapping) != len(token_ids):
            logger.error("Slot mapping should be the same length as token ids")
            logger.error("Block ids: %s", str(block_ids))
            logger.error("Num blocks: %d", num_blocks)
            logger.error("Num tokens: %d", valid_num_tokens)
            logger.error("Slot mapping: %s", str(slot_mapping))
            logger.error("Slot mapping len: %s", len(slot_mapping))
        return ReqMeta(token_ids, slot_mapping, load_spec)


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self,
                    request: "Request",
                    block_size: int,
                    load_spec: Optional[LoadSpec] = None) -> None:
        self.requests.append(
            ReqMeta.from_request(request, block_size, load_spec))


class LMCacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, rank: Optional[int], local_rank: Optional[int],
                 config: "VllmConfig", role: KVConnectorRole):
        super().__init__(rank, local_rank, config, role)

        self.kv_role = config.kv_transfer_config.kv_role
        if role == KVConnectorRole.SCHEDULER:
            self.lookup_client = LMCacheLookupClient(self.kv_role)
        else:
            self.lmcache_engine = init_lmcache_engine(config.model_config,
                                                      config.parallel_config,
                                                      config.cache_config)
            self.lookup_server = LMCacheLookupServer(self.lmcache_engine,
                                                     self.kv_role)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = config.cache_config.block_size

        # request_id -> (vllm cached tokes, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it",
                             layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

    ####################
    # Worker side APIs
    ####################

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # HACK: getting chunk size to correctly calculate retrieve mask
        lmcache_chunk_size = self.lmcache_engine.config.chunk_size

        for request in metadata.requests:
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = request.load_spec.vllm_cached_tokens // \
                lmcache_chunk_size * lmcache_chunk_size
            token_mask[:masked_token_count] = False

            ret_token_mask = self.lmcache_engine.retrieve(
                tokens,
                token_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping)

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = request.load_spec.lmcache_cached_tokens - \
                    request.load_spec.vllm_cached_tokens
            if num_retrieved_tokens != num_expected_tokens:
                logger.warning(
                    "The number of retrieved tokens is not equal to the "
                    "expected number of tokens! This should not happen!")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name not in self.kv_caches:
            self.kv_caches[layer_name] = kv_layer

        ## The attention metadata should be the same for all layers
        #if self._attn_metadata_for_save is not None:
        #    assert self._attn_metadata_for_save == attn_metadata
        #else:
        #    self._attn_metadata_for_save = attn_metadata

    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        for request in connector_metadata.requests:
            token_ids = request.token_ids
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()

            skip_leading_tokens = self.lmcache_engine.lookup(token_ids)
            if skip_leading_tokens == len(token_ids):
                continue  # skip this request

            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            self.lmcache_engine.store(token_ids,
                                      mask=store_mask,
                                      kvcaches=kvcaches,
                                      slot_mapping=slot_mapping,
                                      offset=skip_leading_tokens)

    ###################
    # Scheduler side APIs
    ####################

    def get_external_prefix_cache_blocks(
        self,
        request: "Request",
        computed_blocks: list["KVCacheBlock"],
        num_computed_tokens: int,
        kv_cache_manager: "KVCacheManager",
    ) -> list["KVCacheBlock"]:
        """Get the external prefix cache blocks from the connector.

        This function may change the state of the connector, which will be 
        used by `attach_connector_meta` later.

        Args:
            request (Request): the request object.
            computed_blocks (list[KVCacheBlock]): the 'local' computed blocks.
            num_computed_tokens (int): the number of 'local' computed tokens.
            kv_cache_manager (KVCacheManager): the KV cache manager to 
                allocate/free the blocks if needed.

        Returns:
            The updated list of the computed blocks (appended with the remote
            cached blocks)
        """
        if self.kv_role == "kv_producer":
            # Don't do lookup if the role is kv_producer
            return computed_blocks

        token_ids = torch.tensor(request.prompt_token_ids)
        num_external_hit_tokens = self.lookup_client.lookup(token_ids)
        logger.info("Num external hit tokens: %d", num_external_hit_tokens)

        if num_external_hit_tokens <= num_computed_tokens:
            # No need to load the KV cache from external
            return computed_blocks

        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # HACK: pre-allocate the blocks as a "temp" request
        old_req_id = request.request_id
        request.request_id = "temp-req-id-for-connector"
        allocated_blocks = kv_cache_manager.allocate_slots(
            request, need_to_allocate, computed_blocks, skip_preallocate=True)
        if allocated_blocks is None:
            logger.error("Failed to allocate slots for the connector")
        request.request_id = old_req_id
        kv_cache_manager.req_to_blocks.pop("temp-req-id-for-connector")
        kv_cache_manager.num_cached_block.pop("temp-req-id-for-connector")
        kv_cache_manager.block_pool.free_blocks(allocated_blocks)

        # HACK: the scheduler should not see "all of the blocks" are
        # already allocated. Therefore, we need to back up one block
        num_expected_blocks = need_to_allocate // self._block_size
        if len(allocated_blocks) > num_expected_blocks:
            allocated_blocks = allocated_blocks[:num_expected_blocks]

        if (len(allocated_blocks) + len(computed_blocks)) \
                * self._block_size >= len(token_ids):
            # HACK: back-off one block
            allocated_blocks = allocated_blocks[:-1]

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens)

        return computed_blocks + allocated_blocks

    def attach_connector_meta(
            self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = LMCacheConnectorMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            meta.add_request(request, self._block_size, load_spec)
        scheduler_output.kv_connector_metadata = meta
        return scheduler_output
