# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_events import BlockStored, KVEventAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheKVEvents,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test


class DummyWorkerMeta(KVConnectorWorkerMetadata):
    def __init__(self, tags: set[str]):
        self.tags = set(tags)

    def aggregate(self, other: "DummyWorkerMeta") -> "DummyWorkerMeta":
        return DummyWorkerMeta(self.tags | other.tags)


def make_events(*token_ids: int) -> KVEventAggregator:
    events = KVEventAggregator(num_workers=1)
    events.add_events(
        [
            BlockStored(
                block_hashes=[b"\xab" * 32],
                parent_block_hash=None,
                token_ids=list(token_ids),
                block_size=4,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            )
        ]
    )
    return events


class DummyModelRunnerOutput(ModelRunnerOutput):
    def __init__(
        self,
        finished_sending: set[str] | None = None,
        finished_recving: set[str] | None = None,
        invalid_block_ids: set[int] | None = None,
        failed_recving: set[str] | None = None,
        expected_finished_count: int = 0,
        kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None,
        kv_cache_events: KVEventAggregator | None = None,
    ):
        self.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
            failed_recving=failed_recving or set(),
            expected_finished_count=expected_finished_count,
            kv_connector_worker_meta=kv_connector_worker_meta,
            kv_cache_events=kv_cache_events,
        )

    def __repr__(self):
        return (
            f"DummyModelRunnerOutput("
            f"finished_sending={self.kv_connector_output.finished_sending},"
            f"finished_recving={self.kv_connector_output.finished_recving})"
            f"invalid_block_ids={self.kv_connector_output.invalid_block_ids})"
        )


def test_aggregate_workers_output():
    aggregator = KVOutputAggregator(expected_finished_count=2)

    output1 = DummyModelRunnerOutput()
    output2 = DummyModelRunnerOutput()

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert not aggregated.invalid_block_ids

    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, finished_recving={"req2"}
    )
    output2 = DummyModelRunnerOutput(invalid_block_ids={1})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {1}

    output1 = DummyModelRunnerOutput(invalid_block_ids={2})
    output2 = DummyModelRunnerOutput(finished_sending={"req1"})

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending == {"req1"}
    assert aggregated.finished_recving is None
    assert aggregated.invalid_block_ids == {2}

    output1 = DummyModelRunnerOutput(invalid_block_ids={3, 4})
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"},
        invalid_block_ids={4, 5},
        failed_recving={"req3"},
    )

    aggregated = aggregator.aggregate([output1, output2])

    assert aggregated is output1
    aggregated = aggregated.kv_connector_output
    assert aggregated.finished_sending is None
    assert aggregated.finished_recving == {"req2"}
    assert aggregated.invalid_block_ids == {3, 4, 5}
    assert not aggregated.failed_recving

    output1 = DummyModelRunnerOutput(finished_recving={"req3"})
    output2 = DummyModelRunnerOutput(finished_recving={"req3"})
    aggregated = aggregator.aggregate([output1, output2])
    assert aggregated.kv_connector_output.failed_recving == {"req3"}


def test_aggregate_workers_output_with_expected_finished_count():
    # We create the aggregator expecting to collect from 4 workers
    aggregator = KVOutputAggregator(expected_finished_count=4)
    assert aggregator._expected_finished_count == 4
    # Some request with default expected finished requests
    output1 = DummyModelRunnerOutput(finished_sending={"req1"})
    aggregated = aggregator.aggregate([output1])
    # still expecting to collect from 4 workers
    assert aggregator._send_remaining_count["req1"] == 3
    assert not aggregated.kv_connector_output.finished_sending
    assert not aggregated.kv_connector_output.finished_recving

    # Workers discover and find that in this setup they only need to
    # collect from 2
    output1 = DummyModelRunnerOutput(
        finished_sending={"req1"}, expected_finished_count=2
    )
    output2 = DummyModelRunnerOutput(
        finished_recving={"req2"}, expected_finished_count=2
    )
    output3 = DummyModelRunnerOutput(finished_recving={"req2"})
    # Req2 only needs 2 acks
    aggregated = aggregator.aggregate([output1, output2, output3])
    assert aggregated.kv_connector_output.expected_finished_count == 2

    assert not aggregated.kv_connector_output.finished_sending

    # Req2 is finished
    assert "req2" not in aggregator._recv_remaining_count
    assert aggregated.kv_connector_output.finished_recving == {"req2"}

    # Req1 is still waiting for 2 more acks (expected_finished_count has no effect)
    # NOTE: This is to showcase dynamic update. Workers are responsible for
    # ensuring "req1" termination in this case
    assert aggregator._send_remaining_count["req1"] == 2


def test_recovery_consumes_notifications_without_another_model_step():
    """Explicit KV recovery returns failures and completions exactly once."""
    aggregator = KVOutputAggregator(2, defer_outputs=True)
    output = aggregator.aggregate_kv_outputs(
        [
            KVConnectorOutput(finished_recving={"req"}, invalid_block_ids={7}),
            KVConnectorOutput(finished_recving={"req"}, failed_recving={"req"}),
        ]
    )
    assert output.finished_recving == {"req"}
    assert output.failed_recving == {"req"}
    assert output.invalid_block_ids == {7}
    assert aggregator.aggregate_kv_outputs([None, None]).is_empty()


def test_recovery_retains_partial_transfer_completions():
    """Preserve consumed notifications until every transfer worker finishes."""
    aggregator = KVOutputAggregator(2, defer_outputs=True)
    output = aggregator.aggregate_kv_outputs(
        [
            KVConnectorOutput(
                finished_sending={"req"},
                kv_connector_worker_meta=DummyWorkerMeta({"old"}),
            ),
            None,
        ]
    )
    assert output.finished_sending is None
    assert output.kv_connector_worker_meta.tags == {"old"}
    output = aggregator.aggregate_kv_outputs(
        [None, KVConnectorOutput(finished_sending={"req"})]
    )
    assert output.finished_sending == {"req"}
    assert output.kv_connector_worker_meta is None


def _lmcache_events(*token_ids: int) -> LMCacheKVEvents:
    events = LMCacheKVEvents(1)
    events.add_events(make_events(*token_ids).get_all_events())
    return events


def test_recovery_preserves_worker_quorum_across_sparse_steps():
    """Preserve worker quorum when only the missing rank reports after recovery."""
    aggregator = KVOutputAggregator(2, defer_outputs=True)
    first = KVConnectorOutput(kv_cache_events=_lmcache_events(1, 2))
    assert aggregator.aggregate_kv_outputs([first, None]).kv_cache_events is None
    # Repeated votes from one rank cannot satisfy the quorum.
    assert aggregator.aggregate_kv_outputs([first, None]).kv_cache_events is None
    second = KVConnectorOutput(kv_cache_events=_lmcache_events(1, 2))
    output = aggregator.aggregate_kv_outputs([None, second])
    assert output.kv_cache_events is not None
    common = output.kv_cache_events.aggregate().get_all_events()
    assert [event.token_ids for event in common] == [[1, 2]]
    assert aggregator.aggregate_kv_outputs([None, None]).kv_cache_events is None


def test_recovery_pending_event_does_not_block_new_event_quorum():
    """Preserve worker quorum separately for old and new cache events."""
    aggregator = KVOutputAggregator(2, defer_outputs=True)
    aggregator.aggregate_kv_outputs(
        [KVConnectorOutput(kv_cache_events=_lmcache_events(1)), None]
    )
    output = aggregator.aggregate_kv_outputs(
        [
            KVConnectorOutput(kv_cache_events=_lmcache_events(2)),
            KVConnectorOutput(kv_cache_events=_lmcache_events(2)),
        ]
    )
    assert [e.token_ids for e in output.kv_cache_events.get_all_events()] == [[2]]
    output = aggregator.aggregate_kv_outputs(
        [None, KVConnectorOutput(kv_cache_events=_lmcache_events(1))]
    )
    assert [e.token_ids for e in output.kv_cache_events.get_all_events()] == [[1]]
