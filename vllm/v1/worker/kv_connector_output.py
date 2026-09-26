# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.outputs import KVConnectorOutput


class KVConnectorOutputBuffer:
    """Hold consumed connector notifications until the engine retrieves them."""

    def __init__(self) -> None:
        self._outputs: dict[int, KVConnectorOutput] = {}
        self._collection_failed = False
        self._lock = Lock()

    def put(
        self,
        step_id: int,
        output: "KVConnectorOutput",
        *,
        collection_failed: bool = False,
    ) -> None:
        with self._lock:
            assert step_id not in self._outputs
            self._outputs[step_id] = output
            self._collection_failed |= collection_failed

    def take(self, step_id: int | None) -> dict[int, "KVConnectorOutput"]:
        with self._lock:
            if self._collection_failed:
                raise RuntimeError(
                    "KV notification collection failed; restart required"
                )
            if step_id is None:
                outputs, self._outputs = self._outputs, {}
                return outputs
            output = self._outputs.pop(step_id, None)
            return {} if output is None else {step_id: output}


# Preserve consumed notifications: the buffer outlives a failed model output.
kv_connector_output_buffer = KVConnectorOutputBuffer()
