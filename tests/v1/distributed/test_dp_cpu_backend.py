# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E integration tests for --dp-cpu-backend option.

Launches external LB DP servers with different dp_cpu_backend values
and verifies correctness via completions and GSM8K evaluation.

Reuses ExternalLBServerManager from test_external_lb_dp.py.
"""

import os

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.v1.distributed.test_external_lb_dp import ExternalLBServerManager

MODEL_NAME = "ibm-research/PowerMoE-3b"

DP_SIZE = int(os.getenv("DP_SIZE", "2"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))

NUM_QUESTIONS = 50
NUM_SHOTS = 5
ACCURACY_THRESHOLD = 0.10
TOL = 0.08


@pytest.fixture
def base_server_args():
    return [
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "128",
        "--enforce-eager",
        "--disable-uvicorn-access-log",
    ]


def _run_gsm8k_on_server(server) -> dict:
    url = server.url_for("v1")
    if "://" in url:
        url = url.split("://")[1]
    host_port = url.split("/")[0]
    host, port = (host_port.rsplit(":", 1) + ["8000"])[:2]
    if not host.startswith("http"):
        host = f"http://{host}"
    return evaluate_gsm8k(
        num_questions=NUM_QUESTIONS,
        num_shots=NUM_SHOTS,
        host=host,
        port=int(port),
    )


@pytest.mark.parametrize("dp_cpu_backend", ["gloo", "nccl-side-stream"])
class TestDPCPUBackend:

    def test_completions(
        self, base_server_args, dp_cpu_backend, num_gpus_available
    ):
        if num_gpus_available < DP_SIZE * TP_SIZE:
            pytest.skip(
                f"Need {DP_SIZE * TP_SIZE} GPUs, have {num_gpus_available}"
            )

        server_args = base_server_args + [
            "--dp-cpu-backend", dp_cpu_backend,
        ]

        with ExternalLBServerManager(
            MODEL_NAME, DP_SIZE, 1, server_args, tp_size=TP_SIZE
        ) as servers:
            for server, _ in servers:
                client = server.get_client()
                for _ in range(5):
                    completion = client.completions.create(
                        model=MODEL_NAME,
                        prompt="What is 2 + 2?",
                        max_tokens=32,
                        temperature=0.0,
                    )
                    assert completion.choices[0].text.strip() != ""
                    assert completion.choices[0].finish_reason in (
                        "length", "stop"
                    )

    def test_gsm8k_correctness(
        self, base_server_args, dp_cpu_backend, num_gpus_available
    ):
        if num_gpus_available < DP_SIZE * TP_SIZE:
            pytest.skip(
                f"Need {DP_SIZE * TP_SIZE} GPUs, have {num_gpus_available}"
            )

        server_args = base_server_args + [
            "--dp-cpu-backend", dp_cpu_backend,
        ]

        with ExternalLBServerManager(
            MODEL_NAME, DP_SIZE, 1, server_args, tp_size=TP_SIZE
        ) as servers:
            server, _ = servers[0]
            results = _run_gsm8k_on_server(server)

            print(f"GSM8K with dp_cpu_backend={dp_cpu_backend}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Invalid rate: {results['invalid_rate']:.3f}")
            print(f"  Latency: {results['latency']:.1f}s")

            assert results["accuracy"] >= ACCURACY_THRESHOLD - TOL, (
                f"GSM8K accuracy too low with {dp_cpu_backend}: "
                f"{results['accuracy']:.4f} < "
                f"{ACCURACY_THRESHOLD - TOL:.4f}"
            )
