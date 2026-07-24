# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CoasterBench protocol test: a vLLM-served model plays an agentic
tool-calling game (designing RCT2 coasters against a headless OpenRCT2)
and every round must complete the tool-calling protocol.

The gate is protocol integrity — required/named tool_choice honored,
multi-turn tool_result round-trips, large JSON tool arguments accepted,
every round ends in a game-accepted submission. Coaster quality (did the
track circuit close, what excitement score) is model capability: reported
in the test output, never asserted.

Requires:
  COASTERBENCH_REPO  checkout of github.com/tlrmchlsmth/CoasterBench
  COASTERBENCH_CLI   openrct2-cli binary (from a build or the game image)

Usage:
pytest -s -v tests/evals/coasterbench/test_coasterbench_protocol.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import RemoteOpenAIServer

MODEL = os.environ.get("COASTERBENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ROUNDS = int(os.environ.get("COASTERBENCH_ROUNDS", "2"))
SERVER_ARGS = [
    "--max-model-len", "16384",
    "--enable-auto-tool-choice",
    "--tool-call-parser", os.environ.get("COASTERBENCH_TOOL_PARSER", "hermes"),
]
STARTUP_MAX_WAIT_SECONDS = 1200


def _require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        pytest.skip(f"{name} not set or missing (see tests/evals/coasterbench/README.md)")
    return Path(value)


def test_coasterbench_protocol():
    repo = _require_env("COASTERBENCH_REPO")
    _require_env("COASTERBENCH_CLI")

    with RemoteOpenAIServer(
        MODEL, SERVER_ARGS, max_wait_seconds=STARTUP_MAX_WAIT_SECONDS
    ) as server:
        run_name = "vllm-ci"
        proc = subprocess.run(
            [
                "uv", "run", str(repo / "evals" / "driver.py"),
                "--base-url", server.url_for("v1"),
                "--models", MODEL,
                "--rounds", str(ROUNDS),
                "--no-graphics",
                "--name", run_name,
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        assert proc.returncode == 0, "driver did not complete all rounds"

    run_dirs = sorted((repo / "evals" / "runs").glob(f"*-{run_name}"))
    assert run_dirs, "driver produced no run directory"
    run_dir = run_dirs[-1]

    standings = json.loads((run_dir / "standings.json").read_text())
    attempts = standings["standings"][0]["attempts"]
    assert len(attempts) == ROUNDS, f"expected {ROUNDS} completed rounds, got {len(attempts)}"

    # Every round must have produced a game-accepted submission: a report
    # exists and the game evaluated the program (ok or a placement verdict —
    # both mean the tool-calling loop delivered a well-formed program).
    model_dir = run_dir / MODEL.replace("/", "_")
    for rnd in range(1, ROUNDS + 1):
        report_path = model_dir / f"round_{rnd}" / "report.json"
        assert report_path.exists(), f"round {rnd}: no eval report"
        report = json.loads(report_path.read_text())
        assert report.get("program") is not None, f"round {rnd}: game never received a program"

    # Capability, reported not asserted.
    best = standings["standings"][0].get("best_excitement")
    print(f"\nCoasterBench: {MODEL}, {ROUNDS} rounds, best excitement: {best}")
