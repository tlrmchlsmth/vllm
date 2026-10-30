# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import numpy as np
import pytest
import torch
import vllm._C  # noqa: F401

from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.logits_processors import (
    CPUBadWords,
    CPUPenalties,
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.cpu.states import CPURequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

VOCAB_SIZE = 128
PROMPT = [5, 6, 5]
COMMITTED_OUTPUT = [7, 7, 8]


def _make_req_states(prompt: list[int] = PROMPT) -> tuple[CPURequestState, int]:
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    req_states.add_request(
        req_id="req",
        prompt_len=len(prompt),
        all_token_ids=prompt + COMMITTED_OUTPUT,
        num_computed_tokens=len(prompt) + len(COMMITTED_OUTPUT),
        max_tokens=32,
    )
    return req_states, req_states.req_id_to_index["req"]


def _context(req_idx: int, input_id: int = COMMITTED_OUTPUT[-1]) -> LogitsContext:
    return LogitsContext(
        expanded_idx_mapping=torch.tensor([req_idx], dtype=torch.int64),
        idx_mapping=torch.tensor([req_idx], dtype=torch.int64),
        idx_mapping_np=np.array([req_idx], dtype=np.intp),
        expanded_local_pos=torch.zeros(1, dtype=torch.int32),
        input_ids=torch.tensor([input_id], dtype=torch.int32),
        pos=torch.zeros(1, dtype=torch.int32),
        seq_lens_upper_bound_np=np.ones(1, dtype=np.int64),
    )


def _reference_penalties(
    logits: torch.Tensor,
    prompt: list[int],
    output: list[int],
    repetition: float,
    frequency: float,
    presence: float,
) -> None:
    for token_id in set(prompt + output):
        value = logits[token_id]
        logits[token_id] = value / repetition if value > 0 else value * repetition
    for token_id, count in Counter(output).items():
        logits[token_id] -= frequency * count + presence


@pytest.mark.parametrize("placeholder", [None, VOCAB_SIZE, VOCAB_SIZE + 3, -1])
@pytest.mark.parametrize(
    ("repetition", "frequency", "presence"),
    [(1.7, 0.4, -0.25), (1.0, 0.4, 0.0), (1.0, 0.0, -0.25)],
)
def test_cpu_penalties_match_reference(
    placeholder: int | None, repetition: float, frequency: float, presence: float
) -> None:
    """Ignore multimodal placeholders while penalizing real vocabulary tokens."""
    prompt = PROMPT + ([] if placeholder is None else [placeholder])
    req_states, req_idx = _make_req_states(prompt)
    state = CPUPenalties(None, LogitsProcRequestState.from_request_state(req_states))
    params = SamplingParams(
        repetition_penalty=repetition,
        frequency_penalty=frequency,
        presence_penalty=presence,
    )
    state.add_request(req_idx, params)

    logits = torch.linspace(-2, 2, VOCAB_SIZE).unsqueeze(0)
    logits[:, 5] = 2.0
    logits[:, 6] = -2.0
    logits[:, 7] = 3.0
    logits[:, 8] = -3.0
    expected = logits.clone()
    _reference_penalties(
        expected[0],
        PROMPT,
        COMMITTED_OUTPUT,
        params.repetition_penalty,
        params.frequency_penalty,
        params.presence_penalty,
    )

    actual = state.apply(logits, _context(req_idx))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("placeholder_request", [None, "first", "second"])
def test_cpu_penalties_keep_request_rows_independent(
    placeholder_request: str | None,
) -> None:
    """Ignore multimodal placeholders before they can address another row."""
    req_states = CPURequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        vocab_size=VOCAB_SIZE,
        device=torch.device("cpu"),
    )
    requests = {
        "first": ([1, 2], [3, 3]),
        "second": ([4], [5]),
    }
    for req_id, (prompt, output) in requests.items():
        if req_id == placeholder_request:
            prompt = prompt + [VOCAB_SIZE]
        tokens = prompt + output
        req_states.add_request(
            req_id=req_id,
            prompt_len=len(prompt),
            all_token_ids=tokens,
            num_computed_tokens=len(tokens),
            max_tokens=32,
        )
    first_idx = req_states.req_id_to_index["first"]
    second_idx = req_states.req_id_to_index["second"]
    state = CPUPenalties(None, LogitsProcRequestState.from_request_state(req_states))
    first_params = SamplingParams(
        repetition_penalty=1.5,
        frequency_penalty=0.25,
        presence_penalty=0.1,
    )
    second_params = SamplingParams(
        repetition_penalty=1.2,
        frequency_penalty=-0.5,
        presence_penalty=0.3,
    )
    state.add_request(first_idx, first_params)
    state.add_request(second_idx, second_params)

    # Reverse request order so row number and persistent request slot differ.
    row_mapping = torch.tensor([second_idx, first_idx], dtype=torch.int64)
    ctx = LogitsContext(
        expanded_idx_mapping=row_mapping,
        idx_mapping=row_mapping,
        idx_mapping_np=row_mapping.numpy(),
        expanded_local_pos=torch.zeros(2, dtype=torch.int32),
        input_ids=torch.tensor([5, 3], dtype=torch.int32),
        pos=torch.zeros(2, dtype=torch.int32),
        seq_lens_upper_bound_np=np.array([2, 4], dtype=np.int64),
    )
    logits = torch.linspace(-2, 2, VOCAB_SIZE).repeat(2, 1)
    expected = logits.clone()
    _reference_penalties(
        expected[0],
        *requests["second"],
        second_params.repetition_penalty,
        second_params.frequency_penalty,
        second_params.presence_penalty,
    )
    _reference_penalties(
        expected[1],
        *requests["first"],
        first_params.repetition_penalty,
        first_params.frequency_penalty,
        first_params.presence_penalty,
    )

    actual = state.apply(logits, ctx)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu_model
def test_cpu_penalties_ignore_multimodal_placeholders_generate(
    vllm_runner, monkeypatch
) -> None:
    """Ignore multimodal placeholders on the real LLaVA generation path."""
    from vllm.assets.image import ImageAsset

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    prompt = {
        "prompt": "USER: <image>\nDescribe the image.\nASSISTANT:",
        "multi_modal_data": {"image": ImageAsset("cherry_blossom").pil_image},
    }
    with vllm_runner(
        "llava-hf/llava-1.5-7b-hf",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        max_num_seqs=2,
        enable_prefix_caching=False,
    ) as runner:
        assert runner.llm.llm_engine.vllm_config.use_v2_model_runner
        params = SamplingParams(temperature=0, max_tokens=1, logprobs=1)
        expected = runner.llm.generate([prompt, prompt], params)
        params.frequency_penalty = 1.0
        actual = runner.llm.generate([prompt, prompt], params)

        vocab_size = runner.llm.llm_engine.vllm_config.model_config.get_vocab_size()
        for reference, result in zip(expected, actual):
            assert vocab_size in result.prompt_token_ids
            # With no generated history, frequency penalties must be a no-op.
            assert result.outputs[0].token_ids == reference.outputs[0].token_ids
            assert result.outputs[0].logprobs == reference.outputs[0].logprobs


def test_cpu_penalties_clear_reused_slot() -> None:
    req_states, req_idx = _make_req_states()
    state = CPUPenalties(None, LogitsProcRequestState.from_request_state(req_states))
    state.add_request(req_idx, SamplingParams(repetition_penalty=1.2))
    state.add_request(req_idx, SamplingParams())

    logits = torch.randn(1, VOCAB_SIZE)
    expected = logits.clone()
    state.apply(logits, _context(req_idx))
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize(
    ("bad_words", "masked"),
    [
        ([[7, 8, 60]], {60}),
        ([[8, 8, 50]], set()),
        ([[70]], {70}),
    ],
)
def test_cpu_bad_words_match_committed_output_prefix(
    bad_words: list[list[int]], masked: set[int]
) -> None:
    req_states, req_idx = _make_req_states()
    state = CPUBadWords(None, LogitsProcRequestState.from_request_state(req_states))
    state.add_request(req_idx, SamplingParams(_bad_words_token_ids=bad_words))

    actual = state.apply(torch.zeros((1, VOCAB_SIZE)), _context(req_idx))
    expected = torch.zeros_like(actual)
    for token_id in masked:
        expected[0, token_id] = -torch.inf
    torch.testing.assert_close(actual, expected)


def test_cpu_bad_words_clear_reused_slot() -> None:
    req_states, req_idx = _make_req_states()
    state = CPUBadWords(None, LogitsProcRequestState.from_request_state(req_states))
    state.add_request(req_idx, SamplingParams(_bad_words_token_ids=[[7, 8, 60]]))
    state.add_request(req_idx, SamplingParams())

    logits = torch.zeros((1, VOCAB_SIZE))
    state.apply(logits, _context(req_idx))
    assert not torch.isinf(logits).any()
