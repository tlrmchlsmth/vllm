# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable

from vllm.sizing.report import Strategy


def _divisors(value: int) -> list[int]:
    return [i for i in range(1, value + 1) if value % i == 0]


def parse_strategy_spec(spec: str) -> list[Strategy]:
    values: dict[str, list[int]] = {
        "dp": [1],
        "tp": [1],
        "ep": [1],
        "pp": [1],
        "dcp": [1],
    }
    matches = list(re.finditer(r"(dp|tp|ep|pp|dcp)=([0-9,]+)", spec))
    if not matches:
        raise ValueError(
            "Strategy spec must use keys dp/tp/ep/pp/dcp, e.g. "
            "'dp=1,2 tp=1,4 dcp=1'."
        )
    remainder = spec
    for match in matches:
        remainder = remainder.replace(match.group(0), "", 1)
    if remainder.replace(",", "").strip():
        raise ValueError(
            "Strategy spec must use keys dp/tp/ep/pp/dcp, e.g. "
            "'dp=1,2 tp=1,4 dcp=1'."
        )

    for match in matches:
        key = match.group(1)
        raw_values = match.group(2)
        if key not in values:
            raise ValueError(
                "Strategy spec must use keys dp/tp/ep/pp/dcp, e.g. "
                "'dp=1,2 tp=1,4 dcp=1'."
            )
        values[key] = [int(x) for x in raw_values.split(",") if x]

    return [
        Strategy(dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp)
        for dp in values["dp"]
        for tp in values["tp"]
        for ep in values["ep"]
        for pp in values["pp"]
        for dcp in values["dcp"]
    ]


def default_strategies(max_gpus: int, *, is_moe: bool) -> list[Strategy]:
    rows: list[Strategy] = []
    for gpus in _divisors(max_gpus):
        for pp in _divisors(gpus):
            remaining = gpus // pp
            for tp in _divisors(remaining):
                dp = remaining // tp
                for dcp in _divisors(tp):
                    if is_moe:
                        rows.append(
                            Strategy(
                                dp=dp,
                                tp=tp,
                                ep=dp * tp,
                                pp=pp,
                                dcp=dcp,
                            )
                        )
                    else:
                        rows.append(Strategy(dp=dp, tp=tp, pp=pp, dcp=dcp))
    return _dedupe(rows)


def _dedupe(strategies: Iterable[Strategy]) -> list[Strategy]:
    seen: set[tuple[int, int, int, int, int]] = set()
    result: list[Strategy] = []
    for strategy in strategies:
        key = (
            strategy.dp,
            strategy.tp,
            strategy.ep,
            strategy.pp,
            strategy.dcp,
        )
        if key not in seen:
            seen.add(key)
            result.append(strategy)
    return result
