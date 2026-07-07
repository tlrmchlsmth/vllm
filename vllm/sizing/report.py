# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass, field

from vllm.sizing.gpu import format_bytes


@dataclass(frozen=True)
class Strategy:
    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    dcp: int = 1

    @property
    def gpus(self) -> int:
        return self.dp * self.tp * self.pp


@dataclass
class WeightRecord:
    name: str
    param_name: str
    loaded_shape: tuple[int, ...]
    loaded_dtype: str
    local_shape: tuple[int, ...]
    local_dtype: str
    local_bytes: int
    note: str | None = None


@dataclass
class StrategyReport:
    strategy: Strategy
    weight_bytes: int
    activation_min_bytes: int
    activation_max_bytes: int
    kv_cache_bytes: int
    kv_cache_tokens: int
    max_concurrency: float
    total_min_bytes: int
    total_max_bytes: int
    fits: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class SizingReport:
    model: str
    gpu_type: str | None
    gpu_memory_bytes: int
    rows: list[StrategyReport]
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_table(self) -> str:
        include_row_notes = any(row.notes for row in self.rows)
        headers = [
            "dp",
            "tp",
            "ep",
            "pp",
            "dcp",
            "gpus",
            "weights",
            "activations",
            "kv cache",
            "total",
            "kv tokens",
            "max conc",
            "fits",
        ]
        if include_row_notes:
            headers.append("notes")
        body: list[list[str]] = []
        for row in self.rows:
            strategy = row.strategy
            values = [
                str(strategy.dp),
                str(strategy.tp),
                str(strategy.ep),
                str(strategy.pp),
                str(strategy.dcp),
                str(strategy.gpus),
                format_bytes(row.weight_bytes),
                _format_range(row.activation_min_bytes, row.activation_max_bytes),
                format_bytes(row.kv_cache_bytes),
                _format_range(row.total_min_bytes, row.total_max_bytes),
                f"{row.kv_cache_tokens:,}",
                f"{row.max_concurrency:.2f}x",
                "yes" if row.fits else "no",
            ]
            if include_row_notes:
                values.append("; ".join(row.notes))
            body.append(values)
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(h)
            for i, h in enumerate(headers)
        ]
        lines = [
            "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
            "  ".join("-" * width for width in widths),
        ]
        lines.extend(
            "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
            for row in body
        )
        if self.notes:
            lines.append("")
            lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


def _format_range(min_bytes: int, max_bytes: int) -> str:
    if min_bytes == max_bytes:
        return format_bytes(min_bytes)
    return f"{format_bytes(min_bytes)} - {format_bytes(max_bytes)}"
