# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re

GiB = 1024**3
GB = 1000**3


GPU_MEMORY_BYTES: dict[str, int] = {
    "A10": 24 * GiB,
    "A10G": 24 * GiB,
    "A40": 48 * GiB,
    "A100-40GB": 40 * GiB,
    "A100-80GB": 80 * GiB,
    "H100-80GB": 80 * GiB,
    "H200-141GB": 141 * GiB,
    "B100-192GB": 192 * GiB,
    "B200-180GB": 180 * GiB,
    "GB200-186GB": 186 * GiB,
    "L4": 24 * GiB,
    "L40S": 48 * GiB,
    "MI250-64GB": 64 * GiB,
    "MI300X-192GB": 192 * GiB,
    "MI325X-256GB": 256 * GiB,
}

_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)?\s*$", re.I)


def parse_memory(value: str) -> int:
    match = _MEMORY_RE.match(value)
    if match is None:
        raise ValueError(f"Invalid memory size: {value!r}")

    number = float(match.group(1))
    suffix = (match.group(2) or "gib").lower()
    factors = {
        "": GiB,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "kib": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mib": 1024**2,
        "g": GB,
        "gb": GB,
        "gib": GiB,
        "t": 1000**4,
        "tb": 1000**4,
        "tib": 1024**4,
    }
    if suffix not in factors:
        raise ValueError(f"Invalid memory suffix: {suffix!r}")
    return int(number * factors[suffix])


def format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or suffix == "TiB":
            return f"{value:.2f} {suffix}" if suffix != "B" else f"{value:.0f} B"
        value /= 1024.0
    raise AssertionError("unreachable")
