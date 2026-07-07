# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.sizing.gpu import GPU_MEMORY_BYTES, parse_memory
from vllm.sizing.runner import estimate_model_size

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class SizerSubcommand(CLISubcommand):
    """The `sizer` subcommand for the vLLM CLI."""

    name = "sizer"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.model_tag is not None:
            args.model = args.model_tag
        gpu_memory = _resolve_gpu_memory(args)
        _configure_sizer_platform(args.gpu_type)
        engine_args = EngineArgs.from_cli_args(args)
        report = estimate_model_size(
            engine_args,
            gpu_memory_bytes=gpu_memory,
            gpu_type=args.gpu_type,
            max_gpus=args.max_gpus,
            strategy_spec=args.strategy,
        )
        if args.output == "json":
            print(report.to_json())
        else:
            print(report.to_table())

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "sizer",
            help="Estimate decoder model deployment memory.",
            description="Estimate decoder model deployment memory.",
            usage="vllm sizer [model_tag] [options]",
        )
        parser.add_argument("model_tag", nargs="?", help="HF model tag or local path")
        EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--gpu-type",
            choices=sorted(GPU_MEMORY_BYTES),
            help="GPU type used to select the per-GPU memory budget.",
        )
        parser.add_argument(
            "--gpu-memory",
            help="Override per-GPU memory, e.g. 80GiB or 192GB.",
        )
        parser.add_argument(
            "--max-gpus",
            type=int,
            default=8,
            help="Maximum GPU count to include in the default strategy sweep.",
        )
        parser.add_argument(
            "--strategy",
            help="Explicit strategy sweep, e.g. 'dp=1,2 tp=1,4 dcp=1'.",
        )
        parser.add_argument(
            "--output",
            choices=("table", "json"),
            default="table",
            help="Output format.",
        )
        return parser


def _resolve_gpu_memory(args: argparse.Namespace) -> int:
    if args.gpu_memory:
        return parse_memory(args.gpu_memory)
    if args.gpu_type:
        return GPU_MEMORY_BYTES[args.gpu_type]
    raise ValueError("Either --gpu-type or --gpu-memory must be provided.")


def _configure_sizer_platform(gpu_type: str | None) -> None:
    from vllm.platforms import current_platform

    configure_target = getattr(current_platform, "configure_target", None)
    if configure_target is not None:
        configure_target(gpu_type)


def cmd_init() -> list[CLISubcommand]:
    return [SizerSubcommand()]
