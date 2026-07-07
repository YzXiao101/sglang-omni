# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible Ming-Omni-TTS server.

Example::

    python examples/run_ming_omni_tts_server.py \
        --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
        --ming-tts-tp-size 2 \
        --ming-tts-tp-gpus 0,1 \
        --reference-gpu-id 0 \
        --audio-decode-gpu-id 0 \
        --tts-engine-total-gpu-memory-fraction 0.72 \
        --reference-encode-total-gpu-memory-fraction 0.08 \
        --audio-decode-total-gpu-memory-fraction 0.12 \
        --no-disable-cuda-graph \
        --cuda-graph-bs 1,2,4,8 \
        --cuda-graph-max-bs 8 \
        --max-running-requests 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any


def parse_positive_int_list(value: str, *, name: str) -> list[int]:
    items: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated list of positive integers"
            ) from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"{name} must contain only positive integers"
            )
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError(
            f"{name} must contain at least one positive integer"
        )
    return sorted(set(items))


def parse_gpu_id_list(value: str, *, name: str) -> list[int]:
    items: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a comma-separated list of integers"
            ) from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError(f"{name} must contain GPU ids >= 0")
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one GPU id")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError(f"{name} must contain distinct GPU ids")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="inclusionAI/Ming-omni-tts-16.8B-A3B",
        help="HF repo id or local checkpoint directory.",
    )
    parser.add_argument("--model-name", default="ming-omni-tts")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--reference-gpu-id",
        type=int,
        default=None,
        help="GPU id for reference_encode. Defaults to --gpu-id.",
    )
    parser.add_argument(
        "--audio-decode-gpu-id",
        type=int,
        default=None,
        help="GPU id for audio_decode. Defaults to --gpu-id.",
    )
    parser.add_argument(
        "--ming-tts-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for tts_engine.",
    )
    parser.add_argument(
        "--ming-tts-tp-gpus",
        type=lambda value: parse_gpu_id_list(value, name="--ming-tts-tp-gpus"),
        default=None,
        help="Comma-separated tts_engine TP GPU ids, for example '0,1'.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-decode-steps-cap", type=int, default=256)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help=(
            "Max SGLang AR requests. Defaults to 1 when CUDA graph is disabled, "
            "or to cuda_graph_max_bs when CUDA graph is enabled."
        ),
    )
    parser.add_argument("--preprocessing-max-concurrency", type=int, default=None)
    parser.add_argument("--reference-encode-max-concurrency", type=int, default=None)
    parser.add_argument("--max-prefill-tokens", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument(
        "--tts-engine-total-gpu-memory-fraction",
        type=float,
        default=None,
        help="runtime.resources.total_gpu_memory_fraction for tts_engine.",
    )
    parser.add_argument(
        "--reference-encode-total-gpu-memory-fraction",
        type=float,
        default=None,
        help="runtime.resources.total_gpu_memory_fraction for reference_encode.",
    )
    parser.add_argument(
        "--audio-decode-total-gpu-memory-fraction",
        type=float,
        default=None,
        help="runtime.resources.total_gpu_memory_fraction for audio_decode.",
    )
    parser.add_argument("--audio-decode-max-batch-size", type=int, default=1)
    parser.add_argument("--audio-decode-max-batch-wait-ms", type=int, default=0)
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to SGLang disable_cuda_graph for the Ming AR engine.",
    )
    parser.add_argument(
        "--cuda-graph-bs",
        type=lambda value: parse_positive_int_list(value, name="--cuda-graph-bs"),
        default=None,
        help="Comma-separated CUDA graph batch sizes, for example '1,2,4,8'.",
    )
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--torch-compile-max-bs", type=int, default=None)
    parser.add_argument(
        "--disable-radix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Must remain enabled; Ming-TTS prefix/radix cache is not supported.",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dump-config", default=None)
    return parser.parse_args()


def stage(config: Any, name: str) -> Any:
    for item in config.stages:
        if item.name == name:
            return item
    raise KeyError(f"Missing Ming-Omni-TTS stage {name!r}")


def set_total_gpu_memory_fraction(stage_config: Any, value: float | None) -> None:
    if value is not None:
        stage_config.runtime.resources.total_gpu_memory_fraction = value


def build_pipeline_config(args: argparse.Namespace) -> Any:
    from sglang_omni.models.ming_tts.config import (
        AUDIO_DECODE_STAGE,
        PREPROCESSING_STAGE,
        REFERENCE_ENCODE_STAGE,
        TTS_ENGINE_STAGE,
        MingTTSPipelineConfig,
    )

    config = MingTTSPipelineConfig(model_path=args.model_path)
    config.name = args.model_name

    if args.max_running_requests is not None and args.max_running_requests <= 0:
        raise ValueError("--max-running-requests must be positive")
    if args.cuda_graph_max_bs is not None and args.cuda_graph_max_bs <= 0:
        raise ValueError("--cuda-graph-max-bs must be positive")
    if args.gpu_id < 0:
        raise ValueError("--gpu-id must be >= 0")
    if args.reference_gpu_id is not None and args.reference_gpu_id < 0:
        raise ValueError("--reference-gpu-id must be >= 0")
    if args.audio_decode_gpu_id is not None and args.audio_decode_gpu_id < 0:
        raise ValueError("--audio-decode-gpu-id must be >= 0")
    if args.ming_tts_tp_size <= 0:
        raise ValueError("--ming-tts-tp-size must be positive")
    if args.ming_tts_tp_size == 1 and args.ming_tts_tp_gpus is not None:
        raise ValueError("--ming-tts-tp-gpus only applies when --ming-tts-tp-size > 1")
    if args.ming_tts_tp_size > 1:
        if args.ming_tts_tp_gpus is None:
            raise ValueError("--ming-tts-tp-size > 1 requires --ming-tts-tp-gpus")
        if len(args.ming_tts_tp_gpus) != args.ming_tts_tp_size:
            raise ValueError(
                "--ming-tts-tp-gpus must contain exactly "
                f"{args.ming_tts_tp_size} entries"
            )
        missing_budget_flags = []
        if args.tts_engine_total_gpu_memory_fraction is None:
            missing_budget_flags.append("--tts-engine-total-gpu-memory-fraction")
        if args.reference_encode_total_gpu_memory_fraction is None:
            missing_budget_flags.append("--reference-encode-total-gpu-memory-fraction")
        if args.audio_decode_total_gpu_memory_fraction is None:
            missing_budget_flags.append("--audio-decode-total-gpu-memory-fraction")
        if missing_budget_flags:
            raise ValueError(
                "Ming-Omni-TTS tensor parallelism uses separate process groups "
                "and requires typed runtime memory budgets: "
                + ", ".join(missing_budget_flags)
            )

    reference_gpu_id = (
        args.reference_gpu_id if args.reference_gpu_id is not None else args.gpu_id
    )
    audio_decode_gpu_id = (
        args.audio_decode_gpu_id
        if args.audio_decode_gpu_id is not None
        else args.gpu_id
    )
    tts_engine_gpu: int | list[int] = (
        args.ming_tts_tp_gpus if args.ming_tts_tp_size > 1 else args.gpu_id
    )

    cuda_graph_bs = args.cuda_graph_bs
    cuda_graph_max_bs = args.cuda_graph_max_bs
    if not args.disable_cuda_graph:
        if cuda_graph_bs is None:
            cuda_graph_bs = [1]
        if cuda_graph_max_bs is None:
            cuda_graph_max_bs = max(cuda_graph_bs)
        max_running_requests = int(args.max_running_requests or cuda_graph_max_bs)
        if max_running_requests < int(cuda_graph_max_bs):
            raise ValueError(
                "--max-running-requests must be >= --cuda-graph-max-bs "
                f"({max_running_requests} < {cuda_graph_max_bs})"
            )
    else:
        max_running_requests = int(args.max_running_requests or 1)

    preprocessing_max_concurrency = (
        args.preprocessing_max_concurrency
        if args.preprocessing_max_concurrency is not None
        else max_running_requests
    )
    reference_encode_max_concurrency = (
        args.reference_encode_max_concurrency
        if args.reference_encode_max_concurrency is not None
        else max_running_requests
    )

    preprocessing = stage(config, PREPROCESSING_STAGE)
    if args.ming_tts_tp_size > 1:
        preprocessing.process = PREPROCESSING_STAGE
    preprocessing.factory_args.update(
        {
            "context_length": args.context_length,
            "max_decode_steps_cap": args.max_decode_steps_cap,
            "max_concurrency": preprocessing_max_concurrency,
        }
    )

    reference_encode = stage(config, REFERENCE_ENCODE_STAGE)
    reference_encode.gpu = reference_gpu_id
    if args.ming_tts_tp_size > 1:
        reference_encode.process = (
            "ming_tts_aux"
            if reference_gpu_id == audio_decode_gpu_id
            else REFERENCE_ENCODE_STAGE
        )
    set_total_gpu_memory_fraction(
        reference_encode,
        args.reference_encode_total_gpu_memory_fraction,
    )
    reference_encode.factory_args.update(
        {
            "dtype": args.dtype,
            "context_length": args.context_length,
            "max_concurrency": reference_encode_max_concurrency,
        }
    )

    if not args.disable_radix_cache:
        raise ValueError("--no-disable-radix-cache is not supported for Ming-TTS")
    server_args_overrides: dict[str, Any] = {
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": args.max_prefill_tokens,
        "max_running_requests": max_running_requests,
        "sampling_backend": "pytorch",
        "trust_remote_code": False,
        "chunked_prefill_size": 0,
    }
    if args.mem_fraction_static is not None:
        server_args_overrides["mem_fraction_static"] = args.mem_fraction_static
    if cuda_graph_bs:
        server_args_overrides["cuda_graph_bs"] = cuda_graph_bs
    if cuda_graph_max_bs is not None:
        server_args_overrides["cuda_graph_max_bs"] = cuda_graph_max_bs
    if args.torch_compile_max_bs is not None:
        if args.torch_compile_max_bs <= 0:
            raise ValueError("--torch-compile-max-bs must be positive")
        server_args_overrides["torch_compile_max_bs"] = args.torch_compile_max_bs

    tts_engine = stage(config, TTS_ENGINE_STAGE)
    tts_engine.gpu = tts_engine_gpu
    tts_engine.tp_size = args.ming_tts_tp_size
    tts_engine.parallelism.tp = args.ming_tts_tp_size
    if args.ming_tts_tp_size > 1:
        tts_engine.process = TTS_ENGINE_STAGE
    set_total_gpu_memory_fraction(
        tts_engine,
        args.tts_engine_total_gpu_memory_fraction,
    )
    tts_engine.factory_args.update(
        {
            "dtype": args.dtype,
            "context_length": args.context_length,
            "server_args_overrides": server_args_overrides,
        }
    )

    audio_decode = stage(config, AUDIO_DECODE_STAGE)
    audio_decode.gpu = audio_decode_gpu_id
    if args.ming_tts_tp_size > 1:
        audio_decode.process = (
            "ming_tts_aux"
            if reference_gpu_id == audio_decode_gpu_id
            else AUDIO_DECODE_STAGE
        )
    set_total_gpu_memory_fraction(
        audio_decode,
        args.audio_decode_total_gpu_memory_fraction,
    )
    audio_decode.factory_args.update(
        {
            "dtype": args.dtype,
            "decode_mode": "chunked",
            "max_batch_size": args.audio_decode_max_batch_size,
            "max_batch_wait_ms": args.audio_decode_max_batch_wait_ms,
        }
    )

    return MingTTSPipelineConfig.model_validate(config.model_dump(mode="python"))


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    config = build_pipeline_config(args)

    if args.dump_config:
        dump_path = Path(args.dump_config)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    from sglang_omni.serve import launch_server

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
