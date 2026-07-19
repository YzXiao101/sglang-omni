# SPDX-License-Identifier: Apache-2.0
"""Targeted JSONL observations for Ming-TTS correctness debugging."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch

_TRACE_ENABLED_ENV = "MING_TTS_DEBUG_TRACE"
_TRACE_TEXT_ENV = "MING_TTS_DEBUG_TEXT"
_TRACE_DIR_ENV = "MING_TTS_DEBUG_DIR"
_FIXED_REFERENCE_SEED_ENV = "MING_TTS_DEBUG_FIXED_REFERENCE_SEED"
_FIXED_TAIL_SEED_ENV = "MING_TTS_DEBUG_FIXED_TAIL_SEED"
_TAIL_SEED_SEQUENCE_ENV = "MING_TTS_DEBUG_TAIL_SEEDS"
_DEFAULT_TRACE_DIR = "/tmp/ming_tts_debug_trace"

_LOCK = threading.Lock()
_TRACE_PATHS: dict[str, Path] = {}


def is_enabled() -> bool:
    enabled = os.environ.get(_TRACE_ENABLED_ENV, "0").lower()
    target = os.environ.get(_TRACE_TEXT_ENV)
    return enabled in {"1", "true", "yes", "on"} and bool(target)


def matches_text(text: str | None) -> bool:
    return is_enabled() and text == os.environ.get(_TRACE_TEXT_ENV)


def fixed_reference_seed() -> int | None:
    value = os.environ.get(_FIXED_REFERENCE_SEED_ENV)
    return int(value) if value is not None else None


def fixed_tail_seed() -> int | None:
    value = os.environ.get(_FIXED_TAIL_SEED_ENV)
    return int(value) if value else None


def tail_seed_sequence() -> tuple[int, ...]:
    value = os.environ.get(_TAIL_SEED_SEQUENCE_ENV)
    if not value:
        return ()
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError(
            f"{_TAIL_SEED_SEQUENCE_ENV} must contain non-negative integers"
        )
    return seeds


def tensor_stats(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    source = tensor.detach()
    values = source.abs() if source.is_complex() else source
    values = values.to(device="cpu", dtype=torch.float32).contiguous()
    stats: dict[str, Any] = {
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "device": str(source.device),
        "numel": int(source.numel()),
    }
    if values.numel() == 0:
        return stats

    finite = torch.isfinite(values)
    finite_values = values[finite]
    stats["finite"] = bool(finite.all().item())
    stats["fingerprint"] = hashlib.sha256(values.numpy().tobytes()).hexdigest()[:16]
    stats["preview"] = values.flatten()[:4].tolist()
    if finite_values.numel() != 0:
        stats.update(
            mean=float(finite_values.mean().item()),
            std=float(finite_values.std(unbiased=False).item()),
            rms=float(finite_values.square().mean().sqrt().item()),
            abs_max=float(finite_values.abs().max().item()),
        )
    return stats


def rng_fingerprint(device: torch.device) -> str:
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
    else:
        state = torch.random.get_rng_state()
    return hashlib.sha256(bytes(state.cpu().tolist())).hexdigest()[:16]


def write_event(stage: str, event: str, **payload: Any) -> None:
    record = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "stage": stage,
        "event": event,
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        path = _trace_path(stage)
        with path.open("a", encoding="utf-8") as output:
            output.write(line)
            output.write("\n")


def _trace_path(stage: str) -> Path:
    key = f"{stage}:{os.getpid()}"
    path = _TRACE_PATHS.get(key)
    if path is not None:
        return path

    trace_dir = Path(os.environ.get(_TRACE_DIR_ENV, _DEFAULT_TRACE_DIR))
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "trace"
    path = trace_dir / f"{safe_stage}_pid{os.getpid()}.jsonl"
    _TRACE_PATHS[key] = path
    return path


__all__ = [
    "fixed_reference_seed",
    "fixed_tail_seed",
    "is_enabled",
    "matches_text",
    "rng_fingerprint",
    "tail_seed_sequence",
    "tensor_stats",
    "write_event",
]
