# SPDX-License-Identifier: Apache-2.0
"""Debug-only JSONL tracing for Ming-Omni-TTS serving."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import torch

_TRACE_ENV = "MING_TTS_DEBUG_TRACE"
_TRACE_DIR_ENV = "MING_TTS_DEBUG_DIR"
_DEFAULT_TRACE_DIR = "/tmp/ming_tts_debug_trace"
_LOCK = threading.Lock()
_PATHS: dict[str, Path] = {}


def is_enabled() -> bool:
    return os.environ.get(_TRACE_ENV, "0").lower() in {"1", "true", "yes", "on"}


def tensor_stats(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.detach()
    stats: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return stats

    values = tensor
    if values.is_complex():
        values = values.abs()
    values = values.to(dtype=torch.float32)
    finite = torch.isfinite(values)
    stats["finite"] = bool(finite.all().item())
    values = values[finite]
    if values.numel() == 0:
        return stats

    stats.update(
        {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "abs_mean": float(values.abs().mean().item()),
            "l2": float(torch.linalg.vector_norm(values).item()),
        }
    )
    return stats


def write_event(stage: str, event: str, **payload: Any) -> None:
    if not is_enabled():
        return

    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "stage": stage,
        "event": event,
    }
    record.update(payload)
    line = json.dumps(record, ensure_ascii=False, default=str)

    with _LOCK:
        path = _trace_path(stage)
        with path.open("a", encoding="utf-8") as out:
            out.write(line)
            out.write("\n")


def _trace_path(stage: str) -> Path:
    key = f"{stage}:{os.getpid()}"
    path = _PATHS.get(key)
    if path is not None:
        return path

    trace_dir = Path(os.environ.get(_TRACE_DIR_ENV, _DEFAULT_TRACE_DIR))
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "trace"
    path = trace_dir / f"{safe_stage}_pid{os.getpid()}.jsonl"
    _PATHS[key] = path
    return path


__all__ = ["is_enabled", "tensor_stats", "write_event"]
