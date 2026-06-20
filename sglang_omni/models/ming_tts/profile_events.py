# SPDX-License-Identifier: Apache-2.0
"""Lightweight request-event helpers for Ming-Omni-TTS profiling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from sglang_omni.profiler.event_recorder import emit as _emit_event


def emit_ming_event(
    request_id: str,
    event_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit one Ming request-event without forcing a stage name."""

    _emit_event(
        request_id=request_id,
        stage=None,
        event_name=event_name,
        metadata=metadata,
    )


@contextmanager
def ming_profile_event(
    request_id: str,
    event_base: str,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Emit ``<event_base>_start/end`` around a hot-path block."""

    emit_ming_event(request_id, f"{event_base}_start", metadata)
    try:
        yield
    finally:
        emit_ming_event(request_id, f"{event_base}_end", metadata)


def tensor_metadata(value: Any) -> dict[str, Any]:
    """Return token-safe tensor metadata for profile JSONL events."""

    return {
        "shape": list(getattr(value, "shape", []) or []),
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
    }


__all__ = [
    "emit_ming_event",
    "ming_profile_event",
    "tensor_metadata",
]
