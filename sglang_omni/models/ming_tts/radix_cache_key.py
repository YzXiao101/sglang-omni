# SPDX-License-Identifier: Apache-2.0
"""Radix-cache keys for Ming-Omni-TTS projected prefill rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

MING_ROW_PREFILL_CACHE_VERSION = "ming_tts_row_prefill_v1"


def build_ming_prefill_row_cache_key_ids(
    prefill_input_embeds: torch.Tensor,
) -> list[int]:
    """Build stable synthetic radix ids for final prefill embedding rows.

    The input must already be cast to the dtype used by the Ming AR backbone and
    must include all speaker/prompt-latent row replacements.  We serialize as
    float32 only to avoid bfloat16 NumPy compatibility issues; the tensor values
    have already been quantized to the actual backbone input dtype.
    """

    if prefill_input_embeds.ndim != 2:
        raise ValueError(
            "Ming-TTS prefill row cache expects a 2-D tensor [tokens, hidden], "
            f"got shape {tuple(prefill_input_embeds.shape)}"
        )

    rows = (
        prefill_input_embeds.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def build_ming_row_prefill_extra_key(
    *,
    model_identity: str,
    input_dtype: torch.dtype,
    hidden_size: int,
    patch_size: int,
    latent_dim: int,
    audio_start_token_id: int,
    audio_patch_token_id: int,
    audio_eos_token_id: int,
) -> str:
    """Build the stable radix namespace for Ming row-hash prefill keys."""

    payload: dict[str, Any] = {
        "version": MING_ROW_PREFILL_CACHE_VERSION,
        "model_identity": str(model_identity),
        "input_dtype": str(input_dtype),
        "hidden_size": int(hidden_size),
        "patch_size": int(patch_size),
        "latent_dim": int(latent_dim),
        "audio_start_token_id": int(audio_start_token_id),
        "audio_patch_token_id": int(audio_patch_token_id),
        "audio_eos_token_id": int(audio_eos_token_id),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
    return f"ming_tts:row-prefill:v1:{digest}"


__all__ = [
    "MING_ROW_PREFILL_CACHE_VERSION",
    "build_ming_prefill_row_cache_key_ids",
    "build_ming_row_prefill_extra_key",
]
