# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

import torch


class SGLKernelRoPEProvider:
    """Apply one immutable Ming-TTS RoPE source with the SGLang AOT kernel.

    DiT and Aggregator own separate instances so a cache is never reused across
    different frequency sources. All compatibility checks happen before Q/K are
    mutated; kernel failures propagate because the inputs may already be changed.
    """

    def __init__(self, rotary_embedding: Callable[..., None]) -> None:
        self._rotary_embedding = rotary_embedding
        self._cos_sin_caches: dict[
            tuple[torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._positions: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    @staticmethod
    def _build_cos_sin_cache(
        freqs: torch.Tensor,
        *,
        sequence_length: int,
    ) -> torch.Tensor:
        angles = freqs[0, :sequence_length, 0::2]
        return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rope: tuple[torch.Tensor, object],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if query.ndim != 4 or key.shape != query.shape:
            return None
        freqs, xpos_scale = rope
        if isinstance(xpos_scale, torch.Tensor) or (
            xpos_scale is not None and xpos_scale != 1.0
        ):
            return None

        batch_size, sequence_length, heads, head_dim = query.shape
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or key.device != query.device
            or key.dtype != query.dtype
            or batch_size == 0
            or sequence_length == 0
            or heads == 0
            or head_dim != 64
            or not query.is_contiguous()
            or not key.is_contiguous()
        ):
            return None
        if (
            freqs.device != query.device
            or freqs.dtype != query.dtype
            or freqs.ndim != 3
            or freqs.shape[0] != 1
            or freqs.shape[1] != sequence_length
            or freqs.shape[2] != head_dim
        ):
            return None

        cache_key = (query.device, query.dtype, sequence_length, head_dim)
        cos_sin_cache = self._cos_sin_caches.get(cache_key)
        if cos_sin_cache is None:
            cos_sin_cache = self._build_cos_sin_cache(
                freqs,
                sequence_length=sequence_length,
            )
            self._cos_sin_caches[cache_key] = cos_sin_cache

        position_key = (query.device, batch_size, sequence_length)
        positions = self._positions.get(position_key)
        if positions is None:
            positions = torch.arange(
                sequence_length,
                device=query.device,
                dtype=torch.long,
            ).repeat(batch_size)
            self._positions[position_key] = positions

        query_flat = query.view(batch_size * sequence_length, heads * head_dim)
        key_flat = key.view(batch_size * sequence_length, heads * head_dim)
        # x-transformers RotaryEmbedding uses GPT-J interleaved pairs.
        self._rotary_embedding(
            positions,
            query_flat,
            key_flat,
            head_dim,
            cos_sin_cache,
            False,
        )
        return query, key
