# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/modules.py.

from __future__ import annotations

try:
    from sglang_omni.models.ming_omni.talker.talker_module.modules import (
        Attention as TalkerAttention,
    )
    from sglang_omni.models.ming_omni.talker.talker_module.modules import (
        DiTBlock as TalkerDiTBlock,
    )
    from sglang_omni.models.ming_omni.talker.talker_module.modules import (
        FeedForward,
        FinalLayer,
        RMSNorm,
    )
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "Ming-Omni-TTS fm modules require x-transformers. Install the "
        "project dependencies or run `pip install x-transformers` before "
        "loading MingTTSSGLangModel."
    ) from exc


class Attention(TalkerAttention):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=None,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
        )


class DiTBlock(TalkerDiTBlock):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            qk_norm=None,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
        )


__all__ = [
    "Attention",
    "DiTBlock",
    "FeedForward",
    "FinalLayer",
    "RMSNorm",
]
