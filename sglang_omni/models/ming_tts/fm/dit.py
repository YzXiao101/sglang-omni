# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/dit.py.

from __future__ import annotations

import torch

try:
    from sglang_omni.models.ming_omni.talker.talker_module.aggregator import (
        Aggregator as TalkerAggregator,
    )
    from sglang_omni.models.ming_omni.talker.talker_module.dit import CondEmbedder
    from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT as TalkerDiT
    from sglang_omni.models.ming_omni.talker.talker_module.dit import (
        SinusPositionEmbedding,
        TimestepEmbedder,
    )
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "Ming-Omni-TTS fm DiT requires x-transformers. Install the project "
        "dependencies or run `pip install x-transformers` before loading "
        "MingTTSSGLangModel."
    ) from exc


__all__ = [
    "Aggregator",
    "CondEmbedder",
    "DiT",
    "SinusPositionEmbedding",
    "TimestepEmbedder",
]


class DiT(TalkerDiT):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 896,
        cfg_dropout_prob: float = 0.1,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            llm_cond_dim=llm_cond_dim,
            cfg_dropout_prob=cfg_dropout_prob,
            grad_checkpointing=False,
            qk_norm=None,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
        )

    def initialize_weights(self) -> None:
        # Talker base __init__ calls this; TTS keeps construction init-free.
        return None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.t_embedder(t).unsqueeze(1)
        x_now = self.x_embedder(x)
        x_history = self.x_embedder(latent_history)
        x = torch.cat([x_history, x_now], dim=1)
        c = self.c_embedder(c, self.training)
        x = torch.cat([t + c, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        if mask is not None:
            mask_pad = (
                mask.clone()
                .detach()[:, :1]
                .expand(
                    -1,
                    x_history.shape[1] + c.shape[1],
                )
            )
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        return self.final_layer(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_scale: float,
        latent_history: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        if not cfg_scale == 1:
            x = torch.cat([x, x], dim=0)
            latent_history = torch.cat([latent_history, latent_history], dim=0)
            fake_latent = torch.zeros_like(c)
            c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]


class Aggregator(TalkerAggregator):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_input_dim: int = 896,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            llm_input_dim=llm_input_dim,
            qk_norm=None,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
        )

    def initialize_weights(self) -> None:
        # Talker base __init__ calls this; TTS keeps construction init-free.
        return None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "Ming TTS Aggregator expects latent shape "
                f"[batch, patch_size, latent_dim], got {tuple(x.shape)}"
            )
        return super().forward(x, mask=mask)
