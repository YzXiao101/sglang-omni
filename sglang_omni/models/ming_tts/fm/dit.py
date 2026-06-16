# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/dit.py.

from __future__ import annotations

import math

import torch
from torch import nn

from .modules import DiTBlock, FinalLayer

try:
    from x_transformers.x_transformers import RotaryEmbedding
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "Ming-Omni-TTS fm DiT requires x-transformers. Install the project "
        "dependencies or run `pip install x-transformers` before loading "
        "MingTTSSGLangModel."
    ) from exc


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class CondEmbedder(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        hidden_size: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def cond_drop(self, llm_cond: torch.Tensor) -> torch.Tensor:
        bsz = llm_cond.shape[0]
        drop_latent_mask = torch.rand(bsz) < self.dropout_prob
        drop_latent_mask = drop_latent_mask.unsqueeze(-1).unsqueeze(-1)
        drop_latent_mask = drop_latent_mask.to(llm_cond.dtype).to(llm_cond.device)
        fake_latent = torch.zeros(llm_cond.shape).to(llm_cond.device)
        return drop_latent_mask * fake_latent + (1 - drop_latent_mask) * llm_cond

    def forward(self, llm_cond: torch.Tensor, train: bool) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            llm_cond = self.cond_drop(llm_cond)
        return self.cond_embedder(llm_cond)


class DiT(nn.Module):
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

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
            fake_latent = torch.zeros(c.shape).to(c.device)
            c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]


class Aggregator(nn.Module):
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
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

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

        x = self.x_embedder(x)
        cls_embed = self.word_embedder(
            torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device)
        )
        x = torch.cat([cls_embed, x], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1]
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        return x[:, :1, :]
