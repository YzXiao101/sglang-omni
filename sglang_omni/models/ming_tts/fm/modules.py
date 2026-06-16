# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/modules.py.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from x_transformers.x_transformers import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "Ming-Omni-TTS fm modules require x-transformers. Install the "
        "project dependencies or run `pip install x-transformers` before "
        "loading MingTTSSGLangModel."
    ) from exc


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.native_rms_norm:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            return F.rms_norm(
                x,
                normalized_shape=(x.shape[-1],),
                weight=self.weight,
                eps=self.eps,
            )

        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4,
        dropout: float = 0.0,
        approximate: str = "none",
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate=approximate)),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "Ming-Omni-TTS fm attention requires PyTorch 2.0 or newer."
            )

        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(
            dim=hidden_size,
            mult=mlp_ratio,
            dropout=dropout,
            approximate="tanh",
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        return self.linear(x)
