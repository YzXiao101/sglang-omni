# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/flowloss.py.

from __future__ import annotations

import torch
from torch import nn

from .cfm import CFM
from .dit import DiT


class FlowLoss(nn.Module):
    """Ming-Omni-TTS flow-matching latent head."""

    def __init__(self, z_channels: int, llm_cond_dim: int, **kwargs) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs)
        )

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        latent_history: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        return self.cfm(
            cond=cond,
            target=target,
            latent_history=latent_history,
            mask=mask,
            patch_size=patch_size,
        )

    def sample(
        self,
        z: torch.Tensor,
        latent_history: torch.Tensor,
        cfg: float = 1.0,
        patch_size: int = 1,
        sigma: float = 0.25,
        temperature: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(z.shape[0], self.z_channels, patch_size, device=z.device)
        return self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
        )
