# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/flowloss.py.

from __future__ import annotations

import torch
from torch import nn

from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT

from .cfm import CFM


class FlowLoss(nn.Module):
    """Ming-Omni-TTS flow-matching latent head."""

    def __init__(
        self,
        z_channels: int,
        llm_cond_dim: int,
        patch_size: int | None = None,
        history_patch_size: int | None = None,
        **dit_kwargs,
    ) -> None:
        super().__init__()
        del patch_size, history_patch_size
        self.z_channels = z_channels
        self.cfm = CFM(
            model=DiT(
                in_channels=z_channels,
                llm_cond_dim=llm_cond_dim,
                **dit_kwargs,
            )
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
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: float | torch.Tensor = 1.0,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 0,
    ) -> torch.Tensor:
        return self.cfm.sample(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            sigma=sigma,
            temperature=temperature,
            timesteps=timesteps,
            sde_random=sde_random,
        )
