# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/flowloss.py.

from __future__ import annotations

import torch
from torch import nn

from .cfm import CFM, build_cfm_sampling_schedule
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
        cfg: float | torch.Tensor = 1.0,
        patch_size: int = 1,
        sigma: float = 0.25,
        temperature: float = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(z.shape[0], self.z_channels, patch_size, device=z.device)
        timesteps, sde_random = build_cfm_sampling_schedule(
            steps=10,
            device=z.device,
            dtype=noise.dtype,
            batch_size=int(z.shape[0]),
            patch_size=int(patch_size),
            latent_dim=int(self.z_channels),
        )
        return self.sample_with_noise(
            z=z,
            latent_history=latent_history,
            noise=noise,
            cfg=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            timesteps=timesteps,
            sde_random=sde_random,
        )

    def sample_with_noise(
        self,
        z: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: float | torch.Tensor = 1.0,
        patch_size: int = 1,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 0,
        validate_cfg: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cfm.sample_with_noise(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            timesteps=timesteps,
            sde_random=sde_random,
            validate_cfg=validate_cfg,
        )

    def sample_final_with_noise(
        self,
        z: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: float | torch.Tensor = 1.0,
        patch_size: int = 1,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 0,
        validate_cfg: bool = True,
    ) -> torch.Tensor:
        return self.cfm.sample_final_with_noise(
            noise=noise,
            c=z,
            latent_history=latent_history,
            cfg_scale=cfg,
            patch_size=patch_size,
            sigma=sigma,
            temperature=temperature,
            timesteps=timesteps,
            sde_random=sde_random,
            validate_cfg=validate_cfg,
        )
