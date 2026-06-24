# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/CFM.py.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.ming_omni.talker.talker_module.cfm import get_epss_timesteps


def build_cfm_sampling_schedule(
    *,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    patch_size: int,
    latent_dim: int,
    use_epss: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_epss:
        timesteps = get_epss_timesteps(int(steps), device=device, dtype=dtype)
    else:
        timesteps = torch.linspace(0, 1, int(steps) + 1, device=device, dtype=dtype)
    y0_shape = (int(batch_size), int(patch_size), int(latent_dim))
    sde_random = torch.stack(
        [torch.randn(y0_shape, device=device, dtype=dtype) for _ in range(int(steps))],
        dim=0,
    )
    return timesteps, sde_random


def _expand_batch_param(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)

    if tensor.ndim == 0 or int(tensor.numel()) == 1:
        return tensor.reshape(1, 1, 1).expand(int(batch_size), 1, 1)

    if tuple(tensor.shape) == (int(batch_size),):
        return tensor.reshape(int(batch_size), 1, 1)

    if tuple(tensor.shape) == (int(batch_size), 1):
        return tensor.reshape(int(batch_size), 1, 1)

    if tuple(tensor.shape) == (int(batch_size), 1, 1):
        return tensor

    raise ValueError(
        f"Ming-Omni-TTS CFM {name} must be scalar, [B], [B, 1], "
        f"or [B, 1, 1], got {tuple(tensor.shape)} for B={int(batch_size)}"
    )


def _validate_guided_cfg(cfg_scale: torch.Tensor) -> None:
    invalid = torch.logical_or(cfg_scale < 1e-5, cfg_scale == 1.0)
    valid = torch.logical_not(torch.any(invalid))
    if not bool(valid.detach().cpu().item()):
        raise NotImplementedError(
            "Ming-Omni-TTS currently supports only guided CFM sampling "
            "with cfg >= 1e-5 and cfg != 1.0; public request validation "
            "should reject the disabled/unguided CFG branches."
        )


class Solver:
    def __init__(
        self,
        func,
        y0: torch.Tensor,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 1.5,
    ) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate_final(
        self,
        t: torch.Tensor,
        *,
        sde_random: torch.Tensor,
    ) -> torch.Tensor:
        expected_sde_shape = (int(t.shape[0]) - 1, *self.y0.shape)
        if tuple(sde_random.shape) != expected_sde_shape:
            raise ValueError(
                "Ming-Omni-TTS CFM sde_random must have shape "
                f"{expected_sde_shape}, got {tuple(sde_random.shape)}"
            )
        step_count = int(t.shape[0]) - 1
        if step_count <= 0:
            raise ValueError("Ming-Omni-TTS CFM timesteps require at least one step")

        y0 = self.y0
        final = y0
        for step, (t0, t1) in enumerate(zip(t[:-1], t[1:])):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            y1 = y0 + dt * f0
            final = y1

            if step + 1 < step_count:
                noise = sde_random[step]
                shift = self.sigma * (self.temperature**0.5) * (abs(dt) ** 0.5) * noise
                y0 = y1 + shift

        return final


class CFM(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        cond: torch.Tensor,
        target: torch.Tensor,
        latent_history: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        x1 = target
        batch, dtype = x1.shape[0], x1.dtype
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        t = time.unsqueeze(-1).unsqueeze(-1)
        x = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.model(
            x=x,
            t=time,
            c=cond,
            latent_history=latent_history,
        )
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[mask == 1]
        return loss.mean()

    @torch.no_grad()
    def sample_final_with_noise(
        self,
        noise: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        steps: int = 10,
        cfg_scale: float = 1.0,
        sway_sampling_coef: float | None = -1.0,
        sigma: float | torch.Tensor = 0.25,
        temperature: float | torch.Tensor = 1.5,
        validate_cfg: bool = True,
    ) -> torch.Tensor:
        fn, y0, t, sigma_tensor, temperature_tensor = self._prepare_sampling(
            noise=noise,
            c=c,
            latent_history=latent_history,
            timesteps=timesteps,
            sde_random=sde_random,
            steps=steps,
            cfg_scale=cfg_scale,
            sway_sampling_coef=sway_sampling_coef,
            sigma=sigma,
            temperature=temperature,
            validate_cfg=validate_cfg,
        )
        solver = Solver(fn, y0, sigma=sigma_tensor, temperature=temperature_tensor)
        return solver.integrate_final(t, sde_random=sde_random)

    def _prepare_sampling(
        self,
        *,
        noise: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        steps: int,
        cfg_scale: float | torch.Tensor,
        sway_sampling_coef: float | None,
        sigma: float | torch.Tensor,
        temperature: float | torch.Tensor,
        validate_cfg: bool,
    ):
        batch_size = int(noise.shape[0])
        cfg_tensor = _expand_batch_param(
            cfg_scale,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
            name="cfg_scale",
        )
        sigma_tensor = _expand_batch_param(
            sigma,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
            name="sigma",
        )
        temperature_tensor = _expand_batch_param(
            temperature,
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
            name="temperature",
        )
        if validate_cfg:
            _validate_guided_cfg(cfg_tensor)

        def fn(t, x):
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_tensor

        y0 = noise.transpose(1, 2)
        if timesteps is None or sde_random is None:
            raise ValueError(
                "Ming-Omni-TTS CFM explicit sampling requires timesteps "
                "and sde_random"
            )
        if timesteps.ndim != 1:
            raise ValueError(
                "Ming-Omni-TTS CFM timesteps must be one-dimensional, "
                f"got shape {tuple(timesteps.shape)}"
            )
        if int(timesteps.shape[0]) != int(steps) + 1:
            raise ValueError(
                "Ming-Omni-TTS CFM timesteps length must equal steps + 1, "
                f"got {int(timesteps.shape[0])} for steps={int(steps)}"
            )
        t = timesteps
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        return fn, y0, t, sigma_tensor, temperature_tensor
