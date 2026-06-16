# SPDX-License-Identifier: MIT
# Copyright (c) 2025 inclusionAI
# Adapted from Ming-omni-tts/fm/CFM.py.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Solver:
    def __init__(
        self,
        func,
        y0: torch.Tensor,
        sigma: float = 0.25,
        temperature: float = 1.5,
    ) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        solution = torch.empty(
            len(t),
            *self.y0.shape,
            dtype=self.y0.dtype,
            device=self.y0.device,
        )
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            y1 = y0 + dt * f0

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1

            noise = torch.randn_like(y0)
            shift = self.sigma * (self.temperature**0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution

    @staticmethod
    def _linear_interp(
        t0: torch.Tensor,
        t1: torch.Tensor,
        y0: torch.Tensor,
        y1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


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
            mask=mask.to(torch.bool),
        )
        pred = pred[:, -patch_size:, :]

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[mask == 1]
        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        steps: int = 10,
        cfg_scale: float = 1.0,
        sway_sampling_coef: float | None = -1.0,
        use_epss: bool = True,
        patch_size: int = 1,
        sigma: float = 0.25,
        temperature: float = 1.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cfg_scale < 1e-5 or cfg_scale == 1.0:
            raise NotImplementedError(
                "Ming-Omni-TTS currently supports only guided CFM sampling "
                "with cfg >= 1e-5 and cfg != 1.0; public request validation "
                "should reject the disabled/unguided CFG branches."
            )

        def fn(t, x):
            pred_cfg = self.model.forward_with_cfg(
                x=x,
                t=t,
                c=c,
                latent_history=latent_history,
                cfg_scale=cfg_scale,
                patch_size=patch_size,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_scale

        y0 = noise.transpose(1, 2)
        if use_epss:
            predefined_timesteps = {
                5: [0, 2, 4, 8, 16, 32],
                6: [0, 2, 4, 6, 8, 16, 32],
                7: [0, 2, 4, 6, 8, 16, 24, 32],
                10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
                12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
                16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            }
            raw_timesteps = predefined_timesteps.get(steps, [])
            if raw_timesteps:
                t = (1 / 32) * torch.tensor(
                    raw_timesteps,
                    device=self.device,
                    dtype=noise.dtype,
                )
            else:
                t = torch.linspace(
                    0,
                    1,
                    steps + 1,
                    device=self.device,
                    dtype=noise.dtype,
                )
        else:
            t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=noise.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        solver = Solver(fn, y0, sigma=sigma, temperature=temperature)
        trajectory = solver.integrate(t)
        return trajectory[-1], trajectory
