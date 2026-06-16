# SPDX-License-Identifier: Apache-2.0
"""ISTFT head for the Ming-Omni-TTS AudioVAE.

Adapted from the official Ming-omni-tts ``audio_tokenizer`` package.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ISTFT(nn.Module):
    """Custom ISTFT with streaming buffers and same-padding trimming."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        if padding not in ("center", "same"):
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
        self.buffer_len = self.win_length - self.hop_length

    def _buffer_process(
        self,
        x: torch.Tensor,
        buffer: torch.Tensor | None,
        pad: int,
        *,
        last_chunk: bool = False,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            else:
                x[:, : self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len :]
            if not last_chunk:
                x = x[:, : -self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def forward(
        self,
        spec: torch.Tensor,
        audio_buffer: torch.Tensor | None = None,
        window_buffer: torch.Tensor | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.padding == "center":
            audio = torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
            return audio, audio_buffer, window_buffer

        pad = (self.win_length - self.hop_length) // 2
        if spec.dim() != 3:
            raise RuntimeError(f"ISTFT expects a 3D spec tensor, got {spec.shape}")

        _, _, num_frames = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        output_size = (num_frames - 1) * self.hop_length + self.win_length
        audio = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]
        audio, audio_buffer = self._buffer_process(
            audio,
            audio_buffer,
            pad,
            last_chunk=last_chunk,
            streaming=streaming,
        )

        window_sq = self.window.square().expand(1, num_frames, -1).transpose(1, 2)
        window_envelope = (
            torch.nn.functional.fold(
                window_sq,
                output_size=(1, output_size),
                kernel_size=(1, self.win_length),
                stride=(1, self.hop_length),
            )
            .squeeze(0)
            .squeeze(0)
        )
        window_envelope, window_buffer = self._buffer_process(
            window_envelope,
            window_buffer,
            pad,
            last_chunk=last_chunk,
            streaming=streaming,
        )

        if not bool((window_envelope > 1e-11).all()):
            raise RuntimeError("ISTFT window envelope contains near-zero values")
        audio = audio / window_envelope.squeeze()
        return audio, audio_buffer, window_buffer


class FourierHead(nn.Module):
    """Base class for inverse Fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward")


class ISTFTHead(FourierHead):
    """Predict complex STFT coefficients and reconstruct waveform chunks."""

    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        audio_buffer: torch.Tensor | None = None,
        window_buffer: torch.Tensor | None = None,
        streaming: bool = False,
        last_chunk: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        predicted = self.out(x).transpose(1, 2)
        magnitude, phase = predicted.chunk(2, dim=1)
        magnitude = torch.clip(torch.exp(magnitude), max=1e2)
        spec = magnitude * (torch.cos(phase) + 1j * torch.sin(phase))
        audio, audio_buffer, window_buffer = self.istft(
            spec,
            audio_buffer=audio_buffer,
            window_buffer=window_buffer,
            streaming=streaming,
            last_chunk=last_chunk,
        )
        return audio.unsqueeze(1), predicted, audio_buffer, window_buffer


__all__ = ["FourierHead", "ISTFT", "ISTFTHead"]
