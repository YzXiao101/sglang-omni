# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for the shared Ming AudioVAE ISTFT primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from sglang_omni.models.ming_omni.talker.audio_vae.istft import ISTFT, ISTFTHead


def test_spectrum_primitive_matches_pre_refactor_inline_math() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = ISTFTHead(dim=3, n_fft=8, hop_length=2)
        hidden = torch.randn(2, 5, 3)

    with torch.no_grad():
        head.out.weight[0].zero_()
        head.out.bias[0] = math.log(200.0)

    # This is the inline computation used by ISTFTHead.forward before the
    # shared primitive was extracted.
    expected_projection = head.out(hidden).transpose(1, 2)
    magnitude, phase = expected_projection.chunk(2, dim=1)
    unclipped_magnitude = torch.exp(magnitude)
    expected_spectrum = torch.clip(unclipped_magnitude, max=1e2) * (
        torch.cos(phase) + 1j * torch.sin(phase)
    )

    spectrum, projection = head._predict_spectrum(hidden)

    assert bool((unclipped_magnitude > 1e2).any())
    assert projection.shape == expected_projection.shape
    assert projection.dtype == expected_projection.dtype
    assert spectrum.shape == expected_spectrum.shape
    assert spectrum.dtype == expected_spectrum.dtype
    assert torch.equal(projection, expected_projection)
    assert torch.equal(spectrum, expected_spectrum)


def test_overlap_add_primitive_matches_pre_refactor_inline_math() -> None:
    istft = ISTFT(n_fft=8, hop_length=2, win_length=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        spectrum = torch.complex(torch.randn(2, 5, 4), torch.randn(2, 5, 4))

    # This is the inline computation used by ISTFT.forward before the shared
    # primitive was extracted.
    inverse = torch.fft.irfft(spectrum, istft.n_fft, dim=1, norm="backward")
    inverse = inverse * istft.window[None, :, None]
    output_size = (spectrum.shape[-1] - 1) * istft.hop_length + istft.win_length
    expected_numerator = F.fold(
        inverse,
        output_size=(1, output_size),
        kernel_size=(1, istft.win_length),
        stride=(1, istft.hop_length),
    )[:, 0, 0, :]

    window_frames = (
        istft.window.square().expand(1, spectrum.shape[-1], -1).transpose(1, 2)
    )
    expected_denominator = (
        F.fold(
            window_frames,
            output_size=(1, output_size),
            kernel_size=(1, istft.win_length),
            stride=(1, istft.hop_length),
        )
        .squeeze(0)
        .squeeze(0)
    )

    numerator, denominator = istft._overlap_add(spectrum)

    assert numerator.shape == expected_numerator.shape
    assert numerator.dtype == expected_numerator.dtype
    assert denominator.shape == expected_denominator.shape
    assert denominator.dtype == expected_denominator.dtype
    assert torch.equal(numerator, expected_numerator)
    assert torch.equal(denominator, expected_denominator)
