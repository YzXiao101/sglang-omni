# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeDecoder:
    sample_rate = 44100
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls = 0

    def decode_nonstreaming_batch(
        self,
        latent_batches: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        assert len(latent_batches) == 1
        assert latent_batches[0].shape == (0, 2, 3)
        self.calls += 1
        return [torch.empty((0,), dtype=torch.float32)]


class _FailingAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()))

    @property
    def decoder(self):
        raise AssertionError("empty latents should not call the AudioVAE decoder")


def test_ming_audio_decoder_skips_audio_vae_for_empty_latents() -> None:
    decoder = MingAudioDecoder(_FailingAudioVAE(), sample_rate=44100)

    waveforms = decoder.decode_nonstreaming_batch(
        [torch.empty((0, 2, 3), dtype=torch.float32)]
    )

    assert len(waveforms) == 1
    assert waveforms[0].shape == (0,)
    assert waveforms[0].dtype == torch.float32


def test_ming_tts_audio_decode_accepts_empty_generated_latents() -> None:
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=0,
        generated_latents=torch.empty((0, 2, 3), dtype=torch.float32),
    )
    payload = StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    decoder = _FakeDecoder()

    result = decode_ming_tts_audio_payload(payload, decoder)

    assert decoder.calls == 1
    assert result.data["sample_rate"] == 44100
    assert result.data["duration_s"] == 0.0
    assert result.data["audio_waveform_shape"] == [0]
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == []
