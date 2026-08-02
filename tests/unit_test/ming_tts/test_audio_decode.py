# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)
from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    MingAudioDecoderState,
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

    def decode_nonstreaming(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        assert latents.shape == (0, 2, 3)
        self.calls += 1
        return torch.empty((0,), dtype=torch.float32)


class _FailingAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()))

    @property
    def decoder(self):
        raise AssertionError("empty latents should not call the AudioVAE decoder")


def test_ming_audio_decoder_skips_audio_vae_for_empty_latents() -> None:
    decoder = MingAudioDecoder(_FailingAudioVAE(), sample_rate=44100)

    waveform = decoder.decode_nonstreaming(torch.empty((0, 2, 3), dtype=torch.float32))

    assert waveform.shape == (0,)
    assert waveform.dtype == torch.float32


def test_ming_audio_decoder_incremental_matches_full_sequence_on_cpu() -> None:
    backbone = {
        "_attn_implementation": "sdpa",
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "max_position_embeddings": 256,
        "max_window_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "sliding_window": 64,
        "use_cache": False,
        "use_sliding_window": True,
        "vocab_size": 1,
    }
    config = AudioVAEconfig(
        sample_rate=44100,
        enc_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 4},
            "input_dim": 4,
            "hop_size": 4,
            "latent_dim": 4,
        },
        dec_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 1},
            "output_dim": 4,
            "latent_dim": 4,
        },
        patch_size=4,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        decoder = MingAudioDecoder(AudioVAE(config).eval(), sample_rate=44100)
        latents = torch.randn(5, 4, 4)

        full = decoder.decode_nonstreaming(latents)
        state = MingAudioDecoderState()
        incremental_parts = [
            decoder.decode_streaming_step(
                latents[0],
                state=state,
                is_last=False,
            ),
            decoder.decode_streaming_step(
                latents[1:3].flatten(0, 1),
                state=state,
                is_last=False,
            ),
            decoder.decode_streaming_step(
                latents[3:].flatten(0, 1),
                state=state,
                is_last=True,
            ),
        ]

    incremental = torch.cat(incremental_parts)
    assert incremental_parts[0].numel() == 0
    assert incremental_parts[-1].numel() > 0
    assert full.numel() > 0
    assert incremental.shape == full.shape
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-6)


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
