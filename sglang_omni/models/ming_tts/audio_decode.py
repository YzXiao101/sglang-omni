# SPDX-License-Identifier: Apache-2.0
"""Audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.cache_utils import Cache

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import (
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class MingAudioDecoderState:
    dynamic_cache: Cache | None = None
    upsample_state: dict[str, Any] | None = None
    audio_buffer: torch.Tensor | None = None
    window_buffer: torch.Tensor | None = None


class MingAudioDecoder(torch.nn.Module):
    """Official-path AudioVAE decoder wrapper."""

    def __init__(self, audio_vae: AudioVAE, *, sample_rate: int) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.sample_rate = int(sample_rate)

    @classmethod
    def from_config(
        cls,
        audio_config: AudioVAEconfig,
        *,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
    ) -> "MingAudioDecoder":
        if getattr(audio_config, "semantic_module_kwargs", None) is not None:
            raise ValueError(
                "Ming-Omni-TTS serving currently uses the talker AudioVAE "
                "encode/decode path and does not support semantic_module_kwargs"
            )

        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        elif dtype == "auto":
            torch_dtype = torch.bfloat16
        elif isinstance(dtype, str):
            value = dtype.removeprefix("torch.")
            torch_dtype = getattr(torch, value, None)
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")
        else:
            raise TypeError(f"Unsupported Ming-Omni-TTS AudioVAE dtype: {dtype!r}")

        model = AudioVAE(audio_config).eval()
        model.to(device=torch.device(device), dtype=torch_dtype)
        return cls(model, sample_rate=int(audio_config.sample_rate))

    @property
    def device(self) -> torch.device:
        return next(self.audio_vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.audio_vae.parameters()).dtype

    @torch.inference_mode()
    def decode_streaming_step(
        self,
        latent_sequence: torch.Tensor,
        *,
        state: MingAudioDecoderState,
        is_last: bool,
    ) -> torch.Tensor:
        decoder = self.audio_vae.decoder
        device = self.device
        dtype = self.dtype
        latent_sequence = latent_sequence.to(
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        inputs, next_upsample_state = decoder.project_and_upsample_latents(
            latent_sequence,
            streaming=True,
            upsample_state=state.upsample_state,
            is_last=is_last,
        )

        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        next_dynamic_cache = state.dynamic_cache
        next_audio_buffer = state.audio_buffer
        next_window_buffer = state.window_buffer
        with context:
            if inputs is None:
                waveform = torch.empty(
                    (0,),
                    device=device,
                    dtype=dtype,
                )
            else:
                hidden_states, next_dynamic_cache = decoder.decode_qwen_hidden_states(
                    inputs,
                    past_key_values=state.dynamic_cache,
                    use_cache=True,
                )
                waveform, next_audio_buffer, next_window_buffer = (
                    decoder.synthesize_waveform(
                        hidden_states,
                        streaming=True,
                        audio_buffer=state.audio_buffer,
                        window_buffer=state.window_buffer,
                        is_last=is_last,
                    )
                )
                waveform = waveform[0, 0].detach()

        state.dynamic_cache = next_dynamic_cache
        state.upsample_state = next_upsample_state
        state.audio_buffer = next_audio_buffer
        state.window_buffer = next_window_buffer
        return waveform

    @torch.inference_mode()
    def decode_nonstreaming_batch(
        self,
        latent_batches: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not latent_batches:
            return []

        device = self.device
        dtype = self.dtype
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        waveforms = []
        with context:
            for latents in latent_batches:
                if int(latents.shape[0]) == 0:
                    waveforms.append(latents.new_empty((0,), dtype=torch.float32))
                    continue

                latents = latents.to(device=device, dtype=dtype)
                sequence = latents.reshape(1, -1, latents.shape[-1])
                inputs, _ = self.audio_vae.decoder.project_and_upsample_latents(
                    sequence,
                    streaming=False,
                    upsample_state=None,
                    is_last=True,
                )
                hidden_states, _ = self.audio_vae.decoder.decode_qwen_hidden_states(
                    cast(torch.Tensor, inputs),
                    past_key_values=None,
                    use_cache=False,
                )
                waveform, _, _ = self.audio_vae.decoder.synthesize_waveform(
                    hidden_states,
                    streaming=False,
                    audio_buffer=None,
                    window_buffer=None,
                    is_last=True,
                )
                waveforms.append(waveform[0, 0].detach())

        return waveforms


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    state = load_ming_tts_state(payload)
    waveform = decoder.decode_nonstreaming_batch([state.generated_latents])[0]
    state.sample_rate = int(decoder.sample_rate)
    state.duration_s = float(waveform.numel() / int(decoder.sample_rate))
    if not keep_latents:
        state.generated_latents = None

    payload = store_ming_tts_state(payload, state)
    payload.data.update(
        audio_waveform_payload(
            waveform,
            sample_rate=int(decoder.sample_rate),
            modality="audio",
            source_hint="Ming-Omni-TTS",
        )
    )
    usage = build_usage(state)
    if usage is not None:
        payload.data["usage"] = usage
    return payload


__all__ = [
    "MingAudioDecoder",
    "MingAudioDecoderState",
    "decode_ming_tts_audio_payload",
]
