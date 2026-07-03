# SPDX-License-Identifier: Apache-2.0
"""Terminal audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import torch

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import (
    MING_TTS_SAMPLE_RATE,
    MingTTSState,
    decode_generated_latents,
)
from sglang_omni.models.ming_tts.profile_events import ming_profile_event
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio_payload import audio_waveform_payload


class MingAudioDecoder(torch.nn.Module):
    """Chunked official-path AudioVAE decoder wrapper."""

    def __init__(self, audio_vae: AudioVAE, *, sample_rate: int) -> None:
        super().__init__()
        if int(sample_rate) != MING_TTS_SAMPLE_RATE:
            raise ValueError(
                "Ming-Omni-TTS currently requires AudioVAE sample_rate "
                f"{MING_TTS_SAMPLE_RATE}, got {sample_rate}"
            )
        self.audio_vae = audio_vae
        self.sample_rate = int(sample_rate)

    @classmethod
    def from_config(
        cls,
        audio_config: Any,
        *,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
    ) -> "MingAudioDecoder":
        if isinstance(audio_config, AudioVAEconfig):
            config = deepcopy(audio_config)
        elif isinstance(audio_config, dict):
            config = AudioVAEconfig(**deepcopy(audio_config))
        else:
            config = AudioVAEconfig(
                sample_rate=getattr(audio_config, "sample_rate", None),
                enc_kwargs=deepcopy(getattr(audio_config, "enc_kwargs", None)),
                dec_kwargs=deepcopy(getattr(audio_config, "dec_kwargs", None)),
                init_method=getattr(audio_config, "init_method", "normal"),
                patch_size=getattr(audio_config, "patch_size", -1),
            )

        if not isinstance(config.enc_kwargs, dict):
            raise ValueError("Ming-Omni-TTS AudioVAE config is missing enc_kwargs")
        if not isinstance(config.dec_kwargs, dict):
            raise ValueError("Ming-Omni-TTS AudioVAE config is missing dec_kwargs")
        sample_rate = int(getattr(config, "sample_rate", 0))
        if sample_rate != MING_TTS_SAMPLE_RATE:
            raise ValueError(
                "Ming-Omni-TTS AudioVAE config sample_rate must be "
                f"{MING_TTS_SAMPLE_RATE}, got {sample_rate}"
            )
        # Current TTS serving policy keeps AudioVAE Qwen2 blocks on SDPA. FM and
        # AudioVAE backend selection are still planned optimization work.
        for stage_kwargs in (config.enc_kwargs, config.dec_kwargs):
            stage_kwargs["backbone"]["_attn_implementation"] = "sdpa"
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

        model = AudioVAE(config).eval()
        model.to(device=torch.device(device), dtype=torch_dtype)
        return cls(model, sample_rate=sample_rate)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.audio_vae.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.audio_vae.parameters()).dtype
        except StopIteration:
            return torch.float32

    @torch.inference_mode()
    def decode_chunks(
        self,
        latents: torch.Tensor,
        last_chunks: list[bool],
        *,
        decode_mode: str = "chunked",
        request_id: str | None = None,
    ) -> torch.Tensor:
        if decode_mode != "chunked":
            raise NotImplementedError(
                "Ming-Omni-TTS currently supports only chunked AudioVAE decode"
            )

        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(latents)
        latents = latents.to(device=self.device, dtype=self.dtype)

        last_chunks = [bool(item) for item in last_chunks]

        stream_state = (None, None, None)
        past_key_values = None
        waveform_chunks = []
        autocast_dtype = self.dtype
        if autocast_dtype not in (torch.float16, torch.bfloat16):
            autocast_dtype = torch.bfloat16
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            for step, last_chunk in enumerate(last_chunks):
                chunk = latents[step : step + 1]
                event_context = (
                    ming_profile_event(
                        request_id,
                        "ming_audio_decode_chunk",
                        metadata={
                            "step": int(step),
                            "last_chunk": bool(last_chunk),
                        },
                    )
                    if request_id is not None
                    else nullcontext()
                )
                with event_context:
                    wav, stream_state, past_key_values = self.audio_vae.decode(
                        chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        stream_state=stream_state,
                        last_chunk=last_chunk,
                    )
                wav = self._normalize_waveform_chunk(wav)
                if wav.numel():
                    waveform_chunks.append(wav)

        return torch.cat(waveform_chunks, dim=0)

    @staticmethod
    def _normalize_waveform_chunk(wav: Any) -> torch.Tensor:
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        while wav.ndim > 1:
            wav = wav[0]
        return wav.detach()


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    decode_mode: str = "chunked",
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    started = time.perf_counter()
    state = MingTTSState.from_dict(payload.data)

    latents = decode_generated_latents(
        state,
        device=decoder.device,
        dtype=decoder.dtype,
    )

    with ming_profile_event(
        payload.request_id,
        "ming_audio_decode",
        metadata={"latent_chunks": int(latents.shape[0])},
    ):
        waveform = decoder.decode_chunks(
            latents,
            state.generated_last_chunk,
            decode_mode=decode_mode,
            request_id=payload.request_id,
        )
    state.audio_decode_time_s = time.perf_counter() - started
    sample_rate = decoder.sample_rate

    waveform = MingAudioDecoder._normalize_waveform_chunk(waveform)

    state.sample_rate = int(sample_rate)
    state.duration_s = float(waveform.numel() / int(sample_rate))
    if not keep_latents:
        state.generated_latents_bytes = None
        state.generated_latents_shape = None
        state.generated_latents_dtype = None

    payload.data = state.to_dict()
    payload.data.update(
        audio_waveform_payload(
            waveform,
            sample_rate=int(sample_rate),
            modality="audio",
            source_hint="Ming-Omni-TTS",
        )
    )
    if state.prompt_tokens or state.completion_tokens or state.engine_time_s:
        usage: dict[str, Any] = {
            "prompt_tokens": int(state.prompt_tokens),
            "completion_tokens": int(state.completion_tokens),
            "total_tokens": int(state.prompt_tokens + state.completion_tokens),
        }
        if state.engine_time_s:
            usage["engine_time_s"] = round(float(state.engine_time_s), 6)
        payload.data["usage"] = usage
    return payload


__all__ = [
    "MingAudioDecoder",
    "decode_ming_tts_audio_payload",
]
