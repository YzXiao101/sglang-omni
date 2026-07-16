# SPDX-License-Identifier: Apache-2.0
"""Audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import (
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _MingAudioDecodeState:
    past_key_values: Any = None
    stream_state: tuple[Any, Any, Any] = (None, None, None)


class MingAudioDecoder(torch.nn.Module):
    """Chunked official-path AudioVAE decoder wrapper."""

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
    def decode_chunks(
        self,
        latents: torch.Tensor,
        last_chunks: list[bool],
        *,
        state: _MingAudioDecodeState | None = None,
    ) -> torch.Tensor:
        chunk_count = int(latents.shape[0])
        if len(last_chunks) != chunk_count:
            raise ValueError(
                "Ming-Omni-TTS AudioVAE decode requires one last_chunk flag per "
                f"latent chunk; got {len(last_chunks)} flags for {chunk_count} chunks"
            )
        if chunk_count == 0:
            return latents.new_empty((0,), dtype=torch.float32)
        if state is None and (not last_chunks[-1] or any(last_chunks[:-1])):
            raise ValueError(
                "Ming-Omni-TTS full AudioVAE decode requires exactly one terminal "
                "flag on the final latent chunk"
            )

        latents = latents.to(device=self.device, dtype=self.dtype)
        if state is None:
            state = _MingAudioDecodeState()
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
                wav, stream_state, past_key_values = self.audio_vae.decode(
                    chunk,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    stream_state=state.stream_state,
                    last_chunk=last_chunk,
                )
                state.stream_state = stream_state
                state.past_key_values = past_key_values
                wav = wav[0, 0].detach()
                if last_chunk and wav.numel() == 0:
                    raise RuntimeError(
                        "Ming-Omni-TTS AudioVAE terminal chunk produced no audio"
                    )
                waveform_chunks.append(wav)

        return torch.cat(waveform_chunks, dim=0)


@dataclass
class _MingTTSStreamState:
    decoder_state: _MingAudioDecodeState = field(default_factory=_MingAudioDecodeState)
    expected_chunk_id: int = 0
    pending_patch: torch.Tensor | None = None
    pending_is_last: bool = False
    terminal_patch_seen: bool = False
    emitted_samples: int = 0


class MingTTSStreamingVocoderScheduler(StreamingVocoderBase[_MingTTSStreamState, None]):
    """Decode Ming acoustic latents with request-local AudioVAE state."""

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        keep_latents: bool = False,
    ) -> None:
        self._decoder = decoder
        self._patch_size = int(patch_size)
        self._latent_dim = int(latent_dim)
        super().__init__(
            partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            ),
            sample_rate=decoder.sample_rate,
            stream_source_hint="Ming-Omni-TTS",
        )

    def create_stream_state(self, request_id: str) -> _MingTTSStreamState:
        del request_id
        return _MingTTSStreamState()

    def on_stream_chunk(
        self,
        request_id: str,
        item: StreamItem,
    ) -> list[OutgoingMessage]:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return []
        metadata = item.metadata
        if not isinstance(metadata, dict):
            return super().on_stream_chunk(request_id, item)
        if item.chunk_id != state.expected_chunk_id:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} has "
                f"chunk_id={item.chunk_id}, expected {state.expected_chunk_id}"
            )
        if state.terminal_patch_seen:
            raise RuntimeError(
                f"Ming-Omni-TTS stream chunk arrived after the terminal patch "
                f"for {request_id!r}"
            )
        is_last = metadata.get("is_last")
        if not isinstance(is_last, bool):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "boolean metadata['is_last']"
            )
        state.pending_is_last = is_last
        messages = super().on_stream_chunk(request_id, item)
        state.expected_chunk_id += 1
        return messages

    def validate_chunk(
        self,
        request_id: str,
        state: _MingTTSStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.dtype != torch.float32:
            raise TypeError(
                "Ming-Omni-TTS stream latent dtype must be torch.float32, "
                f"got {codes.dtype}"
            )
        expected_shape = (self._patch_size, self._latent_dim)
        if tuple(codes.shape) != expected_shape:
            raise ValueError(
                f"Ming-Omni-TTS stream latent shape must be {expected_shape}, "
                f"got {tuple(codes.shape)}"
            )
        return codes.to(
            device=self._decoder.device,
            dtype=self._decoder.dtype,
        ).contiguous()

    def ingest(
        self,
        request_id: str,
        state: _MingTTSStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patch = codes

    def decode_delta(
        self,
        request_id: str,
        state: _MingTTSStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if is_final:
            if not state.terminal_patch_seen:
                raise RuntimeError(
                    f"Ming-Omni-TTS stream for {request_id!r} ended without a "
                    "terminal latent patch"
                )
            return None

        patch = state.pending_patch
        is_last = state.pending_is_last
        waveform = self._decoder.decode_chunks(
            patch.unsqueeze(0),
            [is_last],
            state=state.decoder_state,
        )
        state.pending_patch = None
        state.pending_is_last = False
        if is_last:
            state.terminal_patch_seen = True
        if waveform.numel() == 0:
            return None
        state.emitted_samples += int(waveform.numel())
        return waveform

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _MingTTSStreamState,
    ) -> dict[str, Any]:
        del request_id
        final_state = load_ming_tts_state(payload)
        final_state.sample_rate = int(self._decoder.sample_rate)
        final_state.duration_s = float(
            state.emitted_samples / int(self._decoder.sample_rate)
        )
        data = final_state.to_dict()
        data["modality"] = "audio"
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    state = load_ming_tts_state(payload)
    latents = state.generated_latents
    if latents is not None:
        latents = latents.to(device=decoder.device, dtype=decoder.dtype)
    waveform = decoder.decode_chunks(
        latents,
        state.generated_last_chunk,
    )
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
    "MingTTSStreamingVocoderScheduler",
    "decode_ming_tts_audio_payload",
]
