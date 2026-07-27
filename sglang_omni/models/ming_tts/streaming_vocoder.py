# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduling for Ming-Omni-TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import torch

from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    MingAudioDecoderState,
    MingAudioVAEStepPhase,
    decode_ming_tts_audio_payload,
    decode_ming_tts_audio_payload_batch,
)
from sglang_omni.models.ming_tts.payload_types import load_ming_tts_state
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase


@dataclass
class _StreamState:
    decoder_state: MingAudioDecoderState = field(default_factory=MingAudioDecoderState)
    expected_chunk_id: int = 0
    pending_patches: list[torch.Tensor] = field(default_factory=list)
    terminal_received: bool = False
    initial_patch_consumed: bool = False
    terminal_decoded: bool = False
    emitted_samples: int = 0


@dataclass(frozen=True)
class _StepPlan:
    phase: MingAudioVAEStepPhase
    patch_count: int
    is_last: bool


class MingTTSStreamingVocoderScheduler(StreamingVocoderBase[_StreamState, _StepPlan]):
    """Decode Ming acoustic latents with request-local AudioVAE state."""

    _stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        steady_chunk_patches: int,
        keep_latents: bool = False,
    ) -> None:
        self._decoder = decoder
        self._patch_size = int(patch_size)
        self._latent_dim = int(latent_dim)
        self._steady_chunk_patches = int(steady_chunk_patches)
        self._can_batch_stream_chunks = True
        self._stream_chunk_batch_max = decoder.max_decode_batch_size
        batch_compute_fn = None
        if decoder.cuda_graph_enabled:
            batch_compute_fn = partial(
                decode_ming_tts_audio_payload_batch,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            )
        super().__init__(
            partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            ),
            sample_rate=decoder.sample_rate,
            stream_source_hint="Ming-Omni-TTS",
            batch_compute_fn=batch_compute_fn,
            max_batch_size=decoder.max_decode_batch_size,
        )

    def create_stream_state(self, request_id: str) -> _StreamState:
        del request_id
        return _StreamState()

    def on_serving_stop(self) -> None:
        self._decoder.log_cuda_graph_stats()

    def _ingest_stream_item(
        self,
        request_id: str,
        item: StreamItem,
    ) -> _StreamState | None:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return None
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "metadata"
            )
        if item.chunk_id != state.expected_chunk_id:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} has "
                f"chunk_id={item.chunk_id}, expected {state.expected_chunk_id}"
            )
        if state.terminal_received or state.terminal_decoded:
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
        super()._ingest_stream_item(request_id, item)
        state.expected_chunk_id += 1
        if is_last:
            state.terminal_received = True
        return state

    def validate_chunk(
        self,
        request_id: str,
        state: _StreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if codes.device.type != "cpu":
            raise ValueError(
                "Ming-Omni-TTS stream latent must be on CPU, "
                f"got device {codes.device}"
            )
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
        return codes.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _StreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patches.append(codes)

    def _step_plan_for_state(self, state: _StreamState) -> _StepPlan | None:
        if state.terminal_received:
            patch_count = len(state.pending_patches)
            is_last = True
        else:
            target = self._steady_chunk_patches if state.initial_patch_consumed else 1
            if len(state.pending_patches) < target:
                return None
            patch_count = target
            is_last = False

        return _StepPlan(
            phase=self._decoder.select_streaming_phase(
                state.decoder_state,
                is_last=is_last,
            ),
            patch_count=patch_count,
            is_last=is_last,
        )

    def select_step_participants(self) -> list[tuple[str, _StreamState]]:
        graph_participants = []
        for request_id, state in self._stream_state_items():
            plan = self._step_plan_for_state(state)
            if plan is None:
                continue
            participant = (request_id, state)
            if plan.phase is MingAudioVAEStepPhase.EAGER:
                return [participant]
            graph_participants.append(participant)
        return graph_participants[: self._decoder.max_decode_batch_size]

    def build_step_plan(
        self,
        participants: list[tuple[str, _StreamState]],
    ) -> _StepPlan:
        return cast(_StepPlan, self._step_plan_for_state(participants[0][1]))

    def run_step(
        self,
        participants: list[tuple[str, _StreamState]],
        plan: _StepPlan,
    ) -> dict[str, torch.Tensor]:
        latent_sequences = [
            torch.cat(state.pending_patches[: plan.patch_count], dim=0)
            for _, state in participants
        ]
        waveforms = self._decoder.decode_streaming_step(
            latent_sequences,
            [plan.is_last] * len(participants),
            [state.decoder_state for _, state in participants],
            phase=plan.phase,
        )

        decoded = {}
        for (request_id, state), waveform in zip(participants, waveforms):
            del state.pending_patches[: plan.patch_count]
            if not state.initial_patch_consumed and not plan.is_last:
                state.initial_patch_consumed = True
            if plan.is_last:
                state.terminal_received = False
                state.terminal_decoded = True
            if waveform.numel() > 0:
                state.emitted_samples += int(waveform.numel())
                decoded[request_id] = waveform
        return decoded

    def decode_delta(
        self,
        request_id: str,
        state: _StreamState,
        *,
        is_final: bool,
    ) -> None:
        del request_id, state, is_final

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _StreamState,
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


__all__ = ["MingTTSStreamingVocoderScheduler"]
