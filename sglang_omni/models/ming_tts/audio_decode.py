# SPDX-License-Identifier: Apache-2.0
"""Audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, cast

import torch
from transformers.cache_utils import Cache

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.audio_vae_graph import (
    MingAudioVAEGraphRunner,
    MingAudioVAEKVState,
)
from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _MingAudioDecodeState:
    dynamic_cache: Cache | None = None
    fixed_kv_state: MingAudioVAEKVState | None = None
    stream_state: tuple[Any, Any, Any] = (None, None, None)


class _MingAudioVAEStepPhase(Enum):
    EAGER = "eager"
    STEADY = "steady"


class MingAudioDecoder(torch.nn.Module):
    """Official-path AudioVAE decoder wrapper."""

    def __init__(self, audio_vae: AudioVAE, *, sample_rate: int) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.sample_rate = int(sample_rate)
        self._graph_runner: MingAudioVAEGraphRunner | None = None

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

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._graph_runner is not None

    @property
    def max_graph_batch_size(self) -> int:
        if self._graph_runner is None:
            return 1
        return self._graph_runner.max_batch_size

    @torch.inference_mode()
    def capture_cuda_graphs(
        self,
        *,
        batch_sizes: list[int],
        token_sizes: list[int],
        streaming_token_size: int,
    ) -> None:
        runner = MingAudioVAEGraphRunner(
            self.audio_vae.decoder,
            batch_sizes=batch_sizes,
            token_sizes=token_sizes,
            streaming_token_size=streaming_token_size,
            device=self.device,
            dtype=self.dtype,
        )
        runner.capture()
        self._graph_runner = runner

    def log_cuda_graph_stats(self) -> None:
        if self._graph_runner is not None:
            self._graph_runner.log_stats()

    def streaming_phase(
        self,
        state: _MingAudioDecodeState,
        *,
        is_last: bool,
    ) -> _MingAudioVAEStepPhase:
        if (
            self._graph_runner is not None
            and state.fixed_kv_state is not None
            and not is_last
        ):
            return _MingAudioVAEStepPhase.STEADY
        return _MingAudioVAEStepPhase.EAGER

    @torch.inference_mode()
    def decode_chunks(
        self,
        latents: torch.Tensor,
        last_chunks: list[bool],
        *,
        state: _MingAudioDecodeState | None = None,
    ) -> torch.Tensor:
        if state is None:
            return self.decode_nonstreaming_batch([latents], [last_chunks])[0]

        chunk_count = int(latents.shape[0])
        if len(last_chunks) != chunk_count:
            raise ValueError(
                "Ming-Omni-TTS AudioVAE decode requires one last_chunk flag per "
                f"latent chunk; got {len(last_chunks)} flags for {chunk_count} chunks"
            )
        if chunk_count == 0:
            return latents.new_empty((0,), dtype=torch.float32)

        waveform_chunks = []
        for step, last_chunk in enumerate(last_chunks):
            waveform = self.decode_streaming_step(
                [latents[step]],
                [last_chunk],
                [state],
                phase=self.streaming_phase(state, is_last=last_chunk),
            )[0]
            waveform_chunks.append(waveform)

        return torch.cat(waveform_chunks, dim=0)

    @torch.inference_mode()
    def decode_streaming_step(
        self,
        latent_sequences: list[torch.Tensor],
        last_chunks: list[bool],
        states: list[_MingAudioDecodeState],
        *,
        phase: _MingAudioVAEStepPhase,
    ) -> list[torch.Tensor]:
        if not latent_sequences or not (
            len(latent_sequences) == len(last_chunks) == len(states)
        ):
            raise ValueError(
                "Ming-Omni-TTS streaming AudioVAE step requires aligned "
                "latent, terminal-flag, and decoder-state rows"
            )

        decoder = self.audio_vae.decoder
        prepared_inputs: list[torch.Tensor | None] = []
        next_upsample_states = []
        for latent_sequence, is_last, state in zip(
            latent_sequences,
            last_chunks,
            states,
        ):
            latent_sequence = latent_sequence.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0)
            upsample_state, _, _ = state.stream_state
            inputs, upsample_state = decoder.prepare_inputs(
                latent_sequence,
                streaming=True,
                upsample_state=upsample_state,
                is_last=is_last,
            )
            prepared_inputs.append(inputs)
            next_upsample_states.append(upsample_state)

        context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            next_dynamic_caches = [state.dynamic_cache for state in states]
            next_fixed_states = [state.fixed_kv_state for state in states]
            if phase is _MingAudioVAEStepPhase.STEADY:
                graph_runner = cast(MingAudioVAEGraphRunner, self._graph_runner)
                graph_inputs: list[torch.Tensor] = []
                fixed_states: list[MingAudioVAEKVState] = []
                for inputs, state in zip(prepared_inputs, states):
                    graph_inputs.append(cast(torch.Tensor, inputs))
                    fixed_states.append(cast(MingAudioVAEKVState, state.fixed_kv_state))
                result = graph_runner.replay_streaming(
                    graph_inputs,
                    fixed_states,
                )
                hidden_states = [
                    value.unsqueeze(0) for value in result.hidden_states.unbind(dim=0)
                ]
                next_fixed_states = list(result.states)
            else:
                if len(states) != 1:
                    raise RuntimeError(
                        "Ming-Omni-TTS eager AudioVAE phase requires one request"
                    )
                inputs = prepared_inputs[0]
                if inputs is None:
                    hidden_states = [None]
                elif states[0].fixed_kv_state is not None:
                    graph_runner = cast(MingAudioVAEGraphRunner, self._graph_runner)
                    hidden_states_value, next_fixed_state = (
                        graph_runner.forward_streaming_eager(
                            inputs,
                            states[0].fixed_kv_state,
                        )
                    )
                    hidden_states = [hidden_states_value]
                    next_fixed_states[0] = next_fixed_state
                else:
                    hidden_states_value, next_dynamic_cache = (
                        decoder.decode_hidden_states(
                            inputs,
                            past_key_values=states[0].dynamic_cache,
                            use_cache=True,
                        )
                    )
                    hidden_states = [hidden_states_value]
                    next_dynamic_caches[0] = next_dynamic_cache
                    if self._graph_runner is not None and not last_chunks[0]:
                        fixed_state = self._graph_runner.promote_dynamic_cache(
                            next_dynamic_cache
                        )
                        if fixed_state is not None:
                            next_dynamic_caches[0] = None
                            next_fixed_states[0] = fixed_state

            waveforms = []
            next_stream_states = []
            for row, hidden_states_value in enumerate(hidden_states):
                _, audio_buffer, window_buffer = states[row].stream_state
                if hidden_states_value is None:
                    waveform = torch.empty(
                        (0,),
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    waveform, audio_buffer, window_buffer = decoder.synthesize_waveform(
                        hidden_states_value,
                        streaming=True,
                        audio_buffer=audio_buffer,
                        window_buffer=window_buffer,
                        is_last=last_chunks[row],
                    )
                    waveform = waveform[0, 0].detach()
                if last_chunks[row] and waveform.numel() == 0:
                    raise RuntimeError(
                        "Ming-Omni-TTS AudioVAE terminal chunk produced no audio"
                    )
                waveforms.append(waveform)
                next_stream_states.append(
                    (
                        next_upsample_states[row],
                        audio_buffer,
                        window_buffer,
                    )
                )

        for row, state in enumerate(states):
            state.dynamic_cache = next_dynamic_caches[row]
            state.fixed_kv_state = next_fixed_states[row]
            state.stream_state = next_stream_states[row]
        return waveforms

    @torch.inference_mode()
    def decode_nonstreaming_batch(
        self,
        latent_batches: list[torch.Tensor],
        last_chunk_batches: list[list[bool]],
    ) -> list[torch.Tensor]:
        if len(latent_batches) != len(last_chunk_batches):
            raise ValueError(
                "Ming-Omni-TTS AudioVAE batch requires one terminal-flag list "
                "per latent sequence"
            )
        if not latent_batches:
            return []

        waveforms: dict[int, torch.Tensor] = {}
        sequences: dict[int, torch.Tensor] = {}
        for request_index, (latents, last_chunks) in enumerate(
            zip(latent_batches, last_chunk_batches)
        ):
            chunk_count = int(latents.shape[0])
            if len(last_chunks) != chunk_count:
                raise ValueError(
                    "Ming-Omni-TTS AudioVAE decode requires one last_chunk flag "
                    f"per latent chunk; got {len(last_chunks)} flags for "
                    f"{chunk_count} chunks"
                )
            if chunk_count == 0:
                waveforms[request_index] = latents.new_empty((0,), dtype=torch.float32)
                continue
            if not last_chunks[-1] or any(last_chunks[:-1]):
                raise ValueError(
                    "Ming-Omni-TTS full AudioVAE decode requires exactly one "
                    "terminal flag on the final latent chunk"
                )
            latents = latents.to(device=self.device, dtype=self.dtype)
            sequences[request_index] = latents.reshape(1, -1, latents.shape[-1])

        if not sequences:
            return [waveforms[index] for index in range(len(latent_batches))]

        context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            if self._graph_runner is None:
                for request_index, sequence in sequences.items():
                    waveform, _, _ = self.audio_vae.decode(
                        sequence,
                        past_key_values=None,
                        use_cache=False,
                        stream_state=(None, None, None),
                        last_chunk=True,
                    )
                    waveform = waveform[0, 0].detach()
                    if waveform.numel() == 0:
                        raise RuntimeError(
                            "Ming-Omni-TTS AudioVAE terminal chunk produced no audio"
                        )
                    waveforms[request_index] = waveform
            else:
                groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
                for request_index, sequence in sequences.items():
                    inputs, _ = self.audio_vae.decoder.prepare_inputs(
                        sequence,
                        streaming=False,
                        upsample_state=None,
                        is_last=True,
                    )
                    token_size = self._graph_runner.select_token_size(
                        int(inputs.shape[1])
                    )
                    groups.setdefault(token_size, []).append((request_index, inputs))

                for token_size in sorted(groups):
                    group = groups[token_size]
                    result = self._graph_runner.replay(
                        [inputs for _, inputs in group],
                        token_size=token_size,
                    )
                    for row, (request_index, _) in enumerate(group):
                        true_length = result.true_lengths[row]
                        hidden_states = result.hidden_states[
                            row : row + 1, :true_length
                        ]
                        waveform, _, _ = self.audio_vae.decoder.synthesize_waveform(
                            hidden_states,
                            streaming=False,
                            audio_buffer=None,
                            window_buffer=None,
                            is_last=True,
                        )
                        waveform = waveform[0, 0].detach()
                        if waveform.numel() == 0:
                            raise RuntimeError(
                                "Ming-Omni-TTS AudioVAE terminal chunk produced "
                                "no audio"
                            )
                        waveforms[request_index] = waveform

        return [waveforms[index] for index in range(len(latent_batches))]


@dataclass
class _MingTTSStreamState:
    decoder_state: _MingAudioDecodeState = field(default_factory=_MingAudioDecodeState)
    expected_chunk_id: int = 0
    pending_patches: list[torch.Tensor] = field(default_factory=list)
    terminal_pending: bool = False
    cadence_primed: bool = False
    terminal_patch_seen: bool = False
    emitted_samples: int = 0


@dataclass(frozen=True)
class _MingAudioVAEStepPlan:
    phase: _MingAudioVAEStepPhase
    patch_count: int
    is_last: bool


class MingTTSStreamingVocoderScheduler(
    StreamingVocoderBase[_MingTTSStreamState, _MingAudioVAEStepPlan]
):
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
        self._stream_chunk_batch_max = decoder.max_graph_batch_size
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
            max_batch_size=decoder.max_graph_batch_size,
        )

    def create_stream_state(self, request_id: str) -> _MingTTSStreamState:
        del request_id
        return _MingTTSStreamState()

    def on_serving_stop(self) -> None:
        self._decoder.log_cuda_graph_stats()

    def _ingest_stream_item(
        self,
        request_id: str,
        item: StreamItem,
    ) -> _MingTTSStreamState | None:
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
        if state.terminal_pending or state.terminal_patch_seen:
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
            state.terminal_pending = True
        return state

    def validate_chunk(
        self,
        request_id: str,
        state: _MingTTSStreamState,
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
        state: _MingTTSStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patches.append(codes)

    def _step_plan_for_state(
        self,
        state: _MingTTSStreamState,
    ) -> _MingAudioVAEStepPlan | None:
        if state.terminal_pending:
            patch_count = len(state.pending_patches)
            is_last = True
        else:
            target = self._steady_chunk_patches if state.cadence_primed else 1
            if len(state.pending_patches) < target:
                return None
            patch_count = target
            is_last = False

        phase = self._decoder.streaming_phase(
            state.decoder_state,
            is_last=is_last,
        )
        return _MingAudioVAEStepPlan(
            phase=phase,
            patch_count=patch_count,
            is_last=is_last,
        )

    def select_step_participants(self) -> list[tuple[str, _MingTTSStreamState]]:
        steady_participants = []
        for request_id, state in self._stream_state_items():
            plan = self._step_plan_for_state(state)
            if plan is None:
                continue
            participant = (request_id, state)
            if plan.phase is _MingAudioVAEStepPhase.EAGER:
                return [participant]
            steady_participants.append(participant)
        return steady_participants[: self._decoder.max_graph_batch_size]

    def build_step_plan(
        self,
        participants: list[tuple[str, _MingTTSStreamState]],
    ) -> _MingAudioVAEStepPlan:
        _, state = participants[0]
        plan = self._step_plan_for_state(state)
        if plan is None:
            raise RuntimeError("Ming-Omni-TTS selected a stream before it was due")
        return plan

    def run_step(
        self,
        participants: list[tuple[str, _MingTTSStreamState]],
        plan: _MingAudioVAEStepPlan,
    ) -> dict[str, torch.Tensor]:
        latent_sequences = [
            torch.cat(state.pending_patches[: plan.patch_count], dim=0)
            for _, state in participants
        ]
        last_chunks = [plan.is_last] * len(participants)

        waveforms = self._decoder.decode_streaming_step(
            latent_sequences,
            last_chunks,
            [state.decoder_state for _, state in participants],
            phase=plan.phase,
        )
        decoded = {}
        for (request_id, state), waveform in zip(
            participants,
            waveforms,
        ):
            del state.pending_patches[: plan.patch_count]
            if not state.cadence_primed and not plan.is_last:
                state.cadence_primed = True
            if plan.is_last:
                state.terminal_pending = False
                state.terminal_patch_seen = True
            if waveform.numel() > 0:
                state.emitted_samples += int(waveform.numel())
                decoded[request_id] = waveform
        return decoded

    def decode_delta(
        self,
        request_id: str,
        state: _MingTTSStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if not is_final:
            raise RuntimeError(
                "Ming-Omni-TTS coalesced streaming decode must use run_step"
            )
        if not state.terminal_patch_seen:
            raise RuntimeError(
                f"Ming-Omni-TTS stream for {request_id!r} ended without a "
                "terminal latent patch"
            )
        if state.pending_patches or state.terminal_pending:
            raise RuntimeError(
                f"Ming-Omni-TTS stream for {request_id!r} ended with "
                "unconsumed latent patches"
            )
        return None

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
    waveform = decoder.decode_chunks(
        state.generated_latents,
        state.generated_last_chunk,
    )
    return _store_decoded_waveform(
        payload,
        state,
        waveform,
        decoder=decoder,
        keep_latents=keep_latents,
    )


def decode_ming_tts_audio_payload_batch(
    payloads: list[StagePayload],
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> list[StagePayload]:
    """Decode a non-streaming scheduler wave without changing payload order."""

    states = [load_ming_tts_state(payload) for payload in payloads]
    waveforms = decoder.decode_nonstreaming_batch(
        [state.generated_latents for state in states],
        [state.generated_last_chunk for state in states],
    )
    results = []
    for payload, state, waveform in zip(payloads, states, waveforms):
        results.append(
            _store_decoded_waveform(
                payload,
                state,
                waveform,
                decoder=decoder,
                keep_latents=keep_latents,
            )
        )
    return results


def _store_decoded_waveform(
    payload: StagePayload,
    state: MingTTSState,
    waveform: torch.Tensor,
    *,
    decoder: MingAudioDecoder,
    keep_latents: bool,
) -> StagePayload:
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
    "decode_ming_tts_audio_payload_batch",
]
