# SPDX-License-Identifier: Apache-2.0
"""Audio decode helpers for Ming-Omni-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
from transformers.cache_utils import Cache

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.audio_vae_graph import (
    MingAudioVAEFixedKVState,
    MingAudioVAEGraphRunner,
)
from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class MingAudioDecoderState:
    dynamic_cache: Cache | None = None
    fixed_kv_state: MingAudioVAEFixedKVState | None = None
    upsample_state: dict[str, Any] | None = None
    audio_buffer: torch.Tensor | None = None
    window_buffer: torch.Tensor | None = None


class MingAudioVAEStepPhase(Enum):
    EAGER = "eager"
    GRAPH = "graph"


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
    def max_decode_batch_size(self) -> int:
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

    def select_streaming_phase(
        self,
        state: MingAudioDecoderState,
        *,
        is_last: bool,
    ) -> MingAudioVAEStepPhase:
        if (
            self._graph_runner is not None
            and state.fixed_kv_state is not None
            and not is_last
        ):
            return MingAudioVAEStepPhase.GRAPH
        return MingAudioVAEStepPhase.EAGER

    @torch.inference_mode()
    def decode_streaming_step(
        self,
        latent_sequences: list[torch.Tensor],
        last_chunks: list[bool],
        states: list[MingAudioDecoderState],
        *,
        phase: MingAudioVAEStepPhase,
    ) -> list[torch.Tensor]:
        decoder = self.audio_vae.decoder
        qwen_inputs: list[torch.Tensor | None] = []
        next_upsample_states = []
        # Note (yzxiao): Linear upsampling carries overlap across chunks,
        # so each request must advance its own upsample state before batching.
        for latent_sequence, is_last, state in zip(
            latent_sequences,
            last_chunks,
            states,
        ):
            latent_sequence = latent_sequence.to(
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0)
            inputs, upsample_state = decoder.project_and_upsample_latents(
                latent_sequence,
                streaming=True,
                upsample_state=state.upsample_state,
                is_last=is_last,
            )
            qwen_inputs.append(inputs)
            next_upsample_states.append(upsample_state)

        context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            # Note (yzxiao): Only steady fixed-KV rows share graph replay;
            # initial and terminal rows stay eager to preserve cache semantics.
            next_dynamic_caches = [state.dynamic_cache for state in states]
            next_fixed_kv_states = [state.fixed_kv_state for state in states]
            if phase is MingAudioVAEStepPhase.GRAPH:
                graph_runner = cast(MingAudioVAEGraphRunner, self._graph_runner)
                graph_inputs = cast(list[torch.Tensor], qwen_inputs)
                fixed_kv_states = cast(
                    list[MingAudioVAEFixedKVState],
                    [state.fixed_kv_state for state in states],
                )
                result = graph_runner.replay_streaming(
                    graph_inputs,
                    fixed_kv_states,
                )
                qwen_hidden_states = [
                    value.unsqueeze(0) for value in result.hidden_states.unbind(dim=0)
                ]
                next_fixed_kv_states = list(result.states)
            else:
                inputs = qwen_inputs[0]
                if inputs is None:
                    qwen_hidden_states = [None]
                elif states[0].fixed_kv_state is not None:
                    graph_runner = cast(MingAudioVAEGraphRunner, self._graph_runner)
                    hidden_states_value, next_fixed_state = (
                        graph_runner.forward_streaming_eager(
                            inputs,
                            states[0].fixed_kv_state,
                        )
                    )
                    qwen_hidden_states = [hidden_states_value]
                    next_fixed_kv_states[0] = next_fixed_state
                else:
                    hidden_states_value, next_dynamic_cache = (
                        decoder.decode_qwen_hidden_states(
                            inputs,
                            past_key_values=states[0].dynamic_cache,
                            use_cache=True,
                        )
                    )
                    qwen_hidden_states = [hidden_states_value]
                    next_dynamic_caches[0] = next_dynamic_cache
                    if self._graph_runner is not None and not last_chunks[0]:
                        fixed_state = self._graph_runner.promote_dynamic_cache(
                            next_dynamic_cache
                        )
                        if fixed_state is not None:
                            next_dynamic_caches[0] = None
                            next_fixed_kv_states[0] = fixed_state

            # Note (yzxiao): Waveform overlap buffers remain request-local
            # even when the Qwen step is coalesced into one graph replay.
            waveforms = []
            next_audio_buffers = []
            next_window_buffers = []
            for row, hidden_states_value in enumerate(qwen_hidden_states):
                state = states[row]
                audio_buffer = state.audio_buffer
                window_buffer = state.window_buffer
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
                waveforms.append(waveform)
                next_audio_buffers.append(audio_buffer)
                next_window_buffers.append(window_buffer)

        # Note (yzxiao): Cache and overlap state must advance together with
        # the waveform returned by this scheduler step.
        for row, state in enumerate(states):
            state.dynamic_cache = next_dynamic_caches[row]
            state.fixed_kv_state = next_fixed_kv_states[row]
            state.upsample_state = next_upsample_states[row]
            state.audio_buffer = next_audio_buffers[row]
            state.window_buffer = next_window_buffers[row]
        return waveforms

    @torch.inference_mode()
    def decode_nonstreaming_batch(
        self,
        latent_batches: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not latent_batches:
            return []

        waveforms: dict[int, torch.Tensor] = {}
        sequences: dict[int, torch.Tensor] = {}
        for request_index, latents in enumerate(latent_batches):
            chunk_count = int(latents.shape[0])
            if chunk_count == 0:
                waveforms[request_index] = latents.new_empty((0,), dtype=torch.float32)
                continue
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
            prepared_inputs: dict[int, torch.Tensor] = {}
            for request_index, sequence in sequences.items():
                inputs, _ = self.audio_vae.decoder.project_and_upsample_latents(
                    sequence,
                    streaming=False,
                    upsample_state=None,
                    is_last=True,
                )
                prepared_inputs[request_index] = cast(torch.Tensor, inputs)

            hidden_states_by_request: dict[int, torch.Tensor] = {}
            if self._graph_runner is None:
                for request_index, inputs in prepared_inputs.items():
                    hidden_states, _ = self.audio_vae.decoder.decode_qwen_hidden_states(
                        inputs,
                        past_key_values=None,
                        use_cache=False,
                    )
                    hidden_states_by_request[request_index] = hidden_states
            else:
                groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
                for request_index, inputs in prepared_inputs.items():
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
                        hidden_states_by_request[request_index] = result.hidden_states[
                            row : row + 1, :true_length
                        ]

            for request_index, hidden_states in hidden_states_by_request.items():
                waveform, _, _ = self.audio_vae.decoder.synthesize_waveform(
                    hidden_states,
                    streaming=False,
                    audio_buffer=None,
                    window_buffer=None,
                    is_last=True,
                )
                waveforms[request_index] = waveform[0, 0].detach()

        return [waveforms[index] for index in range(len(latent_batches))]


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> StagePayload:
    """Adapt the scalar scheduler callback to the batched decode path."""

    return decode_ming_tts_audio_payload_batch(
        [payload],
        decoder,
        keep_latents=keep_latents,
    )[0]


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
    "MingAudioDecoderState",
    "MingAudioVAEStepPhase",
    "decode_ming_tts_audio_payload",
    "decode_ming_tts_audio_payload_batch",
]
