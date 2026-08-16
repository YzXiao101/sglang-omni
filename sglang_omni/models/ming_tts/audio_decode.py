# SPDX-License-Identifier: Apache-2.0
"""Audio decoding for Ming-Omni-TTS."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
    Decoder,
    StreamingLinearUpsample,
)
from sglang_omni.models.ming_tts.payload_types import (
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


class MingAudioDecoder:
    """Own the loaded AudioVAE and all audio-decode execution resources."""

    def __init__(
        self,
        audio_vae: AudioVAE,
        *,
        stream_capacity: int,
        max_stream_step_latents: int,
        streaming_cuda_graph_required: bool,
    ) -> None:
        self._audio_vae = audio_vae
        self._streaming_transition = _AudioVAEFixedStreamingTransition(
            audio_vae.decoder,
            capacity=stream_capacity,
            max_step_latents=max_stream_step_latents,
        )
        self._streaming_transition.assert_rows_clean()
        self._streaming_runner = _MingAudioStreamingRunner(
            self._streaming_transition,
            cuda_graph_required=streaming_cuda_graph_required,
        )

    @property
    def sample_rate(self) -> int:
        return int(self._audio_vae.config.sample_rate)

    @property
    def stream_capacity(self) -> int:
        return self._streaming_transition.capacity

    @property
    def streaming_ready(self) -> bool:
        return self._streaming_runner.is_ready

    def run_streaming(
        self,
        *,
        slot_ids: tuple[int, ...],
        patch_groups: tuple[tuple[torch.Tensor, ...], ...],
        terminal_flags: tuple[bool, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self._streaming_runner.run(
            slot_ids=slot_ids,
            patch_groups=patch_groups,
            terminal_flags=terminal_flags,
        )

    def reset_stream_rows(self, slot_ids: Sequence[int]) -> None:
        self._streaming_transition.reset_rows(slot_ids)

    def reset_all_stream_rows(self) -> None:
        self._streaming_transition.reset_all()

    def prepare_streaming(self) -> None:
        self._streaming_runner.prepare_cuda_graph()

    def close(self) -> None:
        self._streaming_runner.close()

    @torch.inference_mode()
    def decode_full(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if int(latents.shape[0]) == 0:
            return torch.empty((0,), dtype=torch.float32)

        first_parameter = next(self._audio_vae.parameters())
        device = first_parameter.device
        dtype = first_parameter.dtype
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context:
            latents = latents.to(device=device, dtype=dtype)
            sequence = latents.reshape(1, -1, latents.shape[-1])
            waveform, _, _ = self._audio_vae.decode(
                sequence,
                past_key_values=None,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )

        return waveform[0, 0].detach().to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class _AudioVAEFixedStreamingOutput:
    """Borrowed device output valid until the next decoder execution."""

    waveform: torch.Tensor
    sample_lengths: torch.Tensor


@dataclass(slots=True)
class _AudioVAEFixedStreamingStateBank:
    upsample_pending: torch.Tensor
    upsample_pending_lengths: torch.Tensor
    upsample_left_anchor: torch.Tensor
    qwen_keys: torch.Tensor
    qwen_values: torch.Tensor
    qwen_lengths: torch.Tensor
    qwen_positions: torch.Tensor
    istft_audio_overlap: torch.Tensor
    istft_window_overlap: torch.Tensor
    istft_started: torch.Tensor

    def slot_tensors(self) -> tuple[tuple[str, torch.Tensor, int], ...]:
        return (
            ("upsample_pending", self.upsample_pending, 0),
            ("upsample_pending_lengths", self.upsample_pending_lengths, 0),
            ("upsample_left_anchor", self.upsample_left_anchor, 0),
            ("qwen_keys", self.qwen_keys, 1),
            ("qwen_values", self.qwen_values, 1),
            ("qwen_lengths", self.qwen_lengths, 0),
            ("qwen_positions", self.qwen_positions, 0),
            ("istft_audio_overlap", self.istft_audio_overlap, 0),
            ("istft_window_overlap", self.istft_window_overlap, 0),
            ("istft_started", self.istft_started, 0),
        )


class _FixedQwenCacheLayer(CacheLayerMixin):
    is_sliding = True

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        sliding_window: int,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.values = values
        self.device = keys.device
        self.dtype = keys.dtype
        self.sliding_window = sliding_window
        self.cumulative_length = sliding_window - 1
        self.is_initialized = True
        self.new_keys: torch.Tensor | None = None
        self.new_values: torch.Tensor | None = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        del key_states, value_states
        raise RuntimeError("Fixed AudioVAE cache layers are initialized eagerly")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        self.new_keys = key_states
        self.new_values = value_states
        return (
            torch.cat((self.keys, key_states), dim=-2),
            torch.cat((self.values, value_states), dim=-2),
        )

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.sliding_window - 1 + query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sliding_window


class _AudioVAEFixedStreamingTransition:
    """Ming-private fixed-shape AudioVAE streaming state transition.

    This object must be created after the shared ``Decoder`` module reaches its
    final device and dtype. That module must not be moved or mutated during the
    transition's lifetime.
    CPU FP32 execution is available only for internal numerical verification;
    Ming-TTS serving remains CUDA BF16-only. ``decode`` accepts trusted
    fixed-envelope tensors and returns borrowed device tensors whose contents
    may be replaced by the next execution.
    """

    def __init__(
        self,
        decoder: Decoder,
        *,
        capacity: int,
        max_step_latents: int,
    ) -> None:
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or isinstance(max_step_latents, bool)
            or not isinstance(max_step_latents, Integral)
        ):
            raise TypeError("capacity and max_step_latents must be integers")
        if capacity <= 0 or max_step_latents <= 0:
            raise ValueError("capacity and max_step_latents must be positive")
        self._capacity = int(capacity)
        self._max_step_latents = int(max_step_latents)

        patch_size = int(decoder.patch_size)
        if patch_size <= 0:
            raise ValueError(
                "AudioVAE fixed streaming requires a positive decoder patch_size"
            )
        latent_dim = int(decoder.latent_dim)
        hop_length = int(decoder.hop_length)

        qwen = decoder.decoder
        config = qwen.config
        hidden_size = int(config.hidden_size)
        layer_count = len(qwen.layers)
        attention_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        sliding_window = getattr(config, "sliding_window", None)
        layer_types = tuple(getattr(config, "layer_types", ()))
        attention_backend = getattr(config, "_attn_implementation", None)
        if (
            sliding_window is None
            or sliding_window <= 1
            or len(layer_types) != layer_count
            or set(layer_types) != {"sliding_attention"}
            or attention_backend != "sdpa"
        ):
            raise ValueError(
                "AudioVAE fixed streaming requires sliding-only Qwen2 with "
                f"sliding_window > 1 and SDPA, got layer_types={layer_types!r}, "
                f"sliding_window={sliding_window!r}, backend={attention_backend!r}"
            )
        sliding_window = int(sliding_window)

        configured_head_dim = getattr(config, "head_dim", None)
        head_dim = int(
            hidden_size // attention_heads
            if configured_head_dim is None
            else configured_head_dim
        )
        upsampling = getattr(decoder, "upsampling", None)
        upsampler = getattr(upsampling, "upsampler", None)
        if (
            type(upsampling) is not StreamingLinearUpsample
            or upsampling.scale_factor != patch_size
            or type(upsampler) is not nn.Upsample
            or upsampler.mode != "linear"
            or upsampler.align_corners is not False
            or upsampler.size is not None
            or upsampler.scale_factor != patch_size
            or upsampler.recompute_scale_factor is not None
        ):
            raise ValueError(
                "AudioVAE fixed streaming requires exact "
                "StreamingLinearUpsample(nn.Upsample) semantics matching "
                f"decoder patch_size={patch_size}; got "
                f"wrapper={type(upsampling).__name__}, "
                f"wrapper_scale={getattr(upsampling, 'scale_factor', None)!r}, "
                f"inner={type(upsampler).__name__}, "
                f"mode={getattr(upsampler, 'mode', None)!r}, "
                f"align_corners={getattr(upsampler, 'align_corners', None)!r}, "
                f"size={getattr(upsampler, 'size', None)!r}, "
                f"inner_scale={getattr(upsampler, 'scale_factor', None)!r}, "
                "recompute_scale_factor="
                f"{getattr(upsampler, 'recompute_scale_factor', None)!r}"
            )

        istft = decoder.head.istft
        win_length = int(istft.win_length)
        overlap = win_length - hop_length
        if overlap <= 0 or overlap % 2 != 0 or istft.padding != "same":
            raise ValueError(
                "AudioVAE fixed streaming requires same-padding ISTFT with an "
                "even positive overlap"
            )

        first_parameter = next(decoder.parameters())
        device = first_parameter.device
        input_dtype = first_parameter.dtype
        if decoder.training or not (
            (device.type == "cuda" and input_dtype == torch.bfloat16)
            or (device.type == "cpu" and input_dtype == torch.float32)
        ):
            raise ValueError(
                "AudioVAE fixed streaming requires an eval-mode CUDA BF16 decoder "
                "for serving or an eval-mode CPU FP32 decoder for internal "
                f"verification, got device={device}, dtype={input_dtype}, "
                f"training={decoder.training}"
            )

        self._latent_dim = latent_dim
        self._max_output_samples = (
            2 * self.max_step_latents * patch_size * hop_length + overlap - overlap // 2
        )
        self._device = device
        self._input_dtype = input_dtype

        self._decoder = decoder
        self._upsampler = upsampler
        self._scale_factor = patch_size
        self._max_frames = 2 * self.max_step_latents * patch_size
        self._hidden_size = hidden_size
        self._sliding_window = sliding_window
        self._cache_size = sliding_window - 1
        self._hop_length = hop_length
        self._overlap = overlap
        self._pad = overlap // 2
        self._max_raw_samples = self._max_frames * hop_length + overlap

        kv_shape = (
            layer_count,
            self.capacity,
            kv_heads,
            self._cache_size,
            head_dim,
        )
        self._state = _AudioVAEFixedStreamingStateBank(
            upsample_pending=torch.zeros(
                (
                    self.capacity,
                    self.max_step_latents,
                    hidden_size,
                ),
                device=device,
                dtype=input_dtype,
            ),
            upsample_pending_lengths=torch.zeros(
                self.capacity, device=device, dtype=torch.long
            ),
            upsample_left_anchor=torch.zeros(
                (self.capacity, 1, hidden_size),
                device=device,
                dtype=input_dtype,
            ),
            qwen_keys=torch.zeros(kv_shape, device=device, dtype=input_dtype),
            qwen_values=torch.zeros(kv_shape, device=device, dtype=input_dtype),
            qwen_lengths=torch.zeros(self.capacity, device=device, dtype=torch.long),
            qwen_positions=torch.zeros(self.capacity, device=device, dtype=torch.long),
            istft_audio_overlap=torch.zeros(
                (self.capacity, overlap), device=device, dtype=torch.float32
            ),
            istft_window_overlap=torch.zeros(
                (self.capacity, overlap), device=device, dtype=torch.float32
            ),
            istft_started=torch.zeros(self.capacity, device=device, dtype=torch.bool),
        )
        self._state_nbytes = sum(
            tensor.numel() * tensor.element_size()
            for _, tensor, _ in self._state.slot_tensors()
        )
        state_field_bytes = {
            name: tensor.numel() * tensor.element_size()
            for name, tensor, _ in self._state.slot_tensors()
        }
        logger.info(
            "ming_tts_audio_vae_state stage=audio_decode capacity=%d "
            "max_step_latents=%d state_device_bytes=%d state_field_bytes=%s",
            self.capacity,
            self.max_step_latents,
            self.state_nbytes,
            state_field_bytes,
        )

        self._cache_layers = [
            _FixedQwenCacheLayer(
                self._state.qwen_keys[index],
                self._state.qwen_values[index],
                sliding_window=sliding_window,
            )
            for index in range(layer_count)
        ]
        self._cache = Cache(layers=self._cache_layers)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def max_step_latents(self) -> int:
        return self._max_step_latents

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def max_output_samples(self) -> int:
        return self._max_output_samples

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def input_dtype(self) -> torch.dtype:
        return self._input_dtype

    @property
    def state_nbytes(self) -> int:
        return self._state_nbytes

    def decode(
        self,
        latents: torch.Tensor,
        latent_lengths: torch.Tensor,
        exec_mask: torch.Tensor,
        terminal_mask: torch.Tensor,
    ) -> _AudioVAEFixedStreamingOutput:
        """Advance every active slot by one trusted fixed-envelope step."""
        execution_context = (
            torch.autocast(device_type="cuda", dtype=self.input_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with (
            torch.inference_mode(),
            execution_context,
        ):
            projected = self._decoder.fc1(latents)
            (
                frames,
                frame_lengths,
                (
                    next_upsample_pending,
                    next_upsample_pending_lengths,
                    next_upsample_left_anchor,
                ),
            ) = self._upsample(
                projected,
                latent_lengths,
                exec_mask,
                terminal_mask,
            )
            qwen_exec = exec_mask & (frame_lengths > 0)
            hidden, (
                next_qwen_keys,
                next_qwen_values,
                next_qwen_lengths,
                next_qwen_positions,
            ) = self._qwen(frames, frame_lengths)
            (
                waveform,
                sample_lengths,
                (
                    next_istft_audio_overlap,
                    next_istft_window_overlap,
                    next_istft_started,
                ),
            ) = self._istft(
                hidden,
                frame_lengths,
                qwen_exec,
                terminal_mask,
            )
            next_state = _AudioVAEFixedStreamingStateBank(
                upsample_pending=next_upsample_pending,
                upsample_pending_lengths=next_upsample_pending_lengths,
                upsample_left_anchor=next_upsample_left_anchor,
                qwen_keys=next_qwen_keys,
                qwen_values=next_qwen_values,
                qwen_lengths=next_qwen_lengths,
                qwen_positions=next_qwen_positions,
                istft_audio_overlap=next_istft_audio_overlap,
                istft_window_overlap=next_istft_window_overlap,
                istft_started=next_istft_started,
            )
            self._commit(exec_mask, qwen_exec, terminal_mask, next_state)
        return _AudioVAEFixedStreamingOutput(
            waveform=waveform,
            sample_lengths=sample_lengths,
        )

    def reset_rows(self, slot_ids: Sequence[int]) -> None:
        slots = self._validate_slot_ids(slot_ids)
        if not slots:
            return

        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            indices = torch.tensor(slots, device=self.device, dtype=torch.long)
            for _, tensor, row_dim in self._state.slot_tensors():
                tensor.index_fill_(row_dim, indices, 0)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()

    def reset_all(self) -> None:
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            for _, tensor, _ in self._state.slot_tensors():
                tensor.zero_()
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()

    def assert_rows_clean(self, slot_ids: Sequence[int] | None = None) -> None:
        slots = None if slot_ids is None else self._validate_slot_ids(slot_ids)
        if slots == ():
            return

        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            indices = (
                None
                if slots is None
                else torch.tensor(slots, device=self.device, dtype=torch.long)
            )
            slot_tensors = self._state.slot_tensors()
            dirty_flag_tensors = []
            for _, tensor, row_dim in slot_tensors:
                if indices is not None:
                    tensor = tensor.index_select(row_dim, indices)
                dirty_flag_tensors.append(torch.count_nonzero(tensor) > 0)
            dirty_flags = torch.stack(dirty_flag_tensors)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            dirty_names = tuple(
                name
                for (name, _, _), is_dirty in zip(slot_tensors, dirty_flags.tolist())
                if is_dirty
            )

        if dirty_names:
            raise RuntimeError(
                "AudioVAE streaming rows are not clean: " + ", ".join(dirty_names)
            )

    def _validate_slot_ids(self, slot_ids: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(slot_ids, Sequence):
            raise TypeError("slot_ids must be a host sequence of integers")

        slots: list[int] = []
        for slot_id in slot_ids:
            if isinstance(slot_id, bool) or not isinstance(slot_id, Integral):
                raise TypeError("slot_ids must contain non-bool integers")
            slot = int(slot_id)
            if slot < 0 or slot >= self.capacity:
                raise ValueError(f"slot_id {slot} is outside [0, {self.capacity})")
            slots.append(slot)
        if len(set(slots)) != len(slots):
            raise ValueError("slot_ids must not contain duplicates")
        return tuple(slots)

    def _upsample(
        self,
        current: torch.Tensor,
        current_lengths: torch.Tensor,
        exec_mask: torch.Tensor,
        terminal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        state = self._state
        row = torch.arange(self.capacity, device=self.device)
        latent_index = torch.arange(
            self.max_step_latents, device=self.device
        ).unsqueeze(0)
        has_pending = state.upsample_pending_lengths > 0
        left_anchor = torch.where(
            has_pending.reshape(self.capacity, 1, 1),
            state.upsample_left_anchor,
            current[:, :1],
        )

        timeline = torch.cat(
            (
                left_anchor,
                state.upsample_pending,
                torch.zeros(
                    (
                        self.capacity,
                        self.max_step_latents + 1,
                        self._hidden_size,
                    ),
                    device=self.device,
                    dtype=self.input_dtype,
                ),
            ),
            dim=1,
        )
        current_destination = 1 + state.upsample_pending_lengths.unsqueeze(1)
        current_destination = current_destination + latent_index
        timeline.scatter_(
            1,
            current_destination.unsqueeze(2).expand(-1, -1, self._hidden_size),
            current,
        )

        current_last_index = torch.clamp(current_lengths - 1, min=0)
        current_last = torch.gather(
            current,
            1,
            current_last_index.reshape(self.capacity, 1, 1).expand(
                -1, -1, self._hidden_size
            ),
        )
        right_boundary_destination = (
            1 + state.upsample_pending_lengths + current_lengths
        )
        timeline.scatter_(
            1,
            right_boundary_destination.reshape(self.capacity, 1, 1).expand(
                -1, -1, self._hidden_size
            ),
            current_last,
        )

        upsampled = self._upsampler(timeline.transpose(1, 2)).transpose(1, 2)
        frames = upsampled[
            :,
            self._scale_factor : self._scale_factor + self._max_frames,
        ].to(torch.float32)
        frame_lengths = (
            state.upsample_pending_lengths + terminal_mask * current_lengths
        ) * self._scale_factor
        frame_lengths = torch.where(exec_mask, frame_lengths, 0)
        valid_frames = torch.arange(self._max_frames, device=self.device).unsqueeze(
            0
        ) < frame_lengths.unsqueeze(1)
        frames = torch.where(
            valid_frames.unsqueeze(2),
            frames,
            0,
        )

        valid_current = latent_index < current_lengths.unsqueeze(1)
        next_pending = torch.where(
            valid_current.unsqueeze(2),
            current,
            0,
        )
        next_left_anchor = timeline[row, state.upsample_pending_lengths].unsqueeze(1)
        return (
            frames,
            frame_lengths,
            (
                next_pending,
                current_lengths,
                next_left_anchor,
            ),
        )

    def _qwen(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        state = self._state
        position_ids = state.qwen_positions.unsqueeze(1) + torch.arange(
            self._max_frames, device=self.device
        ).unsqueeze(0)
        attention_mask = self._attention_mask(input_lengths)
        outputs = self._decoder.decoder(
            inputs_embeds=inputs,
            attention_mask={"sliding_attention": attention_mask},
            position_ids=position_ids,
            past_key_values=self._cache,
            use_cache=True,
        )
        new_keys = torch.stack(
            [cast(torch.Tensor, layer.new_keys) for layer in self._cache_layers]
        )
        new_values = torch.stack(
            [cast(torch.Tensor, layer.new_values) for layer in self._cache_layers]
        )
        next_keys = self._advance_kv(state.qwen_keys, new_keys, input_lengths).to(
            state.qwen_keys.dtype
        )
        next_values = self._advance_kv(state.qwen_values, new_values, input_lengths).to(
            state.qwen_values.dtype
        )
        next_lengths = torch.clamp(
            state.qwen_lengths + input_lengths, max=self._cache_size
        )
        next_positions = state.qwen_positions + input_lengths
        return outputs.last_hidden_state, (
            next_keys,
            next_values,
            next_lengths,
            next_positions,
        )

    def _attention_mask(self, input_lengths: torch.Tensor) -> torch.Tensor:
        query = torch.arange(self._max_frames, device=self.device).reshape(
            1, self._max_frames, 1
        )
        key = torch.arange(
            -self._cache_size,
            self._max_frames,
            device=self.device,
        ).reshape(1, 1, self._cache_size + self._max_frames)
        causal = (key <= query) & (key > query - self._sliding_window)

        past_slot = torch.arange(self._cache_size, device=self.device).unsqueeze(0)
        valid_past = past_slot >= (
            self._cache_size - self._state.qwen_lengths
        ).unsqueeze(1)
        current_slot = torch.arange(self._max_frames, device=self.device).unsqueeze(0)
        valid_current = current_slot < input_lengths.unsqueeze(1)
        valid_keys = torch.cat((valid_past, valid_current), dim=1)
        allowed = causal & valid_keys.unsqueeze(1)

        valid_queries = current_slot < input_lengths.unsqueeze(1)
        safe_invalid = F.pad(
            torch.eye(
                self._max_frames,
                device=self.device,
                dtype=torch.bool,
            ),
            (self._cache_size, 0),
        ).unsqueeze(0)
        allowed = allowed | (safe_invalid & ~valid_queries.unsqueeze(2))
        mask = torch.full(
            allowed.shape,
            torch.finfo(self.input_dtype).min,
            device=self.device,
            dtype=self.input_dtype,
        )
        mask.masked_fill_(allowed, 0)
        return mask.unsqueeze(1)

    def _advance_kv(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        current_lengths: torch.Tensor,
    ) -> torch.Tensor:
        layer_count, batch_size, heads, _, head_dim = previous.shape
        full = torch.cat((previous, current), dim=3)
        total = self._state.qwen_lengths + current_lengths
        next_lengths = torch.clamp(total, max=self._cache_size)
        destination = torch.arange(self._cache_size, device=self.device).unsqueeze(0)
        logical = total.unsqueeze(1) - self._cache_size + destination
        from_previous = logical < self._state.qwen_lengths.unsqueeze(1)
        previous_source = (
            self._cache_size - self._state.qwen_lengths.unsqueeze(1) + logical
        )
        current_source = (
            self._cache_size + logical - self._state.qwen_lengths.unsqueeze(1)
        )
        source = torch.where(from_previous, previous_source, current_source)
        source = torch.clamp(
            source,
            min=0,
            max=self._cache_size + self._max_frames - 1,
        )
        gather_index = source.reshape(1, batch_size, 1, self._cache_size, 1).expand(
            layer_count, -1, heads, -1, head_dim
        )
        gathered = torch.gather(full, 3, gather_index)
        valid_destination = destination >= (self._cache_size - next_lengths).unsqueeze(
            1
        )
        return gathered * valid_destination.reshape(
            1,
            batch_size,
            1,
            self._cache_size,
            1,
        )

    def _istft(
        self,
        hidden: torch.Tensor,
        frame_lengths: torch.Tensor,
        exec_mask: torch.Tensor,
        terminal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        head = self._decoder.head
        spectrum, _ = head._predict_spectrum(hidden)
        frame_mask = torch.arange(self._max_frames, device=self.device).unsqueeze(
            0
        ) < frame_lengths.unsqueeze(1)
        numerator, denominator = head.istft._overlap_add(
            spectrum, valid_frame_mask=frame_mask
        )

        started = self._state.istft_started & exec_mask
        numerator[:, : self._overlap].add_(
            self._state.istft_audio_overlap * started.unsqueeze(1)
        )
        denominator[:, : self._overlap].add_(
            self._state.istft_window_overlap * started.unsqueeze(1)
        )

        raw_lengths = frame_lengths * self._hop_length + self._overlap
        buffer_start = torch.clamp(raw_lengths - self._overlap, min=0)
        buffer_index = buffer_start.unsqueeze(1) + torch.arange(
            self._overlap, device=self.device
        ).unsqueeze(0)
        next_audio_overlap = torch.gather(numerator, 1, buffer_index)
        next_window_overlap = torch.gather(denominator, 1, buffer_index)

        safe_denominator = torch.where(denominator > 1e-11, denominator, 1)
        normalized = numerator / safe_denominator
        output_start = torch.where(started, 0, self._pad)
        right_trim = torch.where(terminal_mask, self._pad, self._overlap)
        output_end = torch.clamp(raw_lengths - right_trim, min=0)
        sample_lengths = torch.clamp(output_end - output_start, min=0)
        sample_lengths = torch.where(exec_mask, sample_lengths, 0)
        output_index = output_start.unsqueeze(1) + torch.arange(
            self.max_output_samples, device=self.device
        ).unsqueeze(0)
        output_index = torch.clamp(output_index, max=self._max_raw_samples - 1)
        waveform = torch.gather(normalized, 1, output_index)
        output_mask = torch.arange(
            self.max_output_samples, device=self.device
        ).unsqueeze(0) < sample_lengths.unsqueeze(1)
        waveform = waveform * output_mask
        return (
            waveform,
            sample_lengths,
            (
                next_audio_overlap,
                next_window_overlap,
                self._state.istft_started | exec_mask,
            ),
        )

    def _commit(
        self,
        exec_mask: torch.Tensor,
        qwen_exec: torch.Tensor,
        terminal_mask: torch.Tensor,
        next_state: _AudioVAEFixedStreamingStateBank,
    ) -> None:
        state = self._state
        upsample_alive = exec_mask & ~terminal_mask
        decoded_alive = qwen_exec & ~terminal_mask
        terminal = exec_mask & terminal_mask

        def commit_field(
            destination: torch.Tensor,
            proposal: torch.Tensor,
            alive_selector: torch.Tensor,
            terminal_selector: torch.Tensor,
        ) -> None:
            torch.where(
                alive_selector,
                proposal,
                destination,
                out=destination,
            )
            destination.masked_fill_(terminal_selector, 0)

        upsample_selector = upsample_alive.reshape(-1, 1, 1)
        upsample_terminal = terminal.reshape(-1, 1, 1)
        commit_field(
            state.upsample_pending,
            next_state.upsample_pending,
            upsample_selector,
            upsample_terminal,
        )
        commit_field(
            state.upsample_pending_lengths,
            next_state.upsample_pending_lengths,
            upsample_alive,
            terminal,
        )
        commit_field(
            state.upsample_left_anchor,
            next_state.upsample_left_anchor,
            upsample_selector,
            upsample_terminal,
        )

        qwen_selector = decoded_alive.reshape(1, self.capacity, 1, 1, 1)
        qwen_terminal = terminal.reshape(1, self.capacity, 1, 1, 1)
        commit_field(
            state.qwen_keys,
            next_state.qwen_keys,
            qwen_selector,
            qwen_terminal,
        )
        commit_field(
            state.qwen_values,
            next_state.qwen_values,
            qwen_selector,
            qwen_terminal,
        )
        commit_field(
            state.qwen_lengths,
            next_state.qwen_lengths,
            decoded_alive,
            terminal,
        )
        commit_field(
            state.qwen_positions,
            next_state.qwen_positions,
            decoded_alive,
            terminal,
        )

        decoded_selector = decoded_alive.unsqueeze(1)
        decoded_terminal = terminal.unsqueeze(1)
        commit_field(
            state.istft_audio_overlap,
            next_state.istft_audio_overlap,
            decoded_selector,
            decoded_terminal,
        )
        commit_field(
            state.istft_window_overlap,
            next_state.istft_window_overlap,
            decoded_selector,
            decoded_terminal,
        )
        commit_field(
            state.istft_started,
            next_state.istft_started,
            decoded_alive,
            terminal,
        )


@dataclass(frozen=True, slots=True)
class _CapturedAudioVAEGraph:
    graph: torch.cuda.CUDAGraph
    output: _AudioVAEFixedStreamingOutput


class _MingAudioStreamingRunner:
    """Own fixed AudioVAE I/O and materialize each device result on CPU."""

    _CUDA_GRAPH_WARMUP_ITERATIONS = 3

    def __init__(
        self,
        transition: _AudioVAEFixedStreamingTransition,
        *,
        cuda_graph_required: bool,
    ) -> None:
        self._transition = transition
        self._cuda_graph_required = cuda_graph_required
        self._captured_graph: _CapturedAudioVAEGraph | None = None
        capacity = transition.capacity
        max_step_latents = transition.max_step_latents
        latent_dim = transition.latent_dim

        self._host_latents = torch.empty(
            (capacity, max_step_latents, latent_dim),
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )
        self._host_latent_lengths = torch.empty(
            capacity,
            device="cpu",
            dtype=torch.long,
            pin_memory=True,
        )
        self._host_exec_mask = torch.empty(
            capacity,
            device="cpu",
            dtype=torch.bool,
            pin_memory=True,
        )
        self._host_terminal_mask = torch.empty(
            capacity,
            device="cpu",
            dtype=torch.bool,
            pin_memory=True,
        )

        with torch.cuda.device(transition.device):
            self._latents = torch.empty(
                (capacity, max_step_latents, latent_dim),
                device=transition.device,
                dtype=transition.input_dtype,
            )
            self._latent_lengths = torch.empty(
                capacity,
                device=transition.device,
                dtype=torch.long,
            )
            self._exec_mask = torch.empty(
                capacity,
                device=transition.device,
                dtype=torch.bool,
            )
            self._terminal_mask = torch.empty(
                capacity,
                device=transition.device,
                dtype=torch.bool,
            )

        self._host_waveform = torch.empty(
            (capacity, transition.max_output_samples),
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )
        self._host_sample_lengths = torch.empty(
            capacity,
            device="cpu",
            dtype=torch.long,
            pin_memory=True,
        )
        static_device_input_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._latents,
                self._latent_lengths,
                self._exec_mask,
                self._terminal_mask,
            )
        )
        pinned_host_io_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._host_latents,
                self._host_latent_lengths,
                self._host_exec_mask,
                self._host_terminal_mask,
                self._host_waveform,
                self._host_sample_lengths,
            )
        )
        logger.info(
            "ming_tts_audio_vae_runner stage=audio_decode "
            "streaming_backend=%s streaming_cuda_graph_required=%s "
            "streaming_graph_ready=%s static_device_input_bytes=%d "
            "pinned_host_io_bytes=%d",
            "cuda_graph" if self._cuda_graph_required else "eager",
            self._cuda_graph_required,
            self.is_ready,
            static_device_input_bytes,
            pinned_host_io_bytes,
        )

    @property
    def is_ready(self) -> bool:
        return not self._cuda_graph_required or self._captured_graph is not None

    def run(
        self,
        *,
        slot_ids: tuple[int, ...],
        patch_groups: tuple[tuple[torch.Tensor, ...], ...],
        terminal_flags: tuple[bool, ...],
    ) -> tuple[torch.Tensor, ...]:
        captured = self._captured_graph
        if self._cuda_graph_required and captured is None:
            raise RuntimeError(
                "required Ming-Omni-TTS streaming AudioVAE CUDA graph is not prepared"
            )

        self._host_latents.zero_()
        self._host_latent_lengths.zero_()
        self._host_exec_mask.zero_()
        self._host_terminal_mask.zero_()

        for slot, patches, terminal in zip(
            slot_ids,
            patch_groups,
            terminal_flags,
            strict=True,
        ):
            offset = 0
            for patch in patches:
                end = offset + int(patch.shape[0])
                self._host_latents[slot, offset:end].copy_(patch)
                offset = end
            self._host_latent_lengths[slot] = offset
            self._host_exec_mask[slot] = True
            self._host_terminal_mask[slot] = terminal

        with torch.cuda.device(self._transition.device):
            self._latents.copy_(self._host_latents, non_blocking=True)
            self._latent_lengths.copy_(
                self._host_latent_lengths,
                non_blocking=True,
            )
            self._exec_mask.copy_(self._host_exec_mask, non_blocking=True)
            self._terminal_mask.copy_(
                self._host_terminal_mask,
                non_blocking=True,
            )

            if self._cuda_graph_required:
                captured.graph.replay()
                output = captured.output
            else:
                output = self._execute_device()
            self._host_waveform.copy_(output.waveform, non_blocking=True)
            self._host_sample_lengths.copy_(
                output.sample_lengths,
                non_blocking=True,
            )
            torch.cuda.current_stream(self._transition.device).synchronize()

        waveforms = []
        for slot in slot_ids:
            sample_count = int(self._host_sample_lengths[slot])
            if sample_count < 0 or sample_count > self._transition.max_output_samples:
                raise RuntimeError(
                    "AudioVAE fixed streaming returned invalid sample length "
                    f"{sample_count} for slot {slot}"
                )
            waveforms.append(self._host_waveform[slot, :sample_count].clone())
        return tuple(waveforms)

    def _execute_device(self) -> _AudioVAEFixedStreamingOutput:
        return self._transition.decode(
            self._latents,
            latent_lengths=self._latent_lengths,
            exec_mask=self._exec_mask,
            terminal_mask=self._terminal_mask,
        )

    def prepare_cuda_graph(self) -> None:
        if not self._cuda_graph_required:
            return
        if self._captured_graph is not None:
            raise RuntimeError(
                "Ming-Omni-TTS streaming AudioVAE CUDA graph is already prepared"
            )

        candidate_graph: torch.cuda.CUDAGraph | None = None
        try:
            with torch.cuda.device(self._transition.device):
                torch.cuda.synchronize(self._transition.device)
                allocated_before = int(
                    torch.cuda.memory_allocated(self._transition.device)
                )
                reserved_before = int(
                    torch.cuda.memory_reserved(self._transition.device)
                )
                self._reset_transition()

                self._latents.zero_()
                self._latent_lengths.fill_(self._transition.max_step_latents)
                self._exec_mask.fill_(True)
                self._terminal_mask.fill_(True)

                current_stream = torch.cuda.current_stream(self._transition.device)
                build_stream = torch.cuda.Stream(device=self._transition.device)
                build_stream.wait_stream(current_stream)
                with torch.cuda.stream(build_stream):
                    for _ in range(self._CUDA_GRAPH_WARMUP_ITERATIONS):
                        warm_output = self._execute_device()
                current_stream.wait_stream(build_stream)
                current_stream.synchronize()
                del warm_output
                self._reset_transition()

                candidate_graph = torch.cuda.CUDAGraph()
                build_stream.wait_stream(current_stream)
                try:
                    with torch.cuda.graph(
                        candidate_graph,
                        stream=build_stream,
                        capture_error_mode="thread_local",
                    ):
                        candidate_output = self._execute_device()
                finally:
                    torch.cuda.set_stream(current_stream)
                current_stream.wait_stream(build_stream)
                current_stream.synchronize()
                self._require_output_contract(candidate_output)
                output_ptrs = (
                    candidate_output.waveform.data_ptr(),
                    candidate_output.sample_lengths.data_ptr(),
                )
                self._reset_transition()

                candidate_graph.replay()
                current_stream.synchronize()
                self._require_output_contract(candidate_output)
                replay_output_ptrs = (
                    candidate_output.waveform.data_ptr(),
                    candidate_output.sample_lengths.data_ptr(),
                )
                if replay_output_ptrs != output_ptrs:
                    raise RuntimeError(
                        "Ming-Omni-TTS streaming AudioVAE CUDA graph output "
                        "addresses changed"
                    )
                self._reset_transition()
                allocated_after = int(
                    torch.cuda.memory_allocated(self._transition.device)
                )
                reserved_after = int(
                    torch.cuda.memory_reserved(self._transition.device)
                )

                self._captured_graph = _CapturedAudioVAEGraph(
                    graph=candidate_graph,
                    output=candidate_output,
                )
        except Exception:
            if candidate_graph is not None:
                try:
                    candidate_graph.reset()
                except Exception:
                    logger.exception(
                        "Failed to reset an unpublished Ming-Omni-TTS "
                        "streaming AudioVAE CUDA graph"
                    )
            raise
        logger.info(
            "ming_tts_audio_vae_streaming_graph stage=audio_decode "
            "streaming_graph_ready=true "
            "allocator_allocated_before_bytes=%d "
            "allocator_allocated_after_bytes=%d "
            "allocator_allocated_delta_bytes=%d "
            "allocator_reserved_before_bytes=%d "
            "allocator_reserved_after_bytes=%d "
            "allocator_reserved_delta_bytes=%d",
            allocated_before,
            allocated_after,
            allocated_after - allocated_before,
            reserved_before,
            reserved_after,
            reserved_after - reserved_before,
        )

    def _reset_transition(self) -> None:
        self._transition.reset_all()
        self._transition.assert_rows_clean()

    def _require_output_contract(
        self,
        output: _AudioVAEFixedStreamingOutput,
    ) -> None:
        if type(output) is not _AudioVAEFixedStreamingOutput:
            raise RuntimeError(
                "Ming-Omni-TTS streaming AudioVAE CUDA graph returned an "
                "invalid output type"
            )

        expected_waveform_shape = (
            self._transition.capacity,
            self._transition.max_output_samples,
        )
        waveform = output.waveform
        if (
            tuple(waveform.shape) != expected_waveform_shape
            or waveform.dtype != torch.float32
            or waveform.device != self._transition.device
            or not waveform.is_contiguous()
            or waveform.requires_grad
        ):
            raise RuntimeError(
                "Ming-Omni-TTS streaming AudioVAE CUDA graph returned an invalid "
                "waveform contract"
            )

        sample_lengths = output.sample_lengths
        if (
            tuple(sample_lengths.shape) != (self._transition.capacity,)
            or sample_lengths.dtype != torch.long
            or sample_lengths.device != self._transition.device
            or not sample_lengths.is_contiguous()
            or sample_lengths.requires_grad
        ):
            raise RuntimeError(
                "Ming-Omni-TTS streaming AudioVAE CUDA graph returned an invalid "
                "sample-length contract"
            )

    def close(self) -> None:
        captured = self._captured_graph
        if captured is None:
            return
        with torch.cuda.device(self._transition.device):
            torch.cuda.current_stream(self._transition.device).synchronize()
            self._captured_graph = None
            captured.graph.reset()


def decode_ming_tts_audio_payload(
    payload: StagePayload,
    decoder: MingAudioDecoder,
    *,
    keep_latents: bool = False,
) -> StagePayload:
    """Decode generated acoustic latents into the terminal waveform payload."""

    state = load_ming_tts_state(payload)
    waveform = decoder.decode_full(state.generated_latents)
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


__all__ = ["MingAudioDecoder", "decode_ming_tts_audio_payload"]
