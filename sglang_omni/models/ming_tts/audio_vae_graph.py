# SPDX-License-Identifier: Apache-2.0
"""CUDA graph execution for the Ming-Omni-TTS AudioVAE Qwen2 decoder."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import Decoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MingAudioVAEGraphKey:
    mode: str
    phase: str
    batch_size: int
    token_size: int
    dtype: torch.dtype


@dataclass
class _MingAudioVAECapturedGraph:
    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class _MingAudioVAEStreamingCapturedGraph:
    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor
    attention_mask: dict[str, torch.Tensor]
    position_offsets: torch.Tensor
    position_ids: torch.Tensor
    cache: Cache
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class MingAudioVAEKVState:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    absolute_position: int


@dataclass(frozen=True)
class _MingAudioVAEReplayResult:
    key: _MingAudioVAEGraphKey
    hidden_states: torch.Tensor
    true_lengths: tuple[int, ...]


@dataclass(frozen=True)
class _MingAudioVAEStreamingReplayResult:
    key: _MingAudioVAEGraphKey
    hidden_states: torch.Tensor
    states: tuple[MingAudioVAEKVState, ...]


class _MingAudioVAEFixedKVLayer(CacheLayerMixin):
    is_sliding = True

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        absolute_position: int,
        sliding_window: int,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.values = values
        self.device = keys.device
        self.dtype = keys.dtype
        self.cumulative_length = int(absolute_position)
        self.sliding_window = int(sliding_window)
        self.is_initialized = True
        self.next_keys: torch.Tensor | None = None
        self.next_values: torch.Tensor | None = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        del key_states, value_states
        raise RuntimeError("Ming-Omni-TTS fixed KV layers are initialized eagerly")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        full_keys = torch.cat((self.keys, key_states), dim=-2)
        full_values = torch.cat((self.values, value_states), dim=-2)
        cache_size = self.sliding_window - 1
        self.next_keys = full_keys[..., -cache_size:, :]
        self.next_values = full_values[..., -cache_size:, :]
        return full_keys, full_values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        cache_size = self.sliding_window - 1
        return cache_size + query_length, self.cumulative_length - cache_size

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sliding_window


class MingAudioVAEGraphRunner:
    """Own fixed-shape non-streaming and steady streaming Qwen2 graphs."""

    def __init__(
        self,
        decoder: Decoder,
        *,
        batch_sizes: list[int],
        token_sizes: list[int],
        streaming_token_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Ming-Omni-TTS AudioVAE CUDA graph requires a CUDA device")

        self._qwen = decoder.decoder
        self._device = device
        self._dtype = dtype
        self._batch_sizes = tuple(sorted(set(batch_sizes)))
        self._token_sizes = tuple(sorted(set(token_sizes)))
        self._streaming_token_size = int(streaming_token_size)
        if (
            not self._batch_sizes
            or not self._token_sizes
            or self._streaming_token_size <= 0
        ):
            raise ValueError(
                "Ming-Omni-TTS AudioVAE CUDA graph requires batch, non-streaming "
                "token, and streaming token buckets"
            )

        config = self._qwen.config
        self._hidden_size = int(config.hidden_size)
        self._num_layers = int(config.num_hidden_layers)
        self._num_kv_heads = int(config.num_key_value_heads)
        self._head_dim = int(
            getattr(
                config,
                "head_dim",
                int(config.hidden_size) // int(config.num_attention_heads),
            )
        )
        self._sliding_window = int(config.sliding_window)
        layer_types = tuple(config.layer_types)
        if len(layer_types) != self._num_layers or set(layer_types) != {
            "sliding_attention"
        }:
            raise ValueError(
                "Ming-Omni-TTS AudioVAE streaming graphs require every Qwen2 "
                f"layer to use sliding attention; got {layer_types!r}"
            )

        self._nonstreaming_graphs: dict[
            _MingAudioVAEGraphKey, _MingAudioVAECapturedGraph
        ] = {}
        self._streaming_graphs: dict[
            _MingAudioVAEGraphKey, _MingAudioVAEStreamingCapturedGraph
        ] = {}
        self._observed_keys: set[tuple[int, int, _MingAudioVAEGraphKey]] = set()
        self._replay_count = 0
        self._streaming_replay_count = 0
        self._active_rows = 0
        self._captured_rows = 0
        self._true_tokens = 0
        self._captured_tokens = 0
        self._handoff_count = 0
        self._streaming_gather_bytes = 0
        self._streaming_scatter_bytes = 0

    @property
    def max_batch_size(self) -> int:
        return self._batch_sizes[-1]

    @property
    def token_sizes(self) -> tuple[int, ...]:
        return self._token_sizes

    def capture(self) -> None:
        if self._nonstreaming_graphs or self._streaming_graphs:
            raise RuntimeError("Ming-Omni-TTS AudioVAE graphs are already captured")

        nonstreaming_keys = [
            _MingAudioVAEGraphKey(
                mode="nonstreaming",
                phase="full",
                batch_size=batch_size,
                token_size=token_size,
                dtype=self._dtype,
            )
            for batch_size in self._batch_sizes
            for token_size in self._token_sizes
        ]
        streaming_keys = [
            _MingAudioVAEGraphKey(
                mode="streaming",
                phase="steady",
                batch_size=batch_size,
                token_size=self._streaming_token_size,
                dtype=self._dtype,
            )
            for batch_size in self._batch_sizes
        ]
        nonstreaming_keys.sort(
            key=lambda key: (key.batch_size * key.token_size, key.token_size),
            reverse=True,
        )
        streaming_keys.sort(key=lambda key: key.batch_size, reverse=True)

        logger.info(
            "Capturing Ming-Omni-TTS AudioVAE Qwen2 graphs: batch_sizes=%s "
            "nonstream_token_sizes=%s stream_token_size=%d keys=%d",
            list(self._batch_sizes),
            list(self._token_sizes),
            self._streaming_token_size,
            len(nonstreaming_keys) + len(streaming_keys),
        )
        with torch.cuda.device(self._device):
            for key in nonstreaming_keys:
                self._nonstreaming_graphs[key] = self._capture_nonstreaming_graph(key)
            for key in streaming_keys:
                self._streaming_graphs[key] = self._capture_streaming_graph(key)

    def select_token_size(self, true_token_size: int) -> int:
        for token_size in self._token_sizes:
            if true_token_size <= token_size:
                return token_size
        raise ValueError(
            "Ming-Omni-TTS AudioVAE input exceeds the CUDA graph token "
            f"envelope: true_token_size={true_token_size}, "
            f"max_token_size={self._token_sizes[-1]}"
        )

    def replay(
        self,
        inputs: list[torch.Tensor],
        *,
        token_size: int,
    ) -> _MingAudioVAEReplayResult:
        if not inputs:
            raise ValueError("Ming-Omni-TTS AudioVAE graph replay requires input rows")
        true_batch_size = len(inputs)
        batch_size = self._select_batch_size(true_batch_size)
        key = _MingAudioVAEGraphKey(
            mode="nonstreaming",
            phase="full",
            batch_size=batch_size,
            token_size=token_size,
            dtype=self._dtype,
        )
        captured = self._nonstreaming_graphs.get(key)
        if captured is None:
            raise RuntimeError(
                f"Ming-Omni-TTS AudioVAE graph key was not captured: {key}"
            )

        true_lengths = tuple(int(value.shape[1]) for value in inputs)
        captured.inputs.zero_()
        captured.attention_mask.zero_()
        for row, value in enumerate(inputs):
            true_length = true_lengths[row]
            self._check_input(value)
            if true_length > token_size:
                raise ValueError(
                    "Ming-Omni-TTS AudioVAE graph input exceeds its selected "
                    f"token bucket: true_length={true_length}, token_size={token_size}"
                )
            captured.inputs[row, :true_length].copy_(value[0])
            captured.attention_mask[row, :true_length] = 1

        if true_batch_size < batch_size:
            captured.attention_mask[true_batch_size:, 0] = 1

        with torch.cuda.device(self._device):
            captured.graph.replay()
        self._record_replay(key, true_batch_size, true_lengths)
        return _MingAudioVAEReplayResult(
            key=key,
            hidden_states=captured.hidden_states,
            true_lengths=true_lengths,
        )

    def promote_dynamic_cache(self, cache: Cache) -> MingAudioVAEKVState | None:
        absolute_position = int(cache.get_seq_length())
        if absolute_position < self._sliding_window:
            return None
        if len(cache.layers) != self._num_layers:
            raise RuntimeError(
                "Ming-Omni-TTS AudioVAE dynamic cache does not match the Qwen2 "
                f"layer count: {len(cache.layers)} != {self._num_layers}"
            )

        cache_size = self._sliding_window - 1
        keys = []
        values = []
        for layer in cache.layers:
            if (
                layer.keys is None
                or layer.values is None
                or layer.keys.shape[0] != 1
                or layer.keys.shape[-2] != cache_size
            ):
                raise RuntimeError(
                    "Ming-Omni-TTS AudioVAE dynamic cache cannot enter the "
                    "fixed-KV streaming phase"
                )
            keys.append(layer.keys.detach())
            values.append(layer.values.detach())
        state = MingAudioVAEKVState(
            keys=tuple(keys),
            values=tuple(values),
            absolute_position=absolute_position,
        )
        self._handoff_count += 1
        return state

    def forward_streaming_eager(
        self,
        inputs: torch.Tensor,
        state: MingAudioVAEKVState,
    ) -> tuple[torch.Tensor, MingAudioVAEKVState]:
        self._check_input(inputs)
        cache = self._fixed_cache_from_state(state)
        token_size = int(inputs.shape[1])
        position_ids = torch.arange(
            state.absolute_position,
            state.absolute_position + token_size,
            device=self._device,
            dtype=torch.long,
        ).unsqueeze(0)
        attention_mask = self._streaming_attention_mask(1, token_size)
        hidden_states = self._forward_streaming(
            inputs,
            attention_mask,
            position_ids,
            cache,
        )
        next_state = self._next_fixed_state(
            cache,
            row=0,
            absolute_position=state.absolute_position + token_size,
        )
        return hidden_states, next_state

    def replay_streaming(
        self,
        inputs: list[torch.Tensor],
        states: list[MingAudioVAEKVState],
    ) -> _MingAudioVAEStreamingReplayResult:
        if not inputs or len(inputs) != len(states):
            raise ValueError(
                "Ming-Omni-TTS AudioVAE streaming replay requires aligned "
                "input and fixed-KV rows"
            )
        true_batch_size = len(inputs)
        batch_size = self._select_batch_size(true_batch_size)
        key = _MingAudioVAEGraphKey(
            mode="streaming",
            phase="steady",
            batch_size=batch_size,
            token_size=self._streaming_token_size,
            dtype=self._dtype,
        )
        captured = self._streaming_graphs.get(key)
        if captured is None:
            raise RuntimeError(
                f"Ming-Omni-TTS AudioVAE streaming graph key was not captured: {key}"
            )

        captured.inputs.zero_()
        captured.position_ids.zero_()
        for layer in captured.cache.layers:
            layer.keys.zero_()
            layer.values.zero_()

        for row, (value, state) in enumerate(zip(inputs, states)):
            self._check_input(value)
            if int(value.shape[1]) != self._streaming_token_size:
                raise ValueError(
                    "Ming-Omni-TTS AudioVAE steady streaming graph requires "
                    f"T={self._streaming_token_size}, got T={value.shape[1]}"
                )
            captured.inputs[row].copy_(value[0])
            torch.add(
                captured.position_offsets,
                state.absolute_position,
                out=captured.position_ids[row],
            )
            for layer_index, layer in enumerate(captured.cache.layers):
                layer.keys[row].copy_(state.keys[layer_index][0])
                layer.values[row].copy_(state.values[layer_index][0])

        state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for state in states
            for tensor in (*state.keys, *state.values)
        )
        with torch.cuda.device(self._device):
            captured.graph.replay()
        next_states = tuple(
            self._next_fixed_state(
                captured.cache,
                row=row,
                absolute_position=(
                    states[row].absolute_position + self._streaming_token_size
                ),
            )
            for row in range(true_batch_size)
        )
        self._streaming_replay_count += 1
        self._streaming_gather_bytes += state_bytes
        self._streaming_scatter_bytes += state_bytes
        self._record_replay(
            key,
            true_batch_size,
            (self._streaming_token_size,) * true_batch_size,
        )
        return _MingAudioVAEStreamingReplayResult(
            key=key,
            hidden_states=captured.hidden_states[:true_batch_size],
            states=next_states,
        )

    def log_stats(self) -> None:
        if self._replay_count == 0 and self._handoff_count == 0:
            return
        active_row_ratio = (
            self._active_rows / self._captured_rows if self._captured_rows else 0.0
        )
        true_token_ratio = (
            self._true_tokens / self._captured_tokens if self._captured_tokens else 0.0
        )
        logger.info(
            "Ming-Omni-TTS AudioVAE graph summary: replays=%d "
            "streaming_replays=%d active_row_ratio=%.4f true_token_ratio=%.4f "
            "handoffs=%d gather_bytes=%d scatter_bytes=%d observed_keys=%d",
            self._replay_count,
            self._streaming_replay_count,
            active_row_ratio,
            true_token_ratio,
            self._handoff_count,
            self._streaming_gather_bytes,
            self._streaming_scatter_bytes,
            len(self._observed_keys),
        )

    def _capture_nonstreaming_graph(
        self,
        key: _MingAudioVAEGraphKey,
    ) -> _MingAudioVAECapturedGraph:
        allocated_before, reserved_before, free_before, started_at = (
            self._capture_metrics()
        )
        inputs = torch.zeros(
            (key.batch_size, key.token_size, self._hidden_size),
            device=self._device,
            dtype=self._dtype,
        )
        attention_mask = torch.zeros(
            (key.batch_size, key.token_size),
            device=self._device,
            dtype=torch.long,
        )
        # Note (yzxiao): Capture-time padding keeps Transformers SDPA on the
        # explicit-mask path required by every padded replay.
        attention_mask[:, 0] = 1
        position_ids = torch.arange(
            key.token_size,
            device=self._device,
            dtype=torch.long,
        ).unsqueeze(0)
        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                self._forward_nonstreaming(inputs, attention_mask, position_ids)
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            hidden_states = self._forward_nonstreaming(
                inputs,
                attention_mask,
                position_ids,
            )
        torch.cuda.synchronize(self._device)
        self._log_capture(
            key,
            allocated_before,
            reserved_before,
            free_before,
            started_at,
        )
        return _MingAudioVAECapturedGraph(
            graph=graph,
            inputs=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hidden_states=hidden_states,
        )

    def _capture_streaming_graph(
        self,
        key: _MingAudioVAEGraphKey,
    ) -> _MingAudioVAEStreamingCapturedGraph:
        allocated_before, reserved_before, free_before, started_at = (
            self._capture_metrics()
        )
        inputs = torch.zeros(
            (key.batch_size, key.token_size, self._hidden_size),
            device=self._device,
            dtype=self._dtype,
        )
        position_offsets = torch.arange(
            key.token_size,
            device=self._device,
            dtype=torch.long,
        )
        position_ids = position_offsets.unsqueeze(0).expand(key.batch_size, -1).clone()
        position_ids.add_(self._sliding_window)
        attention_mask = self._streaming_attention_mask(
            key.batch_size,
            key.token_size,
        )
        cache = self._empty_fixed_cache(
            key.batch_size,
            absolute_position=self._sliding_window,
        )
        capture_stream = torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                self._forward_streaming(
                    inputs,
                    attention_mask,
                    position_ids,
                    cache,
                )
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        torch.cuda.synchronize(self._device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            hidden_states = self._forward_streaming(
                inputs,
                attention_mask,
                position_ids,
                cache,
            )
        torch.cuda.synchronize(self._device)
        self._log_capture(
            key,
            allocated_before,
            reserved_before,
            free_before,
            started_at,
        )
        return _MingAudioVAEStreamingCapturedGraph(
            graph=graph,
            inputs=inputs,
            attention_mask=attention_mask,
            position_offsets=position_offsets,
            position_ids=position_ids,
            cache=cache,
            hidden_states=hidden_states,
        )

    def _forward_nonstreaming(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        with self._autocast_context():
            outputs = self._qwen(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        return outputs.last_hidden_state

    def _forward_streaming(
        self,
        inputs: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        position_ids: torch.Tensor,
        cache: Cache,
    ) -> torch.Tensor:
        with self._autocast_context():
            outputs = self._qwen(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        return outputs.last_hidden_state

    def _empty_fixed_cache(
        self,
        batch_size: int,
        *,
        absolute_position: int,
    ) -> Cache:
        cache_size = self._sliding_window - 1
        shape = (batch_size, self._num_kv_heads, cache_size, self._head_dim)
        keys = tuple(
            torch.zeros(shape, device=self._device, dtype=self._dtype)
            for _ in range(self._num_layers)
        )
        values = tuple(
            torch.zeros(shape, device=self._device, dtype=self._dtype)
            for _ in range(self._num_layers)
        )
        return self._fixed_cache(keys, values, absolute_position=absolute_position)

    def _fixed_cache_from_state(self, state: MingAudioVAEKVState) -> Cache:
        return self._fixed_cache(
            state.keys,
            state.values,
            absolute_position=state.absolute_position,
        )

    def _fixed_cache(
        self,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
        *,
        absolute_position: int,
    ) -> Cache:
        if len(keys) != self._num_layers or len(values) != self._num_layers:
            raise RuntimeError(
                "Ming-Omni-TTS AudioVAE fixed KV state does not match the "
                f"Qwen2 layer count {self._num_layers}"
            )
        layers = [
            _MingAudioVAEFixedKVLayer(
                layer_keys,
                layer_values,
                absolute_position=absolute_position,
                sliding_window=self._sliding_window,
            )
            for layer_keys, layer_values in zip(keys, values)
        ]
        return Cache(layers=layers)

    def _next_fixed_state(
        self,
        cache: Cache,
        *,
        row: int,
        absolute_position: int,
    ) -> MingAudioVAEKVState:
        keys = []
        values = []
        for layer in cache.layers:
            if layer.next_keys is None or layer.next_values is None:
                raise RuntimeError(
                    "Ming-Omni-TTS AudioVAE fixed KV update did not produce "
                    "the next request state"
                )
            keys.append(layer.next_keys[row : row + 1].detach().clone())
            values.append(layer.next_values[row : row + 1].detach().clone())
        return MingAudioVAEKVState(
            keys=tuple(keys),
            values=tuple(values),
            absolute_position=absolute_position,
        )

    def _streaming_attention_mask(
        self,
        batch_size: int,
        token_size: int,
    ) -> dict[str, torch.Tensor]:
        cache_size = self._sliding_window - 1
        query_positions = torch.arange(
            token_size,
            device=self._device,
        ).unsqueeze(1)
        key_positions = torch.arange(
            -cache_size,
            token_size,
            device=self._device,
        ).unsqueeze(0)
        allowed = (key_positions <= query_positions) & (
            key_positions > query_positions - self._sliding_window
        )
        mask = torch.full(
            (token_size, cache_size + token_size),
            torch.finfo(self._dtype).min,
            device=self._device,
            dtype=self._dtype,
        )
        mask.masked_fill_(allowed, 0)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return {
            "sliding_attention": mask.expand(
                batch_size,
                1,
                token_size,
                cache_size + token_size,
            ).clone()
        }

    def _select_batch_size(self, true_batch_size: int) -> int:
        batch_size = next(
            (
                candidate
                for candidate in self._batch_sizes
                if true_batch_size <= candidate
            ),
            None,
        )
        if batch_size is None:
            raise ValueError(
                "Ming-Omni-TTS AudioVAE batch exceeds the CUDA graph envelope: "
                f"true_batch_size={true_batch_size}, "
                f"max_batch_size={self._batch_sizes[-1]}"
            )
        return batch_size

    def _check_input(self, value: torch.Tensor) -> None:
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[2] != self._hidden_size
        ):
            raise ValueError(
                "Ming-Omni-TTS AudioVAE graph expects prepared inputs with "
                f"shape [1, T, {self._hidden_size}], got {tuple(value.shape)}"
            )

    def _record_replay(
        self,
        key: _MingAudioVAEGraphKey,
        true_batch_size: int,
        true_lengths: tuple[int, ...],
    ) -> None:
        observed = (true_batch_size, max(true_lengths), key)
        if observed not in self._observed_keys:
            logger.info(
                "Ming-Omni-TTS AudioVAE graph selected: mode=%s phase=%s "
                "true_B=%d true_T=%d graph_B=%d graph_T=%d",
                key.mode,
                key.phase,
                true_batch_size,
                max(true_lengths),
                key.batch_size,
                key.token_size,
            )
            self._observed_keys.add(observed)
        self._replay_count += 1
        self._active_rows += true_batch_size
        self._captured_rows += key.batch_size
        self._true_tokens += sum(true_lengths)
        self._captured_tokens += key.batch_size * key.token_size

    def _capture_metrics(self) -> tuple[int, int, int, float]:
        torch.cuda.synchronize(self._device)
        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        free, _ = torch.cuda.mem_get_info(self._device)
        return allocated, reserved, free, time.perf_counter()

    def _log_capture(
        self,
        key: _MingAudioVAEGraphKey,
        allocated_before: int,
        reserved_before: int,
        free_before: int,
        started_at: float,
    ) -> None:
        allocated_after = torch.cuda.memory_allocated(self._device)
        reserved_after = torch.cuda.memory_reserved(self._device)
        free_after, _ = torch.cuda.mem_get_info(self._device)
        logger.info(
            "Captured Ming-Omni-TTS AudioVAE graph mode=%s phase=%s B=%d T=%d "
            "in %.3fs; allocated_delta=%d reserved_delta=%d free_delta=%d",
            key.mode,
            key.phase,
            key.batch_size,
            key.token_size,
            time.perf_counter() - started_at,
            allocated_after - allocated_before,
            reserved_after - reserved_before,
            free_after - free_before,
        )

    def _autocast_context(self):
        if self._dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self._dtype)
        return nullcontext()


__all__ = [
    "MingAudioVAEGraphRunner",
    "MingAudioVAEKVState",
]
