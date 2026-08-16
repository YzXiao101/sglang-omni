# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduling for Ming-Omni-TTS."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.payload_types import load_ming_tts_state
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

logger = logging.getLogger(__name__)


class _AudioVAEStreamingStateManager:
    """Own request-to-slot bindings and enforce clean rows before reuse."""

    def __init__(
        self,
        decoder: MingAudioDecoder,
    ) -> None:
        self._decoder = decoder
        self._request_to_slot: dict[str, int] = {}
        self._free_slots = list(reversed(range(decoder.stream_capacity)))

    def try_bind(self, request_id: str) -> int | None:
        slot = self._request_to_slot.get(request_id)
        if slot is not None:
            return slot
        if not self._free_slots:
            return None
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        return slot

    def slot_for(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def resolve_slots(self, request_ids: Sequence[str]) -> tuple[int, ...]:
        slots = []
        for request_id in request_ids:
            slot = self._request_to_slot.get(request_id)
            if slot is None:
                raise AssertionError(
                    f"Ming-Omni-TTS stream {request_id!r} has no AudioVAE slot"
                )
            slots.append(slot)
        return tuple(slots)

    def reset_and_release(self, request_ids: Sequence[str]) -> None:
        bindings = {
            request_id: self._request_to_slot[request_id]
            for request_id in request_ids
            if request_id in self._request_to_slot
        }
        if not bindings:
            return

        slots = tuple(bindings.values())
        self._decoder.reset_stream_rows(slots)
        for request_id, slot in bindings.items():
            del self._request_to_slot[request_id]
            self._free_slots.append(slot)

    def release_clean(self, request_ids: Sequence[str]) -> None:
        """Release rows already cleaned by successful terminal transitions."""
        slots = self.resolve_slots(request_ids)
        for request_id, slot in zip(request_ids, slots, strict=True):
            del self._request_to_slot[request_id]
            self._free_slots.append(slot)

    def reset_all(self) -> None:
        self._decoder.reset_all_stream_rows()
        self._request_to_slot.clear()
        self._free_slots = list(reversed(range(self._decoder.stream_capacity)))


@dataclass(slots=True)
class _StreamState:
    expected_chunk_id: int = 0
    pending_patches: list[torch.Tensor] = field(default_factory=list)
    terminal_received: bool = False
    initial_group_consumed: bool = False
    emitted_samples: int = 0
    slot_wait_seq: int | None = None
    terminal_decoded: bool = False
    stream_done_received: bool = False


@dataclass(frozen=True, slots=True)
class _StreamingStepItem:
    patches: tuple[torch.Tensor, ...]
    terminal: bool


_StreamingStepPlan = tuple[_StreamingStepItem, ...]


class _MingAudioVAEFatalError(RuntimeError):
    """Fixed streaming transaction cannot safely continue serving."""


class MingTTSStreamingVocoderScheduler(
    StreamingVocoderBase[_StreamState, _StreamingStepPlan]
):
    """Schedule Ming acoustic latents over one fixed-shape AudioVAE decoder."""

    _can_batch_stream_chunks = True
    _stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        initial_chunk_patches: int,
        steady_chunk_patches: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        keep_latents: bool = False,
    ) -> None:
        if max_batch_size != decoder.stream_capacity:
            raise ValueError(
                "Ming-Omni-TTS max_batch_size must match the fixed AudioVAE "
                f"decoder capacity, got {max_batch_size!r} and "
                f"{decoder.stream_capacity}"
            )
        self._decoder = decoder
        self._state_manager = _AudioVAEStreamingStateManager(decoder)
        self._patch_size = int(patch_size)
        self._latent_dim = int(latent_dim)
        self._initial_chunk_patches = int(initial_chunk_patches)
        self._steady_chunk_patches = int(steady_chunk_patches)
        self._next_slot_wait_seq = 0
        self._pending_release_ids: set[str] = set()
        self._cleanup_wake_queued = False
        self._stop_requested = threading.Event()
        # Abort cleanup can originate outside the scheduler thread, where CUDA
        # reset is unsafe. This private sentinel reuses the thread-safe inbox and
        # is intercepted by object identity before normal request dispatch.
        self._cleanup_wake_message = IncomingMessage(
            request_id=str(uuid.uuid4()),
            type="new_request",
        )
        super().__init__(
            partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            ),
            sample_rate=decoder.sample_rate,
            stream_source_hint="Ming-Omni-TTS",
            stream_input_modality="audio_latents",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def start(self) -> None:
        """Run the common inbox loop while keeping fixed-state faults fatal."""
        try:
            with self._state_lock:
                if self._stop_requested.is_set():
                    return
                self.on_serving_start()
                self._running = True
                if self._stop_requested.is_set():
                    self._running = False
            if not self._running:
                return
            loop = asyncio.new_event_loop()
            try:
                while self._running:
                    msg = self._next_message()
                    if msg is None:
                        continue
                    if self._is_aborted(msg.request_id):
                        continue
                    try:
                        self._handle_message(msg, loop)
                    except (_MingAudioVAEFatalError, AssertionError):
                        raise
                    except Exception as exc:
                        logger.exception(
                            "%s failed for %s",
                            type(self).__name__,
                            msg.request_id,
                        )
                        self._emit_error(msg.request_id, exc)
                        self.abort(msg.request_id)
            finally:
                loop.close()
        finally:
            self._shutdown_stream_states()

    def stop(self) -> None:
        self._stop_requested.set()
        super().stop()

    def create_stream_state(self, request_id: str) -> _StreamState:
        del request_id
        return _StreamState()

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
        if state.terminal_received:
            raise ValueError(
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
        if (
            self._has_executable_work(state)
            and self._state_manager.slot_for(request_id) is None
            and state.slot_wait_seq is None
        ):
            state.slot_wait_seq = self._next_slot_wait_seq
            self._next_slot_wait_seq += 1
        return state

    def validate_chunk(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if latents.device.type != "cpu":
            raise ValueError(
                "Ming-Omni-TTS stream latent must be on CPU, "
                f"got device {latents.device}"
            )
        if latents.dtype != torch.float32:
            raise TypeError(
                "Ming-Omni-TTS stream latent dtype must be torch.float32, "
                f"got {latents.dtype}"
            )
        expected_shape = (self._patch_size, self._latent_dim)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Ming-Omni-TTS stream latent shape must be {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        return latents.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patches.append(latents)

    def _has_executable_work(self, state: _StreamState) -> bool:
        if state.terminal_decoded:
            return False
        if state.terminal_received:
            return bool(state.pending_patches)
        return len(state.pending_patches) >= self._next_chunk_patches(state)

    def _next_chunk_patches(self, state: _StreamState) -> int:
        if state.initial_group_consumed:
            return self._steady_chunk_patches
        return self._initial_chunk_patches

    def select_step_participants(self) -> list[tuple[str, _StreamState]]:
        waiting = sorted(
            (
                (state.slot_wait_seq, request_id, state)
                for request_id, state in self._stream_state_items()
                if state.slot_wait_seq is not None and not self._is_aborted(request_id)
            ),
            key=lambda item: item[0],
        )
        for _, request_id, state in waiting:
            if self._is_aborted(request_id):
                continue
            if self._state_manager.try_bind(request_id) is None:
                break
            state.slot_wait_seq = None

        return [
            (request_id, state)
            for request_id, state in self._stream_state_items()
            if not self._is_aborted(request_id)
            and self._has_executable_work(state)
            and self._state_manager.slot_for(request_id) is not None
        ]

    def build_step_plan(
        self,
        participants: list[tuple[str, _StreamState]],
    ) -> _StreamingStepPlan:
        plan = []
        for _, state in participants:
            pending_count = len(state.pending_patches)
            target = self._next_chunk_patches(state)
            if state.terminal_received:
                consume = min(target, pending_count)
                terminal = pending_count <= target
            else:
                consume = target
                terminal = False
            plan.append(
                _StreamingStepItem(
                    patches=tuple(state.pending_patches[:consume]),
                    terminal=terminal,
                )
            )
        return tuple(plan)

    def run_step(
        self,
        participants: list[tuple[str, _StreamState]],
        plan: _StreamingStepPlan,
    ) -> dict[str, torch.Tensor]:
        request_ids = tuple(request_id for request_id, _ in participants)
        slot_ids = self._state_manager.resolve_slots(request_ids)
        waveforms = self._decoder.run_streaming(
            slot_ids=slot_ids,
            patch_groups=tuple(item.patches for item in plan),
            terminal_flags=tuple(item.terminal for item in plan),
        )
        decoded = {}
        for (request_id, state), item, waveform in zip(
            participants,
            plan,
            waveforms,
            strict=True,
        ):
            del state.pending_patches[: len(item.patches)]
            if item.terminal:
                state.terminal_decoded = True
            elif not state.initial_group_consumed:
                state.initial_group_consumed = True
            sample_count = int(waveform.numel())
            state.emitted_samples += sample_count
            if sample_count > 0:
                decoded[request_id] = waveform

        self._state_manager.release_clean(
            tuple(
                request_id
                for request_id, item in zip(request_ids, plan, strict=True)
                if item.terminal
            )
        )
        return decoded

    def on_step_failure(
        self,
        participants: list[tuple[str, _StreamState]],
        exc: BaseException,
    ) -> list[str]:
        del participants
        raise _MingAudioVAEFatalError(
            "Ming-Omni-TTS fixed AudioVAE wave failed before a complete commit"
        ) from exc

    def decode_delta(
        self,
        request_id: str,
        state: _StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if not is_final:
            raise AssertionError(
                "Ming-Omni-TTS streaming decode must use the coalesced runner"
            )
        if not state.terminal_decoded:
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} ended before its terminal "
                "AudioVAE transition completed"
            )
        if state.emitted_samples <= 0:
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} completed without audio"
            )
        return None

    def _handle_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            if request_id not in self._stream_payloads:
                if request_id in self._completed_non_streaming_request_ids:
                    return
                self._pending_done.add(request_id)
                return

            state = self._get_or_create_stream_state(request_id)
            if state is None:
                return
            state.stream_done_received = True
            if not state.terminal_received:
                raise RuntimeError(
                    f"Ming-Omni-TTS stream {request_id!r} ended without a "
                    "terminal latent patch"
                )
            if state.terminal_decoded:
                super()._handle_stream_done(request_id)
                return
            failed = self._pump_streams()
        for failed_request_id in failed:
            self._cleanup_aborted_request(failed_request_id)

    def _pump_streams(self) -> list[str]:
        """Own Ming's fixed transaction and exact-request publication policy."""
        failed: list[str] = []
        while True:
            participants = self.select_step_participants()
            if not participants:
                break
            plan = self.build_step_plan(participants)
            try:
                decoded = self.run_step(participants, plan)
            except Exception as exc:
                failed.extend(self.on_step_failure(participants, exc))
                break
            for request_id, _ in participants:
                waveform = decoded.get(request_id)
                if waveform is None or self._is_aborted(request_id):
                    continue
                try:
                    message = self._stream_chunk_message(request_id, waveform)
                except Exception as exc:
                    logger.exception(
                        "Ming-Omni-TTS failed to build streaming output for %s",
                        request_id,
                    )
                    self._emit_error(request_id, exc)
                    self._abort_state(request_id)
                    failed.append(request_id)
                    continue
                self._mark_stream_emitted(request_id)
                self.outbox.put(message)

        finalization_failures = []
        for request_id, state in self._stream_state_items():
            if (
                self._is_aborted(request_id)
                or not state.stream_done_received
                or not state.terminal_decoded
            ):
                continue
            try:
                super()._handle_stream_done(request_id)
            except AssertionError:
                raise
            except Exception as exc:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                finalization_failures.append(request_id)
        return failed + finalization_failures

    def release_stream_resources(
        self,
        request_id: str,
        state: _StreamState,
    ) -> None:
        del state
        if self._state_manager.slot_for(request_id) is None:
            return
        self._pending_release_ids.add(request_id)
        if self._cleanup_wake_queued:
            return
        self._cleanup_wake_queued = True
        self.inbox.put(self._cleanup_wake_message)

    def _is_aborted(self, request_id: str) -> bool:
        # The base applies its request-id tombstone filter before dispatch, so
        # the retained sentinel id must reach the object-identity check below.
        if request_id is self._cleanup_wake_message.request_id:
            return False
        return super()._is_aborted(request_id)

    def _handle_message(
        self,
        msg: IncomingMessage,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if msg is self._cleanup_wake_message:
            self._handle_cleanup_wake()
            return
        super()._handle_message(msg, loop)

    def _handle_cleanup_wake(self) -> None:
        with self._state_lock:
            pending = tuple(self._pending_release_ids)
            try:
                self._state_manager.reset_and_release(pending)
            except Exception as exc:
                raise _MingAudioVAEFatalError(
                    "Ming-Omni-TTS AudioVAE deferred slot reset failed"
                ) from exc
            self._pending_release_ids.difference_update(pending)
            self._cleanup_wake_queued = False
            failed = self._pump_streams()
        for request_id in failed:
            self._cleanup_aborted_request(request_id)

    def warmup_now(self) -> None:
        self._decoder.prepare_streaming()

    def on_serving_start(self) -> None:
        if not self._decoder.streaming_ready:
            raise RuntimeError(
                "required Ming-Omni-TTS streaming AudioVAE CUDA graph was not prepared"
            )

    def on_serving_stop(self) -> None:
        self._state_manager.reset_all()
        self._pending_release_ids.clear()
        self._cleanup_wake_queued = False
        self._decoder.close()

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
