# SPDX-License-Identifier: Apache-2.0
"""SGLang engine I/O adapters for Ming-Omni-TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch

from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

_PROMPT_CACHE_SCHEMA = "ming-tts:prompt:v3"


@dataclass
class MingTTSLatentPatch:
    latent: torch.Tensor
    is_last: bool


@dataclass
class MingTTSSGLangRequestData(SGLangARRequestData):
    """Per-request scheduler state for Ming-Omni-TTS."""

    enforce_request_limits: bool = True
    state: MingTTSState | None = None
    audio_patch_token_id: int = 0
    audio_eos_token_id: int = 0
    engine_start_s: float = 0.0
    generated_latents: torch.Tensor | None = None
    stop_step: int | None = None
    is_streaming: bool = False
    pending_stream_patch: MingTTSLatentPatch | None = None


def _prompt_cache_extra_key(
    state: MingTTSState,
    *,
    enabled: bool,
    request_id: str,
) -> str | None:
    is_reference = state.ref_audio is not None
    has_speaker = state.spk_emb is not None
    has_latent = state.prompt_latent is not None
    if is_reference:
        if not has_speaker or not has_latent:
            raise ValueError(
                f"Ming-Omni-TTS reference request {request_id!r} is missing "
                "reference conditioning"
            )
    elif has_speaker or has_latent:
        raise ValueError(
            f"Ming-Omni-TTS text-only request {request_id!r} unexpectedly "
            "contains reference conditioning"
        )

    if not enabled:
        return None
    if not is_reference:
        return f"{_PROMPT_CACHE_SCHEMA}:text"

    producer_digest = state.prompt_conditioning_digest
    if producer_digest is None:
        raise ValueError(
            f"Ming-Omni-TTS reference request {request_id!r} is missing the "
            "prompt cache digest while Radix cache is enabled"
        )

    return f"{_PROMPT_CACHE_SCHEMA}:reference:{producer_digest}"


def make_ming_tts_scheduler_adapters(
    *,
    model: Any,
    tokenizer: MingTTSTokenizerBundle,
    reset_request: Callable[[str], None],
    prompt_radix_cache_enabled: bool,
    owns_acoustic_result: bool = True,
):
    """Build StagePayload <-> SGLang request adapters for Ming-Omni-TTS."""

    def request_builder(payload: StagePayload) -> MingTTSSGLangRequestData:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        state = load_ming_tts_state(payload)
        input_ids_list = [int(token_id) for token_id in (state.input_ids or [])]
        vocab_size = int(model.vocab_size)

        sampling_params = SamplingParams(
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            stop_token_ids=[int(tokenizer.special.end_of_audio)],
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        requires_projected_prefill = (
            state.spk_emb is not None or state.prompt_latent is not None
        )
        extra_key = _prompt_cache_extra_key(
            state,
            enabled=prompt_radix_cache_enabled,
            request_id=payload.request_id,
        )
        # Note (yzxiao): origin_input_ids is the Radix identity, while
        # data.input_ids below keeps the actual ids used to build projected
        # embeddings. Row tags prevent conditioning rows from aliasing ordinary
        # audio-patch tokens in the cache.
        radix_input_ids = input_ids_list
        if prompt_radix_cache_enabled and state.ref_audio is not None:
            radix_input_ids = input_ids_list.copy()
            speaker_positions = cast(list[int], state.spk_injection_positions)
            for speaker_row, position in enumerate(speaker_positions):
                radix_input_ids[position] = vocab_size + 2 * speaker_row

            latent_start = cast(int, state.prompt_latent_start_position)
            for latent_offset in range(state.prompt_latent_token_count):
                radix_input_ids[latent_start + latent_offset] = (
                    vocab_size + 2 * latent_offset + 1
                )

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=radix_input_ids,
            sampling_params=sampling_params,
            eos_token_ids={int(tokenizer.special.end_of_audio)},
            vocab_size=vocab_size,
            extra_key=extra_key,
        )
        req.tokenizer = None
        req._input_embeds_are_projected = requires_projected_prefill

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        data = MingTTSSGLangRequestData(
            input_ids=input_ids,
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            output_ids=req.output_ids,
            req=req,
            state=state,
            input_embeds_are_projected=requires_projected_prefill,
            audio_patch_token_id=int(tokenizer.special.audio_patch),
            audio_eos_token_id=int(tokenizer.special.end_of_audio),
            engine_start_s=time.perf_counter(),
            is_streaming=bool((payload.request.params or {}).get("stream", False)),
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: MingTTSSGLangRequestData) -> StagePayload:
        request_id = data.stage_payload.request_id
        try:
            if not owns_acoustic_result:
                return data.stage_payload
            payload = data.stage_payload
            state = data.state
            generated = data.generated_latents
            if generated is None and not data.is_streaming:
                generated = torch.empty(
                    (0, int(model.patch_size), int(model.latent_dim)),
                    dtype=torch.float32,
                )
            completion_tokens = (
                int(data.generation_steps)
                if data.is_streaming
                else int(generated.shape[0])
            )

            raw = data.finish_reason
            if raw is None and data.req is not None:
                finished_reason = getattr(data.req, "finished_reason", None)
                if finished_reason is not None and hasattr(finished_reason, "to_json"):
                    raw = finished_reason.to_json().get("type")
                elif finished_reason is not None:
                    raw = str(finished_reason)

            normalized = str(raw).lower() if raw is not None else None
            if data.stop_step is not None:
                finish_reason = "stop"
            elif normalized is not None:
                if "length" in normalized:
                    finish_reason = "length"
                elif "abort" in normalized:
                    finish_reason = "abort"
                elif "error" in normalized:
                    finish_reason = "error"
                else:
                    finish_reason = str(raw)
            elif completion_tokens >= int(data.max_new_tokens):
                finish_reason = "length"
            else:
                finish_reason = "stop"

            state.stop_step = data.stop_step
            state.finish_reason = finish_reason
            state.prompt_tokens = len(data.input_ids)
            state.completion_tokens = completion_tokens
            state.engine_time_s = time.perf_counter() - data.engine_start_s
            state.generated_latents = None if data.is_streaming else generated

            return store_ming_tts_state(payload, state)
        finally:
            reset_request(request_id)

    return request_builder, result_adapter


def build_ming_tts_stream_output(
    request_id: str,
    data: MingTTSSGLangRequestData,
    req_output: Any,
) -> list[OutgoingMessage]:
    del req_output
    patch = data.pending_stream_patch
    if not data.is_streaming or patch is None:
        return []

    latent = patch.latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
    data.pending_stream_patch = None
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=latent,
            target="audio_decode",
            metadata={
                "modality": "audio_codes",
                "stream": True,
                "is_last": bool(patch.is_last),
            },
        )
    ]


__all__ = [
    "MingTTSLatentPatch",
    "MingTTSSGLangRequestData",
    "build_ming_tts_stream_output",
    "make_ming_tts_scheduler_adapters",
]
