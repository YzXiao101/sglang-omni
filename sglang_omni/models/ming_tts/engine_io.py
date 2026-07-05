# SPDX-License-Identifier: Apache-2.0
"""SGLang engine I/O adapters for Ming-Omni-TTS."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    decode_prompt_latent,
    decode_speaker_embedding,
    encode_generated_latents,
    load_ming_tts_state,
    store_ming_tts_state,
)
from sglang_omni.models.ming_tts.profile_events import ming_profile_event
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

MING_ROW_PREFILL_CACHE_VERSION = "ming_tts_row_prefill_v1"


@dataclass
class MingTTSDecodeState:
    """Per-request Ming-TTS decode recurrence state."""

    max_decode_steps: int = 0
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0
    audio_patch_token_id: int = 0
    audio_eos_token_id: int = 0
    prompt_latent_for_history: Any = None
    latent_history: Any = None
    generated_latents: list[Any] = field(default_factory=list)
    generated_last_chunk: list[bool] = field(default_factory=list)
    stop_step: int | None = None
    engine_start_s: float = 0.0

    def ensure_latent_history(
        self,
        *,
        device: torch.device,
        history_patch_size: int,
        latent_dim: int,
    ) -> torch.Tensor:
        history = self.latent_history
        if history is None:
            history = torch.zeros(
                1,
                int(history_patch_size),
                int(latent_dim),
                device=device,
                dtype=torch.float32,
            )
            prompt_latent = self.prompt_latent_for_history
            if prompt_latent is not None:
                if not isinstance(prompt_latent, torch.Tensor):
                    prompt_latent = torch.as_tensor(prompt_latent)
                if prompt_latent.ndim == 2:
                    prompt_latent = prompt_latent.unsqueeze(0)
                prompt_latent = prompt_latent.to(device=device, dtype=torch.float32)
                history_len = int(history.shape[1])
                prompt_len = int(prompt_latent.shape[1])
                if prompt_len >= history_len:
                    history.copy_(prompt_latent[:, -history_len:, :])
                else:
                    history[:, -prompt_len:, :].copy_(prompt_latent)
            self.latent_history = history
            return history

        if not isinstance(history, torch.Tensor):
            history = torch.as_tensor(history)
        if history.ndim == 2:
            history = history.unsqueeze(0)
        history = history.to(device=device, dtype=torch.float32)
        self.latent_history = history
        return history

    def release_tensors(self) -> None:
        self.latent_history = None
        self.prompt_latent_for_history = None
        self.generated_latents.clear()


@dataclass
class MingTTSStaticPrefillCacheKey:
    """Synthetic prefill radix key for projected Ming-TTS embeddings."""

    input_ids: list[int]
    extra_key: str


def _build_prefill_row_cache_key_ids(prefill_input_embeds: torch.Tensor) -> list[int]:
    rows = (
        prefill_input_embeds.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    key_ids: list[int] = []
    for row in rows:
        digest = hashlib.blake2b(row.numpy().tobytes(), digest_size=8).digest()
        key_ids.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
    return key_ids


def _build_row_prefill_extra_key(
    *,
    model_identity: str,
    input_dtype: torch.dtype,
    hidden_size: int,
    patch_size: int,
    latent_dim: int,
    audio_start_token_id: int,
    audio_patch_token_id: int,
    audio_eos_token_id: int,
) -> str:
    payload: dict[str, Any] = {
        "version": MING_ROW_PREFILL_CACHE_VERSION,
        "model_identity": str(model_identity),
        "input_dtype": str(input_dtype),
        "hidden_size": int(hidden_size),
        "patch_size": int(patch_size),
        "latent_dim": int(latent_dim),
        "audio_start_token_id": int(audio_start_token_id),
        "audio_patch_token_id": int(audio_patch_token_id),
        "audio_eos_token_id": int(audio_eos_token_id),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
    return f"ming_tts:row-prefill:v1:{digest}"


@dataclass
class MingTTSSGLangRequestData(SGLangARRequestData):
    """Scheduler-owned state for Ming-Omni-TTS generation."""

    enforce_request_limits: bool = True
    static_prefill_cache_key: MingTTSStaticPrefillCacheKey | None = None
    decode_state: MingTTSDecodeState | None = None
    state: MingTTSState | None = None
    prompt_input_ids: Any = None

    def release_tensors(self) -> None:
        self.prefill_input_embeds = None
        self.decode_input_embeds = None
        if self.decode_state is not None:
            self.decode_state.release_tensors()
        self.pending_feedback_queue.clear()


def _build_prefill_input_embeds(
    *,
    model: Any,
    input_ids_list: list[int],
    state: MingTTSState,
    spk_emb: Any,
    prompt_latent: Any,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    # Ming projected prefill is model-weight dependent: speaker embeddings and
    # prompt latents must be projected before SGLang sees the prefill rows.
    embedding = model.get_input_embeddings()
    input_ids_for_embedding = torch.tensor(
        input_ids_list,
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        prompt_embeds = embedding(input_ids_for_embedding).to(
            device=device,
            dtype=dtype,
        )
        if spk_emb is not None:
            positions = state.spk_injection_positions
            if positions is None:
                positions = [
                    int(position) + 1 for position in (state.spk_token_positions or [])
                ]
            projected_spk = model.spk_head(spk_emb)
            for row, position in enumerate(positions):
                position = int(position)
                prompt_embeds[position] = projected_spk[row].to(
                    device=prompt_embeds.device,
                    dtype=prompt_embeds.dtype,
                )

        prompt_latent_for_history = None
        if prompt_latent is not None:
            token_count = int(state.prompt_latent_token_count)
            start = state.prompt_latent_start_position
            if start is None:
                start = int(state.audio_token_position) + 1
            projected_prompt = model.linear_proj_audio(
                prompt_latent.to(dtype=dtype).reshape(
                    -1,
                    int(model.patch_size),
                    int(model.latent_dim),
                )
            )
            projected_prompt = projected_prompt.reshape(
                1,
                -1,
                int(projected_prompt.shape[-1]),
            )[0]
            end = int(start) + int(token_count)
            prompt_embeds[int(start) : end] = projected_prompt.to(
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
            prompt_latent_for_history = prompt_latent.detach()

        return prompt_embeds.detach(), prompt_latent_for_history


def make_ming_tts_scheduler_adapters(
    *,
    model: Any,
    tokenizer: MingTTSTokenizerBundle,
    radix_cache_enabled: bool = False,
    model_cache_identity: str = "",
    owns_acoustic_result: bool = True,
):
    """Build StagePayload <-> SGLang request adapters for Ming-Omni-TTS."""

    def request_builder(payload: StagePayload) -> MingTTSSGLangRequestData:
        with ming_profile_event(payload.request_id, "ming_ar_request_build"):
            return _request_builder_impl(payload)

    def _request_builder_impl(payload: StagePayload) -> MingTTSSGLangRequestData:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        def config_value(config: Any, field: str) -> Any:
            if config is None:
                return None
            if isinstance(config, dict):
                return config.get(field)
            return getattr(config, field, None)

        state = load_ming_tts_state(payload)
        input_ids_list = [int(token_id) for token_id in (state.input_ids or [])]

        vocab_size = None
        for owner in (
            getattr(model, "config", None),
            getattr(model, "model_config", None),
            getattr(model, "hf_text_config", None),
            model,
        ):
            value = config_value(owner, "vocab_size")
            if value is not None:
                vocab_size = int(value)
                break
            llm_config = config_value(owner, "llm_config")
            value = config_value(llm_config, "vocab_size")
            if value is not None:
                vocab_size = int(value)
                break
        if vocab_size is None:
            vocab_size = int(len(tokenizer.tokenizer))

        sampling_params = SamplingParams(
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            stop_token_ids=[int(tokenizer.special.end_of_audio)],
        )
        sampling_params.normalize(None)
        sampling_params.verify(vocab_size)

        embedding = model.get_input_embeddings()
        weight = embedding.weight
        prompt_latent_for_history = None
        spk_emb = decode_speaker_embedding(
            state,
            device=weight.device,
            dtype=weight.dtype,
        )
        prompt_latent = decode_prompt_latent(
            state,
            device=weight.device,
            dtype=torch.float32,
        )
        requires_projected_prefill = (
            radix_cache_enabled or spk_emb is not None or prompt_latent is not None
        )
        prefill_input_embeds = None
        if requires_projected_prefill:
            prefill_input_embeds, prompt_latent_for_history = (
                _build_prefill_input_embeds(
                    model=model,
                    input_ids_list=input_ids_list,
                    state=state,
                    spk_emb=spk_emb,
                    prompt_latent=prompt_latent,
                    dtype=weight.dtype,
                    device=weight.device,
                )
            )

        static_prefill_cache_key = None
        if radix_cache_enabled:
            static_prefill_cache_key = MingTTSStaticPrefillCacheKey(
                input_ids=_build_prefill_row_cache_key_ids(prefill_input_embeds),
                extra_key=_build_row_prefill_extra_key(
                    model_identity=model_cache_identity,
                    input_dtype=weight.dtype,
                    hidden_size=int(weight.shape[1]),
                    patch_size=int(model.patch_size),
                    latent_dim=int(model.latent_dim),
                    audio_start_token_id=int(tokenizer.special.audio_start),
                    audio_patch_token_id=int(tokenizer.special.audio_patch),
                    audio_eos_token_id=int(tokenizer.special.end_of_audio),
                ),
            )

        # Default to a request-local radix namespace; only row-prefill synthetic
        # ids opt into cross-request sharing.
        if static_prefill_cache_key is None:
            req_input_ids_list = input_ids_list
            req_extra_key = f"ming_tts:{payload.request_id}"
        else:
            req_input_ids_list = static_prefill_cache_key.input_ids
            req_extra_key = static_prefill_cache_key.extra_key
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=req_input_ids_list,
            sampling_params=sampling_params,
            eos_token_ids={int(tokenizer.special.end_of_audio)},
            vocab_size=vocab_size,
            extra_key=req_extra_key,
        )
        req.tokenizer = None
        req._input_embeds_are_projected = prefill_input_embeds is not None

        input_ids = torch.tensor(req_input_ids_list, dtype=torch.long)
        prompt_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        decode_state = MingTTSDecodeState(
            max_decode_steps=int(state.max_decode_steps),
            cfg=float(state.cfg),
            sigma=float(state.sigma),
            temperature=float(state.temperature),
            audio_patch_token_id=int(tokenizer.special.audio_patch),
            audio_eos_token_id=int(tokenizer.special.end_of_audio),
            prompt_latent_for_history=prompt_latent_for_history,
            engine_start_s=time.perf_counter(),
        )
        data = MingTTSSGLangRequestData(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            max_new_tokens=int(state.max_decode_steps),
            temperature=0.0,
            output_ids=req.output_ids,
            req=req,
            state=state,
            prefill_input_embeds=prefill_input_embeds,
            static_prefill_cache_key=static_prefill_cache_key,
            decode_state=decode_state,
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: MingTTSSGLangRequestData) -> StagePayload:
        try:
            if not owns_acoustic_result:
                return data.stage_payload
            request_id = data.stage_payload.request_id
            with ming_profile_event(request_id, "ming_response_serialize"):
                return _result_adapter_impl(data)
        finally:
            data.release_tensors()

    def _result_adapter_impl(data: MingTTSSGLangRequestData) -> StagePayload:
        decode_state = data.decode_state
        payload = data.stage_payload
        state = data.state if data.state is not None else load_ming_tts_state(payload)

        latent_chunks: list[Any] = []
        for latent in decode_state.generated_latents:
            tensor = (
                latent.detach()
                if hasattr(latent, "detach")
                else torch.as_tensor(latent)
            )
            if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
                tensor = tensor.squeeze(0)
            latent_chunks.append(tensor)

        raw = data.finish_reason
        if raw is None and data.req is not None:
            finished_reason = getattr(data.req, "finished_reason", None)
            if finished_reason is not None and hasattr(finished_reason, "to_json"):
                raw = finished_reason.to_json().get("type")
            elif finished_reason is not None:
                raw = str(finished_reason)

        normalized = str(raw).lower() if raw is not None else None
        if decode_state.stop_step is not None:
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
        elif len(decode_state.generated_latents) >= int(decode_state.max_decode_steps):
            finish_reason = "length"
        else:
            finish_reason = "stop"

        prompt_input_ids = data.prompt_input_ids
        if prompt_input_ids is None:
            prompt_tokens = 0
        else:
            shape = getattr(prompt_input_ids, "shape", None)
            prompt_tokens = (
                int(shape[0])
                if shape is not None and len(shape)
                else len(prompt_input_ids)
            )

        generated = torch.stack(latent_chunks, dim=0)
        state.generated_last_chunk = [
            bool(item) for item in decode_state.generated_last_chunk
        ]
        state.stop_step = decode_state.stop_step
        state.finish_reason = finish_reason
        state.prompt_tokens = prompt_tokens
        state.completion_tokens = len(latent_chunks)
        state.engine_time_s = time.perf_counter() - decode_state.engine_start_s

        for field_name, value in encode_generated_latents(generated).items():
            setattr(state, field_name, value)

        return store_ming_tts_state(payload, state)

    return request_builder, result_adapter


__all__ = [
    "MingTTSDecodeState",
    "MingTTSSGLangRequestData",
    "MingTTSStaticPrefillCacheKey",
    "make_ming_tts_scheduler_adapters",
]
