# SPDX-License-Identifier: Apache-2.0
"""SGLang scheduler adapters for Ming-Omni-TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.ming_tts.ar_runtime import (
    MingARRequestState,
    get_ming_ar_state,
    release_ming_ar_runtime_tensors,
)
from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    decode_prompt_latent,
    decode_speaker_embedding,
    encode_generated_latents,
)
from sglang_omni.models.ming_tts.profile_events import ming_profile_event
from sglang_omni.models.ming_tts.radix_cache_key import (
    build_ming_prefill_row_cache_key_ids,
    build_ming_row_prefill_extra_key,
)
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.types import ARRequestData


@dataclass
class MingTTSSGLangRequestData(ARRequestData):
    """Scheduler-owned state for Ming-Omni-TTS generation."""

    enforce_request_limits: bool = True
    req: Any = None
    synced: bool = False
    suppress_tokens: list[int] | None = None
    prefill_input_embeds: Any = None
    row_prefill_radix_cache_enabled: bool = False
    row_prefill_extra_key: str | None = None
    row_prefill_input_ids: Any = None
    stage_payload: Any = None
    ar_state: MingARRequestState | None = None
    state: MingTTSState = field(default_factory=MingTTSState)
    prompt_input_ids: Any = None


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
    import torch

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
            if len(positions) != int(spk_emb.shape[0]):
                raise ValueError(
                    "Ming-Omni-TTS speaker embedding count does not match "
                    "prompt injection positions: "
                    f"{int(spk_emb.shape[0])} != {len(positions)}"
                )
            projected_spk = model.spk_head(spk_emb)
            for row, position in enumerate(positions):
                position = int(position)
                if position < 0 or position >= int(prompt_embeds.shape[0]):
                    raise ValueError(
                        "Ming-Omni-TTS speaker injection position is outside "
                        f"prompt length: {position}"
                    )
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
            if token_count <= 0:
                raise ValueError(
                    "Ming-Omni-TTS prompt_latent_token_count must be > 0 "
                    "when prompt_latent is present"
                )
            if int(prompt_latent.shape[1]) != token_count * int(model.patch_size):
                raise ValueError(
                    "Ming-Omni-TTS prompt latent frame count does not match "
                    "prompt latent token count"
                )
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
            if end > int(prompt_embeds.shape[0]):
                raise ValueError(
                    "Ming-Omni-TTS prompt latent injection range exceeds "
                    f"prompt length: start={start}, count={token_count}"
                )
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
    projected_prefill_requires_radix_disabled: bool = False,
    radix_cache_disabled: bool = False,
    projected_prefill_radix_cache_enabled: bool = False,
    model_cache_identity: str = "",
):
    """Build StagePayload <-> SGLang request adapters for Ming-Omni-TTS."""

    def request_builder(payload: StagePayload) -> MingTTSSGLangRequestData:
        with ming_profile_event(payload.request_id, "ming_ar_request_build"):
            return _request_builder_impl(payload)

    def _request_builder_impl(payload: StagePayload) -> MingTTSSGLangRequestData:
        import torch
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        def config_value(config: Any, field: str) -> Any:
            if config is None:
                return None
            if isinstance(config, dict):
                return config.get(field)
            return getattr(config, field, None)

        state = MingTTSState.from_dict(payload.data)
        input_ids_list = [int(token_id) for token_id in (state.input_ids or [])]
        if not input_ids_list:
            raise ValueError(
                "Ming-Omni-TTS SGLang request requires preprocessed input_ids"
            )
        if state.audio_token_position is None:
            raise ValueError(
                "Ming-Omni-TTS SGLang request requires audio_token_position"
            )

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

        for name, token_id in (
            ("audio_patch", int(tokenizer.special.audio_patch)),
            ("audio_eos", int(tokenizer.special.end_of_audio)),
        ):
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    "Ming-Omni-TTS control token id exceeds vocab_size: "
                    f"{name}={token_id}, vocab_size={vocab_size}"
                )
        bad_prompt_ids = [
            token_id
            for token_id in input_ids_list
            if token_id < 0 or token_id >= vocab_size
        ]
        if bad_prompt_ids:
            raise ValueError(
                "Ming-Omni-TTS prompt contains token ids outside vocab_size: "
                f"{bad_prompt_ids[:8]!r}, vocab_size={vocab_size}"
            )

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
            projected_prefill_radix_cache_enabled
            or spk_emb is not None
            or prompt_latent is not None
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
            if (
                projected_prefill_requires_radix_disabled
                and not radix_cache_disabled
                and not projected_prefill_radix_cache_enabled
            ):
                raise RuntimeError(
                    "Ming-Omni-TTS projected reference prefill requires "
                    "disable_radix_cache=True unless "
                    "enable_ming_ar_projected_prefill_radix_cache=True. The "
                    "plain token ids do not uniquely encode speaker/prompt "
                    "latent embeddings or generated continuous audio state."
                )

        row_prefill_input_ids_list: list[int] | None = None
        row_prefill_extra_key: str | None = None
        if projected_prefill_radix_cache_enabled:
            if prefill_input_embeds is None:
                raise RuntimeError(
                    "Ming-Omni-TTS row-prefill radix cache requires "
                    "prefill_input_embeds"
                )
            row_prefill_input_ids_list = build_ming_prefill_row_cache_key_ids(
                prefill_input_embeds
            )
            row_prefill_extra_key = build_ming_row_prefill_extra_key(
                model_identity=model_cache_identity,
                input_dtype=weight.dtype,
                hidden_size=int(weight.shape[1]),
                patch_size=int(model.patch_size),
                latent_dim=int(model.latent_dim),
                audio_start_token_id=int(tokenizer.special.audio_start),
                audio_patch_token_id=int(tokenizer.special.audio_patch),
                audio_eos_token_id=int(tokenizer.special.end_of_audio),
            )

        req_input_ids_list = row_prefill_input_ids_list or input_ids_list
        # Default to a request-local radix namespace; only row-prefill synthetic
        # ids opt into cross-request sharing.
        req_extra_key = row_prefill_extra_key or f"ming_tts:{payload.request_id}"

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
        req._codec_suppress_tokens = None

        input_ids = torch.tensor(req_input_ids_list, dtype=torch.long)
        prompt_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        row_prefill_input_ids = (
            torch.tensor(row_prefill_input_ids_list, dtype=torch.long)
            if row_prefill_input_ids_list is not None
            else None
        )
        ar_state = MingARRequestState(
            generation_steps=0,
            max_decode_steps=int(state.max_decode_steps),
            cfg=float(state.cfg),
            sigma=float(state.sigma),
            flow_temperature=float(state.temperature),
            audio_patch_token_id=int(tokenizer.special.audio_patch),
            audio_eos_token_id=int(tokenizer.special.end_of_audio),
            audio_token_id=int(tokenizer.special.audio_start),
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
            row_prefill_radix_cache_enabled=bool(projected_prefill_radix_cache_enabled),
            row_prefill_extra_key=row_prefill_extra_key,
            row_prefill_input_ids=row_prefill_input_ids,
            ar_state=ar_state,
        )
        data.stage_payload = payload
        return data

    def result_adapter(data: MingTTSSGLangRequestData) -> StagePayload:
        request_id = data.stage_payload.request_id
        with ming_profile_event(request_id, "ming_response_serialize"):
            return _result_adapter_impl(data)

    def _result_adapter_impl(data: MingTTSSGLangRequestData) -> StagePayload:
        import torch

        ar_state = get_ming_ar_state(data)
        payload = data.stage_payload
        state = data.state or MingTTSState.from_dict(payload.data)
        if not ar_state.generated_latents:
            raise ValueError(
                "Ming-Omni-TTS engine finished without generated latent chunks"
            )

        latent_chunks: list[Any] = []
        for latent in ar_state.generated_latents:
            tensor = (
                latent.detach()
                if hasattr(latent, "detach")
                else torch.as_tensor(latent)
            )
            if tensor.ndim == 3 and int(tensor.shape[0]) == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(
                    "Ming-Omni-TTS generated latent chunks must have shape "
                    "[patch_size, latent_dim]"
                )
            latent_chunks.append(tensor)

        raw = data.finish_reason
        if raw is None and data.req is not None:
            finished_reason = getattr(data.req, "finished_reason", None)
            if finished_reason is not None and hasattr(finished_reason, "to_json"):
                raw = finished_reason.to_json().get("type")
            elif finished_reason is not None:
                raw = str(finished_reason)

        normalized = str(raw).lower() if raw is not None else None
        if ar_state.stop_step is not None:
            if ar_state.generated_last_chunk and not bool(
                ar_state.generated_last_chunk[-1]
            ):
                raise RuntimeError(
                    "Ming-Omni-TTS stop_step was recorded but the final latent "
                    "chunk is not marked as last_chunk"
                )
            if normalized is not None and not (
                normalized in ("stop", "matched", "eos", "finish_matched_token")
                or "matched" in normalized
                or "eos" in normalized
            ):
                raise RuntimeError(
                    "Ming-Omni-TTS stop-head recorded stop_step="
                    f"{ar_state.stop_step}, but SGLang finished_reason was {raw!r}"
                )
            finish_reason = "stop"
        elif normalized is not None:
            if "length" in normalized:
                finish_reason = "length"
            elif "abort" in normalized:
                finish_reason = "abort"
            elif "error" in normalized:
                finish_reason = "error"
            elif normalized in ("stop", "matched", "eos", "finish_matched_token"):
                raise RuntimeError(
                    "Ming-Omni-TTS SGLang reported a stop finish without a "
                    "runner stop_step"
                )
            else:
                finish_reason = str(raw)
        elif len(ar_state.generated_latents) >= int(ar_state.max_decode_steps):
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
            bool(item) for item in ar_state.generated_last_chunk
        ]
        state.stop_step = ar_state.stop_step
        state.finish_reason = finish_reason
        state.prompt_tokens = prompt_tokens
        state.completion_tokens = len(latent_chunks)
        state.engine_time_s = time.perf_counter() - ar_state.engine_start_s

        for field_name, value in encode_generated_latents(generated).items():
            setattr(state, field_name, value)

        result = StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )
        release_ming_ar_runtime_tensors(data)
        return result

    return request_builder, result_adapter


__all__ = [
    "MingTTSSGLangRequestData",
    "make_ming_tts_scheduler_adapters",
]
