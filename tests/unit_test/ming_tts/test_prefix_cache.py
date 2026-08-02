# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from sglang_omni.models.ming_tts.engine_io import (
    MingTTSSGLangRequestData,
    make_ming_tts_scheduler_adapters,
)
from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _make_radix_cache() -> tuple[RadixCache, ReqToTokenPool]:
    req_to_token_pool = ReqToTokenPool(
        size=4,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable_finished_insert=True,
        )
    )
    return cache, req_to_token_pool


def _build_reference_request(
    request_id: str,
    marker: int,
    *,
    input_ids: list[int] | None = None,
    latent_start: int = 2,
) -> MingTTSSGLangRequestData:
    if input_ids is None:
        input_ids = [11, 3, 3, 3]
    state = MingTTSState(
        text="hello",
        ref_audio=f"ref-{marker}.wav",
        input_ids=input_ids,
        max_decode_steps=4,
        spk_emb=torch.full((1, 3), float(marker)),
        prompt_latent=torch.full((1, 4, 3), float(marker + 1)),
        prompt_conditioning_digest=f"conditioning-{marker}",
        spk_injection_positions=[1],
        prompt_latent_start_position=latent_start,
        prompt_latent_token_count=2,
    )
    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=SimpleNamespace(vocab_size=128),
        tokenizer=SimpleNamespace(
            special=SimpleNamespace(end_of_audio=5, audio_patch=3)
        ),
        reset_request=lambda _request_id: None,
        prompt_radix_cache_enabled=True,
    )
    data = request_builder(
        StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello"),
            data=state.to_dict(),
        )
    )
    OmniScheduler._normalize_req_token_arrays(data.req)
    return data


def _cache_initial_prompt(
    data: MingTTSSGLangRequestData,
    cache: RadixCache,
    req_to_token_pool: ReqToTokenPool,
) -> torch.Tensor:
    req = data.req
    req.init_next_round_input(cache)
    cache.inc_lock_ref(req.last_node)
    req_to_token_pool.alloc([req])
    prompt_len = len(req.origin_input_ids)
    prompt_kv = torch.arange(100, 100 + prompt_len, dtype=torch.int64)
    req_to_token_pool.write(
        (req.req_pool_idx, slice(0, prompt_len)),
        prompt_kv,
    )
    req.set_extend_range(0, prompt_len)
    req.output_ids.append(data.audio_patch_token_id)
    maybe_cache_unfinished_req(req, cache)
    return prompt_kv


def test_ming_tts_prompt_cache_populates_reuses_and_isolates_reference() -> None:
    cache, req_to_token_pool = _make_radix_cache()
    first = _build_reference_request("reference-a", marker=1)
    prompt_kv = _cache_initial_prompt(first, cache, req_to_token_pool)
    prompt_len = len(first.req.origin_input_ids)

    assert cache.total_size() == prompt_len
    assert list(first.req.get_fill_ids()) == list(first.req.origin_input_ids)
    assert first.input_ids.tolist() == [11, 3, 3, 3]
    assert list(first.req.origin_input_ids) == [11, 128, 129, 131]

    repeated = _build_reference_request("reference-a-repeat", marker=1)
    repeated.req.init_next_round_input(cache)
    assert torch.equal(repeated.req.prefix_indices, prompt_kv[:-1])

    other = _build_reference_request("reference-b", marker=2)
    assert other.req.origin_input_ids == repeated.req.origin_input_ids
    other.req.init_next_round_input(cache)
    assert len(other.req.prefix_indices) == 0


def test_ming_tts_prompt_cache_reuses_prefix_before_target_layout_diverges() -> None:
    cache, req_to_token_pool = _make_radix_cache()
    short = _build_reference_request(
        "short-target",
        marker=1,
        input_ids=[10, 3, 20, 30, 3, 3],
        latent_start=4,
    )
    _cache_initial_prompt(short, cache, req_to_token_pool)

    long = _build_reference_request(
        "long-target",
        marker=1,
        input_ids=[10, 3, 20, 40, 30, 3, 3],
        latent_start=5,
    )
    match = cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(long.req.origin_input_ids, long.req.extra_key),
        )
    )

    assert list(short.req.origin_input_ids) == [10, 128, 20, 30, 129, 131]
    assert list(long.req.origin_input_ids) == [10, 128, 20, 40, 30, 129, 131]
    assert len(match.device_indices) == 3


def test_ming_tts_prompt_cache_distinguishes_literal_and_conditioning_rows() -> None:
    cache, req_to_token_pool = _make_radix_cache()
    conditioned = _build_reference_request(
        "conditioned-row",
        marker=1,
        input_ids=[10, 3, 20, 3, 3],
        latent_start=3,
    )
    _cache_initial_prompt(conditioned, cache, req_to_token_pool)

    literal = _build_reference_request(
        "literal-audio-patch",
        marker=1,
        input_ids=[10, 3, 20, 3, 4, 3, 3],
        latent_start=5,
    )
    match = cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(literal.req.origin_input_ids, literal.req.extra_key),
        )
    )

    assert list(conditioned.req.origin_input_ids) == [10, 128, 20, 129, 131]
    assert list(literal.req.origin_input_ids) == [10, 128, 20, 3, 4, 129, 131]
    assert len(match.device_indices) == 3


def test_ming_tts_retraction_does_not_extend_prompt_cache() -> None:
    cache, req_to_token_pool = _make_radix_cache()
    first = _build_reference_request("reference-a", marker=1)
    prompt_kv = _cache_initial_prompt(first, cache, req_to_token_pool)
    prompt_len = len(first.req.origin_input_ids)

    retracted = _build_reference_request("reference-a-retracted", marker=1)
    retracted.req.output_ids.append(retracted.audio_patch_token_id)
    retracted.req.reset_for_retract()
    retracted.req.init_next_round_input(cache)

    assert retracted.req.retracted_stain is True
    assert len(retracted.req.prefix_indices) == prompt_len

    cache.inc_lock_ref(retracted.req.last_node)
    req_to_token_pool.alloc([retracted.req])
    generated_kv = torch.tensor([200], dtype=torch.int64)
    req_to_token_pool.write(
        (retracted.req.req_pool_idx, slice(0, prompt_len)),
        prompt_kv,
    )
    req_to_token_pool.write(
        (retracted.req.req_pool_idx, slice(prompt_len, prompt_len + 1)),
        generated_kv,
    )
    retracted.req.set_extend_range(prompt_len, prompt_len + 1)

    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner._materialize_request_state = lambda _request: None
    scheduled_request = SimpleNamespace(data=retracted)
    runner.before_prefill(None, None, [scheduled_request])
    maybe_cache_unfinished_req(retracted.req, cache)

    full_key = RadixKey(
        retracted.req.full_untruncated_fill_ids,
        retracted.req.extra_key,
    )
    match = cache.match_prefix(MatchPrefixParams(key=full_key))

    assert retracted.req.skip_radix_cache_insert is True
    assert cache.total_size() == prompt_len
    assert len(match.device_indices) == prompt_len
