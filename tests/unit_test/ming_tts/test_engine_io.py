# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_tts import engine_io
from sglang_omni.models.ming_tts.audio_decode import MingAudioDecoderState
from sglang_omni.models.ming_tts.engine_io import (
    MingTTSLatentPatch,
    MingTTSSGLangRequestData,
    build_ming_tts_stream_output,
    make_ming_tts_scheduler_adapters,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.models.ming_tts.streaming_vocoder import (
    MingTTSStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


def _payload() -> StagePayload:
    state = MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2)
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def _result_adapter(reset_request):
    model = SimpleNamespace(patch_size=2, latent_dim=3)
    _, result_adapter = make_ming_tts_scheduler_adapters(
        model=model,
        tokenizer=SimpleNamespace(),
        reset_request=reset_request,
        prompt_radix_cache_enabled=False,
    )
    return result_adapter


def _request_adapter(*, prompt_radix_cache_enabled: bool = False):
    request_adapter, _ = make_ming_tts_scheduler_adapters(
        model=SimpleNamespace(vocab_size=128),
        tokenizer=SimpleNamespace(
            special=SimpleNamespace(end_of_audio=5, audio_patch=3)
        ),
        reset_request=lambda _: None,
        prompt_radix_cache_enabled=prompt_radix_cache_enabled,
    )
    return request_adapter


def _adapt_state(
    state: MingTTSState,
    *,
    request_id: str,
    prompt_radix_cache_enabled: bool,
) -> MingTTSSGLangRequestData:
    return _request_adapter(prompt_radix_cache_enabled=prompt_radix_cache_enabled)(
        StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello"),
            data=state.to_dict(),
        )
    )


def _request_data(
    *,
    generated_latents: torch.Tensor | None = None,
    stop_step: int | None = None,
    finish_reason=None,
    req_finished_reason=None,
) -> MingTTSSGLangRequestData:
    return MingTTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=req_finished_reason,
        ),
        state=MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2),
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        max_new_tokens=2,
        generated_latents=generated_latents,
        stop_step=stop_step,
        finish_reason=finish_reason,
        stage_payload=_payload(),
    )


class _FakeStreamingDecoder:
    sample_rate = 44100

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, bool]] = []

    def decode_streaming_step(
        self,
        latent_sequence: torch.Tensor,
        *,
        state: MingAudioDecoderState,
        is_last: bool,
    ) -> torch.Tensor:
        del state
        self.calls.append((latent_sequence.clone(), is_last))
        return torch.ones(latent_sequence.shape[0], dtype=torch.float32)


def test_ming_tts_result_adapter_serializes_empty_latent_output() -> None:
    reset_requests = []

    payload = _result_adapter(reset_requests.append)(_request_data())
    restored = MingTTSState.from_dict(payload.data)
    latents = restored.generated_latents

    assert latents is not None
    assert latents.shape == (0, 2, 3)
    assert restored.completion_tokens == 0
    assert restored.finish_reason == "stop"
    assert reset_requests == ["req-ming-tts"]


def test_ming_tts_request_builder_skips_radix_key_hash_when_disabled(
    monkeypatch,
) -> None:
    def fail_hash(*_args, **_kwargs):
        raise AssertionError("Radix-key hashing must be disabled")

    monkeypatch.setattr(engine_io.hashlib, "blake2b", fail_hash)
    states = [
        MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2),
        MingTTSState(
            text="hello",
            ref_audio="ref.wav",
            input_ids=[1, 2, 3],
            max_decode_steps=2,
            spk_emb=torch.full((1, 3), 1.0),
            prompt_latent=torch.full((1, 2, 3), 2.0),
            spk_injection_positions=[1],
            prompt_latent_start_position=2,
            prompt_latent_token_count=2,
        ),
    ]

    for index, state in enumerate(states):
        data = _adapt_state(
            state,
            request_id=f"cache-off-{index}",
            prompt_radix_cache_enabled=False,
        )
        assert data.req.extra_key is None


def test_ming_tts_request_builder_namespaces_enabled_prompt_cache() -> None:
    text = MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2)

    def reference_state(marker: int) -> MingTTSState:
        return MingTTSState(
            text="hello",
            ref_audio=f"ref-{marker}.wav",
            input_ids=[1, 2, 3],
            max_decode_steps=2,
            spk_emb=torch.full((1, 3), float(marker)),
            prompt_latent=torch.full((1, 2, 3), float(marker + 1)),
            prompt_conditioning_digest=f"conditioning-{marker}",
            spk_injection_positions=[1],
            prompt_latent_start_position=2,
            prompt_latent_token_count=2,
        )

    text_key = _adapt_state(
        text,
        request_id="text",
        prompt_radix_cache_enabled=True,
    ).req.extra_key
    reference_key = _adapt_state(
        reference_state(1),
        request_id="reference-a",
        prompt_radix_cache_enabled=True,
    ).req.extra_key
    repeated_key = _adapt_state(
        reference_state(1),
        request_id="reference-a-repeat",
        prompt_radix_cache_enabled=True,
    ).req.extra_key
    other_key = _adapt_state(
        reference_state(2),
        request_id="reference-b",
        prompt_radix_cache_enabled=True,
    ).req.extra_key

    assert text_key == "ming-tts:prompt:v2:text"
    assert reference_key is not None
    assert reference_key.startswith("ming-tts:prompt:v2:reference:")
    assert repeated_key == reference_key
    assert other_key != reference_key


@pytest.mark.parametrize(
    ("state_kwargs", "enabled", "error"),
    [
        pytest.param(
            {
                "ref_audio": "ref.wav",
                "spk_emb": torch.ones(1, 3),
            },
            False,
            "missing reference conditioning",
            id="incomplete-reference-bundle",
        ),
        pytest.param(
            {
                "ref_audio": "ref.wav",
                "spk_emb": torch.ones(1, 3),
                "prompt_latent": torch.ones(1, 2, 3),
            },
            True,
            "missing the prompt cache digest",
            id="enabled-without-producer-digest",
        ),
        pytest.param(
            {
                "spk_emb": torch.ones(1, 3),
                "prompt_latent": torch.ones(1, 2, 3),
            },
            False,
            "unexpectedly contains reference conditioning",
            id="text-with-reference-bundle",
        ),
    ],
)
def test_ming_tts_request_builder_rejects_invalid_reference_contract(
    state_kwargs: dict,
    enabled: bool,
    error: str,
) -> None:
    state = MingTTSState(
        text="hello",
        input_ids=[1, 2, 3],
        max_decode_steps=2,
        **state_kwargs,
    )

    with pytest.raises(ValueError, match=error):
        _adapt_state(
            state,
            request_id="invalid-reference",
            prompt_radix_cache_enabled=enabled,
        )


def test_ming_tts_result_adapter_prefers_stop_head_finish_reason() -> None:
    data = _request_data(
        generated_latents=torch.ones(1, 2, 3),
        stop_step=0,
        finish_reason="length",
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "stop"
    assert restored.stop_step == 0
    assert restored.completion_tokens == 1


def test_ming_tts_result_adapter_preserves_sglang_length_finish_reason() -> None:
    class FinishedReason:
        def to_json(self):
            return {"type": "length"}

    data = _request_data(
        generated_latents=torch.ones(1, 2, 3),
        req_finished_reason=FinishedReason(),
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.stop_step is None


def test_ming_tts_result_adapter_infers_length_at_max_steps() -> None:
    data = _request_data(
        generated_latents=torch.stack(
            (torch.ones(2, 3), torch.ones(2, 3) * 2),
            dim=0,
        ),
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.completion_tokens == 2


def test_ming_tts_stream_output_consumes_pending_patch_once() -> None:
    data = _request_data()
    data.is_streaming = True
    data.pending_stream_patch = MingTTSLatentPatch(
        latent=torch.ones((2, 3), dtype=torch.float64),
        is_last=True,
    )

    messages = build_ming_tts_stream_output("req-ming-tts", data, None)

    assert len(messages) == 1
    assert messages[0].request_id == "req-ming-tts"
    assert messages[0].target == "audio_decode"
    assert messages[0].metadata == {
        "modality": "audio_codes",
        "stream": True,
        "is_last": True,
    }
    assert messages[0].data.device.type == "cpu"
    assert messages[0].data.dtype == torch.float32
    assert data.pending_stream_patch is None
    assert build_ming_tts_stream_output("req-ming-tts", data, None) == []


def test_ming_tts_streaming_vocoder_initial_and_terminal_cadence() -> None:
    decoder = _FakeStreamingDecoder()
    scheduler = MingTTSStreamingVocoderScheduler(
        decoder,
        patch_size=2,
        latent_dim=3,
        steady_chunk_patches=2,
    )
    request_id = "req-ming-tts"
    payload = _payload()
    payload.request.params["stream"] = True

    def stream_item(chunk_id: int, value: float, *, is_last: bool) -> StreamItem:
        return StreamItem(
            chunk_id=chunk_id,
            data=torch.full((2, 3), value),
            from_stage="tts_engine",
            metadata={
                "modality": "audio_codes",
                "stream": True,
                "is_last": is_last,
            },
        )

    scheduler.on_stream_chunk_batch([(request_id, stream_item(0, 1.0, is_last=False))])
    assert len(decoder.calls) == 1

    scheduler.on_stream_chunk_batch([(request_id, stream_item(1, 2.0, is_last=False))])
    assert len(decoder.calls) == 1

    scheduler.on_stream_chunk_batch([(request_id, stream_item(2, 3.0, is_last=True))])
    state = scheduler._stream_states[request_id]
    assert state.terminal_decoded is True
    assert state.emitted_samples == 6

    scheduler._on_done(request_id)
    assert request_id in scheduler._stream_states
    assert request_id in scheduler._pending_done
    scheduler._on_streaming_new_request(request_id, payload)

    assert [(tuple(latent.shape), is_last) for latent, is_last in decoder.calls] == [
        ((2, 3), False),
        ((4, 3), True),
    ]
    assert torch.equal(decoder.calls[1][0][:2], torch.full((2, 3), 2.0))
    assert torch.equal(decoder.calls[1][0][2:], torch.full((2, 3), 3.0))
    assert request_id not in scheduler._stream_states

    outputs = [scheduler.outbox.get_nowait() for _ in range(3)]
    assert [output.type for output in outputs] == ["stream", "stream", "result"]
    assert outputs[-1].data.data["duration_s"] == pytest.approx(6 / 44100)
    assert "audio_waveform" not in outputs[-1].data.data
    assert scheduler.outbox.empty()


def test_ming_tts_result_adapter_resets_state_after_serialization_error(
    monkeypatch,
) -> None:
    reset_requests = []

    def fail_serialization(*_args):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(engine_io, "store_ming_tts_state", fail_serialization)
    data = _request_data(generated_latents=torch.ones(1, 2, 3))

    with pytest.raises(RuntimeError, match="serialization failed"):
        _result_adapter(reset_requests.append)(data)

    assert reset_requests == ["req-ming-tts"]
