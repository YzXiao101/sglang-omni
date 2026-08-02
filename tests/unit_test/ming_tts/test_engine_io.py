# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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
from sglang_omni.models.ming_tts.model_runner import (
    MingTTSModelRunner,
    _MingTTSRequestState,
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

    def __init__(self, *, first_call_empty: bool = False) -> None:
        self.first_call_empty = first_call_empty
        self.calls: list[tuple[torch.Tensor, bool]] = []
        self.states: list[MingAudioDecoderState] = []

    def decode_streaming_step(
        self,
        latent_sequence: torch.Tensor,
        *,
        state: MingAudioDecoderState,
        is_last: bool,
    ) -> torch.Tensor:
        self.calls.append((latent_sequence.clone(), is_last))
        self.states.append(state)
        if self.first_call_empty and len(self.calls) == 1:
            return torch.empty(0, dtype=torch.float32)
        return latent_sequence[:, 0].clone()


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


def test_ming_tts_request_builder_preserves_prompt_ids_when_cache_disabled() -> None:
    states = [
        MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2),
        MingTTSState(
            text="hello",
            ref_audio="ref.wav",
            input_ids=[11, 3, 3, 3],
            max_decode_steps=2,
            spk_emb=torch.full((1, 3), 1.0),
            prompt_latent=torch.full((1, 4, 3), 2.0),
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
        assert list(data.req.origin_input_ids) == state.input_ids
        assert data.input_ids.tolist() == state.input_ids


def test_ming_tts_request_builder_namespaces_text_prompt_cache() -> None:
    text = MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2)
    data = _adapt_state(
        text,
        request_id="text",
        prompt_radix_cache_enabled=True,
    )

    assert data.req.extra_key == "ming-tts:prompt:v3:text"
    assert list(data.req.origin_input_ids) == [1, 2, 3]
    assert data.input_ids.tolist() == [1, 2, 3]


def test_ming_tts_request_builder_tags_reference_prompt_cache_rows() -> None:
    actual_ids = [11, 3, 3, 3]

    def reference_state(marker: int) -> MingTTSState:
        return MingTTSState(
            text="hello",
            ref_audio=f"ref-{marker}.wav",
            input_ids=actual_ids,
            max_decode_steps=2,
            spk_emb=torch.full((1, 3), float(marker)),
            prompt_latent=torch.full((1, 4, 3), float(marker + 1)),
            prompt_conditioning_digest=f"conditioning-{marker}",
            spk_injection_positions=[1],
            prompt_latent_start_position=2,
            prompt_latent_token_count=2,
        )

    reference = _adapt_state(
        reference_state(1),
        request_id="reference-a",
        prompt_radix_cache_enabled=True,
    )
    repeated = _adapt_state(
        reference_state(1),
        request_id="reference-a-repeat",
        prompt_radix_cache_enabled=True,
    )
    other = _adapt_state(
        reference_state(2),
        request_id="reference-b",
        prompt_radix_cache_enabled=True,
    )

    assert reference.req.extra_key == ("ming-tts:prompt:v3:reference:conditioning-1")
    assert repeated.req.extra_key == reference.req.extra_key
    assert other.req.extra_key == "ming-tts:prompt:v3:reference:conditioning-2"
    assert reference.input_ids.tolist() == actual_ids
    assert list(reference.req.origin_input_ids) == [11, 128, 129, 131]


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

    scheduler._on_chunk(request_id, stream_item(0, 1.0, is_last=False))
    assert len(decoder.calls) == 1

    scheduler._on_chunk(request_id, stream_item(1, 2.0, is_last=False))
    assert len(decoder.calls) == 1

    scheduler._on_chunk(request_id, stream_item(2, 3.0, is_last=True))
    state = scheduler._stream_states[request_id]
    assert state.terminal_received is True
    assert state.pending_patches == []
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


def test_ming_tts_streaming_retraction_preserves_latent_and_decoder_state() -> None:
    request_id = "req-ming-tts-retracted"
    data = _adapt_state(
        MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=3),
        request_id=request_id,
        prompt_radix_cache_enabled=False,
    )
    data.is_streaming = True
    data.stage_payload.request.params["stream"] = True

    decoder = _FakeStreamingDecoder(first_call_empty=True)
    scheduler = MingTTSStreamingVocoderScheduler(
        decoder,
        patch_size=2,
        latent_dim=3,
        steady_chunk_patches=2,
    )
    chunk_id = 0

    def publish(value: float, *, is_last: bool) -> None:
        nonlocal chunk_id
        data.pending_stream_patch = MingTTSLatentPatch(
            latent=torch.full((2, 3), value),
            is_last=is_last,
        )
        messages = build_ming_tts_stream_output(request_id, data, None)
        assert len(messages) == 1
        assert build_ming_tts_stream_output(request_id, data, None) == []
        message = messages[0]
        scheduler._on_chunk(
            request_id,
            StreamItem(
                chunk_id=chunk_id,
                data=message.data,
                from_stage="tts_engine",
                metadata=message.metadata,
            ),
        )
        chunk_id += 1

    publish(1.0, is_last=False)
    decoder_state = scheduler._stream_states[request_id].decoder_state
    assert decoder.calls[0][0].tolist() == [[1.0] * 3, [1.0] * 3]
    assert scheduler.outbox.empty()

    request_state = _MingTTSRequestState(
        feedback_embeddings=[torch.full((3,), 4.0)],
        latent_history=torch.full((1, 2, 3), 5.0),
    )
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner._request_states = {request_id: request_state}
    data.generation_steps = 1
    data.req.output_ids.append(data.audio_patch_token_id)
    data.req.reset_for_retract()
    runner.before_prefill(
        None,
        None,
        [SimpleNamespace(request_id=request_id, data=data)],
    )

    assert runner._request_states[request_id] is request_state
    assert request_state.feedback_embeddings[0].tolist() == [4.0, 4.0, 4.0]
    assert torch.all(request_state.latent_history == 5.0)
    assert data.generation_steps == 1
    assert list(data.req.output_ids) == [data.audio_patch_token_id]
    assert data.req.skip_radix_cache_insert is True

    publish(2.0, is_last=False)
    assert len(decoder.calls) == 1
    publish(3.0, is_last=True)

    scheduler._on_done(request_id)
    assert request_id in scheduler._pending_done
    scheduler._on_streaming_new_request(request_id, data.stage_payload)

    assert len(decoder.states) == 2
    assert all(state is decoder_state for state in decoder.states)
    assert [is_last for _, is_last in decoder.calls] == [False, True]
    assert decoder.calls[1][0].tolist() == [
        [2.0] * 3,
        [2.0] * 3,
        [3.0] * 3,
        [3.0] * 3,
    ]
    assert request_id not in scheduler._stream_states

    stream_output = scheduler.outbox.get_nowait()
    result_output = scheduler.outbox.get_nowait()
    assert stream_output.type == "stream"
    assert result_output.type == "result"
    audio = np.frombuffer(stream_output.data["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(audio, [2.0, 2.0, 3.0, 3.0])
    assert result_output.data.data["duration_s"] == pytest.approx(4 / 44100)
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
