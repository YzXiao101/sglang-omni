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
    )
    return result_adapter


def _request_adapter():
    request_adapter, _ = make_ming_tts_scheduler_adapters(
        model=SimpleNamespace(vocab_size=128),
        tokenizer=SimpleNamespace(
            special=SimpleNamespace(end_of_audio=5, audio_patch=3)
        ),
        reset_request=lambda _: None,
    )
    return request_adapter


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


def test_ming_tts_request_builder_namespaces_cache_by_reference() -> None:
    request_adapter = _request_adapter()

    def extra_key(request_id: str, speaker: float, latent: float) -> str:
        state = MingTTSState(
            text="hello",
            input_ids=[1, 2, 3],
            max_decode_steps=2,
            spk_emb=torch.full((1, 3), speaker),
            prompt_latent=torch.full((2, 3), latent),
            spk_injection_positions=[1],
            prompt_latent_start_position=2,
            prompt_latent_token_count=2,
        )
        data = request_adapter(
            StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs="hello"),
                data=state.to_dict(),
            )
        )
        return data.req.extra_key

    key = extra_key("reference-a", 1.0, 2.0)

    assert extra_key("reference-b", 1.0, 2.0) == key
    assert extra_key("speaker-changed", 3.0, 2.0) != key
    assert extra_key("latent-changed", 1.0, 4.0) != key


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
    state = scheduler.create_stream_state("req-ming-tts")
    participants = [("req-ming-tts", state)]

    scheduler.ingest("req-ming-tts", state, torch.full((2, 3), 1.0))
    scheduler.run_step(participants, scheduler.build_step_plan(participants))

    scheduler.ingest("req-ming-tts", state, torch.full((2, 3), 2.0))
    assert scheduler._step_plan_for_state(state) is None

    scheduler.ingest("req-ming-tts", state, torch.full((2, 3), 3.0))
    state.terminal_received = True
    scheduler.run_step(participants, scheduler.build_step_plan(participants))

    assert [(tuple(latent.shape), is_last) for latent, is_last in decoder.calls] == [
        ((2, 3), False),
        ((4, 3), True),
    ]
    assert torch.equal(decoder.calls[1][0][:2], torch.full((2, 3), 2.0))
    assert torch.equal(decoder.calls[1][0][2:], torch.full((2, 3), 3.0))


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
