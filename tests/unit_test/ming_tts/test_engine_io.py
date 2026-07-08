# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.ming_tts.engine_io import (
    MingTTSDecodeState,
    MingTTSSGLangRequestData,
    make_ming_tts_scheduler_adapters,
)
from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    decode_generated_latents,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _payload() -> StagePayload:
    state = MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2)
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def _result_adapter():
    model = SimpleNamespace(patch_size=2, latent_dim=3)
    _, result_adapter = make_ming_tts_scheduler_adapters(
        model=model,
        tokenizer=SimpleNamespace(),
    )
    return result_adapter


def _request_data(
    decode_state: MingTTSDecodeState,
    *,
    finish_reason=None,
    req_finished_reason=None,
) -> MingTTSSGLangRequestData:
    return MingTTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=req_finished_reason,
        ),
        state=MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2),
        prompt_input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        decode_state=decode_state,
        finish_reason=finish_reason,
        stage_payload=_payload(),
    )


def test_ming_tts_result_adapter_serializes_empty_latent_output() -> None:
    data = _request_data(MingTTSDecodeState(max_decode_steps=2))

    payload = _result_adapter()(data)
    restored = MingTTSState.from_dict(payload.data)
    latents = decode_generated_latents(restored)

    assert latents is not None
    assert latents.shape == (0, 2, 3)
    assert restored.generated_last_chunk == []
    assert restored.completion_tokens == 0
    assert restored.finish_reason == "stop"


def test_ming_tts_result_adapter_prefers_stop_head_finish_reason() -> None:
    state = MingTTSDecodeState(max_decode_steps=2, stop_step=0)
    state.generated_latents.append(torch.ones(2, 3))
    state.generated_last_chunk.append(True)
    data = _request_data(state, finish_reason="length")

    payload = _result_adapter()(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "stop"
    assert restored.stop_step == 0
    assert restored.completion_tokens == 1


def test_ming_tts_result_adapter_preserves_sglang_length_finish_reason() -> None:
    class FinishedReason:
        def to_json(self):
            return {"type": "length"}

    state = MingTTSDecodeState(max_decode_steps=2)
    state.generated_latents.append(torch.ones(2, 3))
    state.generated_last_chunk.append(True)
    data = _request_data(state, req_finished_reason=FinishedReason())

    payload = _result_adapter()(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.stop_step is None


def test_ming_tts_result_adapter_infers_length_at_max_steps() -> None:
    state = MingTTSDecodeState(max_decode_steps=2)
    state.generated_latents.extend([torch.ones(2, 3), torch.ones(2, 3) * 2])
    state.generated_last_chunk.extend([False, True])
    data = _request_data(state)

    payload = _result_adapter()(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.completion_tokens == 2
