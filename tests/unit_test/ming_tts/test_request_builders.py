# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.models.ming_tts.payload_types import MING_TTS_DEFAULT_MAX_DECODE_STEPS
from sglang_omni.models.ming_tts.request_builders import preprocess_ming_tts_payload
from sglang_omni.models.ming_tts.tokenizer import (
    AUDIO_START_TOKEN,
    MingTTSSpecialTokenIds,
    MingTTSTokenizerBundle,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in ("<role>HUMAN</role>", "<role>ASSISTANT</role>"):
            return [1, 2]
        if text == AUDIO_START_TOKEN:
            return [4]
        if text:
            return [10]
        return []

    def __len__(self) -> int:
        return 128


def _tokenizer() -> MingTTSTokenizerBundle:
    return MingTTSTokenizerBundle(
        tokenizer=_FakeTokenizer(),
        special=MingTTSSpecialTokenIds(
            bos=8,
            eos=9,
            pad=9,
            role_start=1,
            role_end=2,
            audio_patch=3,
            audio_start=4,
            end_of_audio=5,
            spk_start=6,
            spk_end=7,
        ),
    )


def _payload(*, params: dict | None = None, tts_params: dict | None = None):
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(
            inputs="hello",
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


@pytest.mark.parametrize(
    ("params", "tts_params"),
    [
        ({}, {"seed": 1}),
        ({"seed": 1}, {}),
        ({"stage_params": {"tts_engine": {"seed": 1}}}, {}),
    ],
)
def test_ming_tts_rejects_seed_until_fl_rng_contract_exists(
    params: dict,
    tts_params: dict,
) -> None:
    with pytest.raises(ValueError, match="seed is currently unsupported"):
        preprocess_ming_tts_payload(
            _payload(params=params, tts_params=tts_params),
            tokenizer=_tokenizer(),
            context_length=MING_TTS_DEFAULT_MAX_DECODE_STEPS + 64,
        )
