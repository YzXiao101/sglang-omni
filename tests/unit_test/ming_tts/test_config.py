# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)


def _audio_decode_stage(raw_config: dict[str, Any]) -> dict[str, Any]:
    return next(
        stage for stage in raw_config["stages"] if stage["name"] == AUDIO_DECODE_STAGE
    )


def test_ming_tts_pipeline_requires_audio_decode_stream_edge() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    tts_engine = next(
        stage for stage in raw["stages"] if stage["name"] == TTS_ENGINE_STAGE
    )
    assert tts_engine["stream_to"] == [AUDIO_DECODE_STAGE]

    tts_engine["stream_to"] = []
    with pytest.raises(
        ValueError,
        match="tts_engine stream_to must include 'audio_decode'",
    ):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_pipeline_requires_audio_decode_stream_capability() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    audio_decode = _audio_decode_stage(raw)
    assert audio_decode["can_accept_stream_before_payload"] is True

    audio_decode["can_accept_stream_before_payload"] = False
    with pytest.raises(
        ValueError,
        match="audio_decode must set can_accept_stream_before_payload=true",
    ):
        MingTTSPipelineConfig.model_validate(raw)
