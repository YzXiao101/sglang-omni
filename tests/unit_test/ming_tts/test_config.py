# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)
from sglang_omni.models.ming_tts.engine_builder import MingTtsEngineBuilder
from tests.unit_test.fakes import FakeServerArgs


@pytest.mark.parametrize(
    ("factory_value", "runtime_value", "expected_disable"),
    [
        (None, None, True),
        (False, None, False),
        (True, False, False),
    ],
)
def test_ming_tts_pipeline_resolves_prompt_cache_policy(
    factory_value: bool | None,
    runtime_value: bool | None,
    expected_disable: bool,
) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    stages = {stage["name"]: stage for stage in raw["stages"]}
    reference_args = stages[REFERENCE_ENCODE_STAGE]["factory_args"]
    tts_args = stages[TTS_ENGINE_STAGE]["factory_args"]
    reference_args.pop("compute_prompt_cache_digest")
    tts_args.pop("expected_disable_radix_cache")

    server_args = tts_args["server_args_overrides"]
    if factory_value is None:
        server_args.pop("disable_radix_cache")
    else:
        server_args["disable_radix_cache"] = factory_value
    if runtime_value is not None:
        raw["runtime_overrides"][TTS_ENGINE_STAGE] = {
            "server_args_overrides": {"disable_radix_cache": runtime_value}
        }

    config = MingTTSPipelineConfig.model_validate(raw)
    resolved_stages = {stage.name: stage for stage in config.stages}
    resolved_reference_args = resolve_stage_static_factory_args(
        resolved_stages[REFERENCE_ENCODE_STAGE], config
    )
    resolved_tts_args = resolve_stage_static_factory_args(
        resolved_stages[TTS_ENGINE_STAGE], config
    )

    assert (
        resolved_tts_args["server_args_overrides"]["disable_radix_cache"]
        is expected_disable
    )
    assert resolved_tts_args["expected_disable_radix_cache"] is expected_disable
    assert resolved_reference_args["compute_prompt_cache_digest"] is (
        not expected_disable
    )
    assert (
        resolved_stages[TTS_ENGINE_STAGE].factory_args["server_args_overrides"][
            "disable_radix_cache"
        ]
        is expected_disable
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("non_mapping_server_args", "server_args_overrides must be a mapping"),
        ("non_boolean_policy", "disable_radix_cache must be a boolean"),
        ("reference_policy_override", "owned by the pipeline config"),
        ("tts_policy_override", "owned by the pipeline config"),
    ],
)
def test_ming_tts_pipeline_rejects_invalid_prompt_cache_policy(
    mutation: str,
    error: str,
) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    stages = {stage["name"]: stage for stage in raw["stages"]}

    if mutation == "non_mapping_server_args":
        stages[TTS_ENGINE_STAGE]["factory_args"]["server_args_overrides"] = []
    elif mutation == "non_boolean_policy":
        stages[TTS_ENGINE_STAGE]["factory_args"]["server_args_overrides"] = {
            "disable_radix_cache": "false"
        }
    elif mutation == "reference_policy_override":
        raw["runtime_overrides"][REFERENCE_ENCODE_STAGE] = {
            "compute_prompt_cache_digest": True
        }
    else:
        raw["runtime_overrides"][TTS_ENGINE_STAGE] = {
            "expected_disable_radix_cache": False
        }

    with pytest.raises(ValueError, match=error):
        MingTTSPipelineConfig.model_validate(raw)


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


@pytest.mark.parametrize("expected", [False, True])
@pytest.mark.parametrize("actual", [False, True])
def test_ming_tts_builder_validates_final_prompt_cache_policy(
    expected: bool,
    actual: bool,
) -> None:
    builder = MingTtsEngineBuilder(expected_disable_radix_cache=expected)
    server_args = FakeServerArgs(disable_radix_cache=actual)

    if actual == expected:
        builder.customize_server_args(server_args)
    else:
        with pytest.raises(ValueError, match="conflicts with"):
            builder.customize_server_args(server_args)

    assert builder.infra_kwargs()["disable_finished_insert"] is True
