# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_ming_tts_audio_decode_defaults_are_full_sequence_and_serial() -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    factory_args = _audio_decode_stage(raw)["factory_args"]

    assert "decode_mode" not in factory_args
    assert factory_args["max_batch_size"] == 1
    assert factory_args["max_batch_wait_ms"] == 0


def test_ming_tts_example_config_uses_supported_audio_decode_contract() -> None:
    config_path = _REPO_ROOT / "examples/configs/ming_omni_tts.yaml"
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file)
    assert raw.pop("config_cls") == "MingTTSPipelineConfig"
    config = MingTTSPipelineConfig.model_validate(raw)
    factory_args = next(
        stage.factory_args
        for stage in config.stages
        if stage.name == AUDIO_DECODE_STAGE
    )

    assert "decode_mode" not in factory_args
    assert factory_args["max_batch_size"] == 1
    assert factory_args["max_batch_wait_ms"] == 0


@pytest.mark.parametrize("source", ["factory_args", "runtime_overrides"])
def test_ming_tts_rejects_legacy_audio_decode_mode(source: str) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    if source == "factory_args":
        _audio_decode_stage(raw)["factory_args"]["decode_mode"] = "chunked"
    else:
        raw["runtime_overrides"] = {AUDIO_DECODE_STAGE: {"decode_mode": "chunked"}}

    with pytest.raises(ValueError, match="no longer supports 'decode_mode'"):
        MingTTSPipelineConfig.model_validate(raw)


@pytest.mark.parametrize("source", ["factory_args", "runtime_overrides"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("max_batch_size", 2, "max_batch_size=1 only"),
        ("max_batch_wait_ms", 1, "max_batch_wait_ms=0 only"),
    ],
)
def test_ming_tts_rejects_unsupported_audio_decode_batch_config(
    source: str,
    field: str,
    value: int,
    error: str,
) -> None:
    raw = MingTTSPipelineConfig(model_path="fake-model").model_dump()
    if source == "factory_args":
        _audio_decode_stage(raw)["factory_args"][field] = value
    else:
        raw["runtime_overrides"] = {AUDIO_DECODE_STAGE: {field: value}}

    with pytest.raises(ValueError, match=error):
        MingTTSPipelineConfig.model_validate(raw)


def test_ming_tts_audio_decode_factory_exposes_only_supported_contract() -> None:
    from sglang_omni.models.ming_tts.stages import create_audio_decode_executor

    parameters = inspect.signature(create_audio_decode_executor).parameters

    assert "decode_mode" not in parameters
    assert parameters["max_batch_size"].default == 1
    assert parameters["max_batch_wait_ms"].default == 0


def test_ming_tts_legacy_engine_factory_alias_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.ming_tts import stages

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def fake_factory(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(stages, "create_sglang_tts_engine_executor", fake_factory)

    result = stages.create_tts_engine_executor("model", gpu_id=2)

    assert result is sentinel
    assert calls == [(("model",), {"gpu_id": 2})]


@pytest.mark.parametrize(
    ("factory_args", "error"),
    [
        ({"max_batch_size": 2}, "max_batch_size=1 only"),
        ({"max_batch_wait_ms": 1}, "max_batch_wait_ms=0 only"),
    ],
)
def test_ming_tts_audio_decode_factory_rejects_batch_config_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    factory_args: dict[str, int],
    error: str,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(ValueError, match=error):
        stages.create_audio_decode_executor("unused", **factory_args)


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
