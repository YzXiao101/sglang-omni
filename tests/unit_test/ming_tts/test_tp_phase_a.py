# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.config.schema import ParallelismConfig, StageConfig
from sglang_omni.models.ming_tts.config import (
    AUDIO_DECODE_STAGE,
    PREPROCESSING_STAGE,
    REFERENCE_ENCODE_STAGE,
    TTS_ENGINE_STAGE,
    MingTTSPipelineConfig,
)


def _stage(config: MingTTSPipelineConfig, name: str) -> StageConfig:
    return next(stage for stage in config.stages if stage.name == name)


def _stage_from_list(stages: list[StageConfig], name: str) -> StageConfig:
    return next(stage for stage in stages if stage.name == name)


def _copied_stages() -> list[StageConfig]:
    return [
        stage.model_copy(deep=True)
        for stage in MingTTSPipelineConfig(model_path="dummy").stages
    ]


def test_ming_tts_config_allows_tts_engine_tp2_only() -> None:
    stages = _copied_stages()
    tts_engine = next(stage for stage in stages if stage.name == TTS_ENGINE_STAGE)
    tts_engine.gpu = [0, 1]
    tts_engine.tp_size = 2
    tts_engine.parallelism.tp = 2

    config = MingTTSPipelineConfig(model_path="dummy", stages=stages)

    assert _stage(config, TTS_ENGINE_STAGE).tp_size == 2
    assert _stage(config, TTS_ENGINE_STAGE).parallelism.tp == 2


def test_ming_tts_config_rejects_non_tts_engine_tp() -> None:
    stages = _copied_stages()
    reference_encode = next(
        stage for stage in stages if stage.name == REFERENCE_ENCODE_STAGE
    )
    reference_encode.gpu = [0, 1]
    reference_encode.tp_size = 2
    reference_encode.parallelism.tp = 2

    with pytest.raises(ValueError, match="supports tensor parallelism only"):
        MingTTSPipelineConfig(model_path="dummy", stages=stages)


def test_ming_tts_config_rejects_tts_engine_tp2_without_gpu_list() -> None:
    stages = _copied_stages()
    tts_engine = next(stage for stage in stages if stage.name == TTS_ENGINE_STAGE)
    tts_engine.gpu = 0
    tts_engine.tp_size = 2
    tts_engine.parallelism.tp = 2

    with pytest.raises(ValueError, match="requires gpu=\\[rank0_gpu"):
        MingTTSPipelineConfig(model_path="dummy", stages=stages)


def test_ming_tts_tp2_topology_uses_explicit_rank_processes() -> None:
    stages = _copied_stages()
    preprocessing = _stage_from_list(stages, PREPROCESSING_STAGE)
    reference_encode = _stage_from_list(stages, REFERENCE_ENCODE_STAGE)
    tts_engine = _stage_from_list(stages, TTS_ENGINE_STAGE)
    audio_decode = _stage_from_list(stages, AUDIO_DECODE_STAGE)

    preprocessing.process = PREPROCESSING_STAGE
    reference_encode.process = "ming_tts_aux"
    audio_decode.process = "ming_tts_aux"
    tts_engine.process = TTS_ENGINE_STAGE

    reference_encode.runtime.resources.total_gpu_memory_fraction = 0.08
    audio_decode.runtime.resources.total_gpu_memory_fraction = 0.12
    tts_engine.runtime.resources.total_gpu_memory_fraction = 0.72

    tts_engine.gpu = [0, 1]
    tts_engine.tp_size = 2
    tts_engine.parallelism = ParallelismConfig(tp=2)
    config = MingTTSPipelineConfig(model_path="dummy", stages=stages)

    assert preprocessing.process == PREPROCESSING_STAGE
    assert reference_encode.process == "ming_tts_aux"
    assert audio_decode.process == "ming_tts_aux"
    assert tts_engine.process == TTS_ENGINE_STAGE
    assert tts_engine.gpu == [0, 1]
    assert tts_engine.tp_size == 2
    assert tts_engine.parallelism.tp == 2
    assert tts_engine.factory_args["gpu_id"] == 0
    assert "total_gpu_memory_fraction" not in tts_engine.factory_args
    assert tts_engine.runtime.resources.total_gpu_memory_fraction == pytest.approx(0.72)
    assert (
        reference_encode.runtime.resources.total_gpu_memory_fraction
        == pytest.approx(0.08)
    )
    assert audio_decode.runtime.resources.total_gpu_memory_fraction == pytest.approx(
        0.12
    )

    placement = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, placement)

    assert topology.tp_stage_to_processes[TTS_ENGINE_STAGE] == (
        "tts_engine_tp0",
        "tts_engine_tp1",
    )
    assert placement.gpus[0].total_gpu_memory_fraction == pytest.approx(0.92)
    assert placement.gpus[1].total_gpu_memory_fraction == pytest.approx(0.72)


def test_ming_tts_tp2_topology_requires_colocated_memory_budgets() -> None:
    stages = _copied_stages()
    reference_encode = _stage_from_list(stages, REFERENCE_ENCODE_STAGE)
    audio_decode = _stage_from_list(stages, AUDIO_DECODE_STAGE)
    tts_engine = _stage_from_list(stages, TTS_ENGINE_STAGE)

    reference_encode.process = "ming_tts_aux"
    audio_decode.process = "ming_tts_aux"
    tts_engine.process = TTS_ENGINE_STAGE
    tts_engine.gpu = [0, 1]
    tts_engine.tp_size = 2
    tts_engine.parallelism = ParallelismConfig(tp=2)
    config = MingTTSPipelineConfig(model_path="dummy", stages=stages)
    placement = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="without runtime.resources"):
        build_process_topology_plan(config, placement)
