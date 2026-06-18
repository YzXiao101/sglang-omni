# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni-TTS 16B."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.ming_tts"

PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
TTS_ENGINE_STAGE = "tts_engine"
AUDIO_DECODE_STAGE = "audio_decode"


class MingTTSPipelineConfig(PipelineConfig):
    """Ming-Omni-TTS pipeline: preprocessing -> TTS engine -> audio decode."""

    architecture: ClassVar[str] = "BailingMMNativeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"gpu_id": 0, "dtype": "bfloat16"},
            gpu=0,
            next=AUDIO_DECODE_STAGE,
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_decode_executor",
            factory_args={
                "gpu_id": 0,
                "dtype": "bfloat16",
                "decode_mode": "chunked",
            },
            gpu=0,
            terminal=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        for stage in self.stages:
            if stage.tp_size != 1:
                raise ValueError(
                    "Ming-Omni-TTS currently supports only tp_size=1; "
                    "AR TP/EP is currently unsupported. "
                    f"Stage {stage.name!r} has tp_size={stage.tp_size}."
                )


EntryClass = MingTTSPipelineConfig
