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
    """Ming-Omni-TTS pipeline.

    preprocessing -> reference_encode -> tts_engine -> audio_decode.
    The reference stage is kept as a fixed cheap/no-op boundary for text-only
    requests so reference-conditioned requests use the same serving graph.
    """

    architecture: ClassVar[str] = "BailingMMNativeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": TTS_ENGINE_STAGE}

    @classmethod
    def isolation_role_to_stage(cls) -> dict[str, str]:
        return {"vocoder": AUDIO_DECODE_STAGE}

    @classmethod
    def process_safe_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({(TTS_ENGINE_STAGE, AUDIO_DECODE_STAGE)})

    @classmethod
    def process_edge_resources(
        cls,
    ) -> dict[tuple[str, str], dict[str, float]]:
        return {
            (TTS_ENGINE_STAGE, AUDIO_DECODE_STAGE): {
                REFERENCE_ENCODE_STAGE: 0.08,
                TTS_ENGINE_STAGE: 0.72,
                AUDIO_DECODE_STAGE: 0.12,
            }
        }

    model_path: str
    max_decode_steps_cap: int | None = 256
    audio_vae_steady_chunk_patches: int = 2
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
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=AUDIO_DECODE_STAGE,
            stream_to=[AUDIO_DECODE_STAGE],
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_decode_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if self.max_decode_steps_cap is not None and self.max_decode_steps_cap <= 0:
            raise ValueError("Ming-Omni-TTS max_decode_steps_cap must be positive")
        if self.audio_vae_steady_chunk_patches <= 0:
            raise ValueError(
                "Ming-Omni-TTS audio_vae_steady_chunk_patches must be positive"
            )
        stages = {stage.name: stage for stage in self.stages}
        required_stages = {
            PREPROCESSING_STAGE,
            REFERENCE_ENCODE_STAGE,
            TTS_ENGINE_STAGE,
            AUDIO_DECODE_STAGE,
        }
        missing_stages = required_stages - stages.keys()
        if missing_stages:
            raise ValueError(
                "Ming-Omni-TTS pipeline is missing required stages: "
                f"{sorted(missing_stages)}"
            )
        preprocessing = stages[PREPROCESSING_STAGE]
        tts_engine = stages[TTS_ENGINE_STAGE]
        audio_decode = stages[AUDIO_DECODE_STAGE]
        if AUDIO_DECODE_STAGE not in tts_engine.stream_to:
            raise ValueError(
                "Ming-Omni-TTS tts_engine stream_to must include "
                f"{AUDIO_DECODE_STAGE!r}"
            )
        if not audio_decode.can_accept_stream_before_payload:
            raise ValueError(
                "Ming-Omni-TTS audio_decode must set "
                "can_accept_stream_before_payload=true because tts_engine sends "
                "stream data and stream_done before the terminal payload"
            )

        for stage, expected_args in (
            (
                preprocessing,
                {"max_decode_steps_cap": self.max_decode_steps_cap},
            ),
            (
                audio_decode,
                {
                    "audio_vae_steady_chunk_patches": (
                        self.audio_vae_steady_chunk_patches
                    ),
                },
            ),
        ):
            runtime_overrides = self.runtime_overrides.get(stage.name, {})
            for arg, expected_value in expected_args.items():
                if arg in runtime_overrides:
                    raise ValueError(
                        f"Ming-Omni-TTS {arg!r} is owned by the pipeline config, "
                        f"not stage {stage.name!r}"
                    )
                if (
                    arg in stage.factory_args
                    and stage.factory_args[arg] != expected_value
                ):
                    raise ValueError(
                        f"Ming-Omni-TTS stage {stage.name!r} {arg!r} conflicts "
                        "with the pipeline config"
                    )
                stage.factory_args[arg] = expected_value

        for stage in self.stages:
            if stage.name != TTS_ENGINE_STAGE:
                if stage.tp_size != 1:
                    raise ValueError(
                        "Ming-Omni-TTS supports tensor parallelism only on "
                        f"{TTS_ENGINE_STAGE!r}; stage {stage.name!r} has "
                        f"tp_size={stage.tp_size}."
                    )
                continue

            if stage.tp_size <= 0:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tp_size must be positive; "
                    f"got tp_size={stage.tp_size}."
                )
            if stage.tp_size == 1:
                continue
            if not isinstance(stage.gpu, list):
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tensor parallelism requires "
                    "gpu=[rank0_gpu, rank1_gpu, ...]."
                )
            if len(stage.gpu) != stage.tp_size:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine TP GPU list length must match "
                    f"tp_size; got gpu={stage.gpu!r}, tp_size={stage.tp_size}."
                )


EntryClass = MingTTSPipelineConfig
