# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)


def test_torch_mps_uses_single_request_native_attention() -> None:
    builder = FunCosyVoice3EngineBuilder(enable_overlap=False)
    builder.device = "mps:0"

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["attention_backend"] == "torch_native"
    assert defaults["max_running_requests"] == 1

    with pytest.raises(ValueError, match="max_running_requests=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=2))


def cuda_overrides(
    **server_args_overrides: object,
) -> tuple[FunCosyVoice3EngineBuilder, dict[str, object]]:
    builder = FunCosyVoice3EngineBuilder(enable_overlap=False)
    builder.device = "cuda:0"
    overrides = build_generation_batch_overrides(
        server_args_overrides=server_args_overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(overrides)
    return builder, overrides


def test_cuda_engine_caps_the_kv_pool_at_the_running_requests() -> None:
    builder, overrides = cuda_overrides()

    assert overrides["max_total_tokens"] == 32 * builder.context_length


def test_kv_pool_cap_follows_an_operator_max_running_requests() -> None:
    builder, overrides = cuda_overrides(max_running_requests=64)

    assert overrides["max_running_requests"] == 64
    assert overrides["max_total_tokens"] == 64 * builder.context_length


def test_operator_max_total_tokens_is_kept() -> None:
    _, overrides = cuda_overrides(max_total_tokens=5000)

    assert overrides["max_total_tokens"] == 5000


@pytest.mark.parametrize("enable_overlap", [False, True])
def test_overlap_conflict_fails_before_draft_materialization(
    enable_overlap: bool,
) -> None:
    """A contradictory native flag must fail before resolving a draft checkpoint."""
    builder = FunCosyVoice3EngineBuilder(enable_overlap=enable_overlap)
    overrides = build_generation_batch_overrides(
        server_args_overrides={
            "speculative_algorithm": "DSPARK",
            "speculative_draft_model_path": "unavailable-draft",
            "disable_overlap_schedule": enable_overlap,
        },
        **builder.generation_defaults(dtype="bfloat16"),
    )

    with pytest.raises(ValueError, match="conflicts with factory.enable_overlap"):
        builder.adjust_overrides(overrides)


def test_without_draft_preserves_native_scheduling_override() -> None:
    """The speculative option must not reject an existing non-draft configuration."""
    _, overrides = cuda_overrides(disable_overlap_schedule=False)

    assert overrides["disable_overlap_schedule"] is False
