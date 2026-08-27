# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest


def test_ming_tts_audio_decode_factory_exposes_only_supported_contract() -> None:
    from sglang_omni.models.ming_tts.config import (
        MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
        MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
        MING_TTS_DEFAULT_STREAM_SLOTS,
    )
    from sglang_omni.models.ming_tts.stages import create_audio_decode_executor

    parameters = inspect.signature(create_audio_decode_executor).parameters

    assert "decode_mode" not in parameters
    assert (
        parameters["initial_chunk_patches"].default
        == MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    )
    assert (
        parameters["steady_chunk_patches"].default
        == MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    )
    assert parameters["stream_slots"].default == MING_TTS_DEFAULT_STREAM_SLOTS
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
        ({"stream_slots": 0}, "stream_slots must be a positive integer"),
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


def test_ming_tts_audio_decode_factory_uses_stream_slots_for_decoder_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from sglang_omni.models.ming_tts import audio_decode, stages, streaming_vocoder

    decoder_calls: list[dict[str, Any]] = []

    class FakeDecoder:
        sample_rate = 44100

        def __init__(self, audio_vae: object, **kwargs: Any) -> None:
            del audio_vae
            decoder_calls.append(kwargs)

        def close(self) -> None:
            return None

    class FakeScheduler:
        def __init__(self, decoder: object, **kwargs: Any) -> None:
            del decoder, kwargs

        def warmup_now(self) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda _: "checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_ming_tts_config",
        lambda _: SimpleNamespace(
            audio_tokenizer_config=object(),
            audio_patch_size=2,
            latent_dim=4,
        ),
    )
    monkeypatch.setattr(
        stages,
        "resolve_ming_tts_audio_vae_config",
        lambda *_args, **_kwargs: SimpleNamespace(dec_kwargs={"latent_dim": 4}),
    )
    monkeypatch.setattr(stages, "_load_ming_tts_audio_vae", lambda *_a, **_k: object())
    monkeypatch.setattr(
        stages,
        "get_gpu_device_info",
        lambda _: SimpleNamespace(total_memory_bytes=1),
    )
    monkeypatch.setattr(stages, "get_process_gpu_memory_bytes", lambda _: 0)
    monkeypatch.setattr(audio_decode, "MingAudioDecoder", FakeDecoder)
    monkeypatch.setattr(
        streaming_vocoder,
        "MingTTSStreamingVocoderScheduler",
        FakeScheduler,
    )

    stages.create_audio_decode_executor("unused", stream_slots=3)

    assert decoder_calls == [
        {
            "stream_capacity": 3,
            "max_stream_step_latents": 8,
            "streaming_cuda_graph_required": False,
        }
    ]


@pytest.mark.parametrize("field", ["initial_chunk_patches", "steady_chunk_patches"])
@pytest.mark.parametrize("value", [True, 1.5, "2", 0, -1])
def test_audio_decode_factory_rejects_invalid_cadence_before_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    from sglang_omni.models.ming_tts import stages

    def fail_if_called(model_path: str) -> str:
        raise AssertionError(f"unexpected checkpoint resolution for {model_path}")

    monkeypatch.setattr(stages, "_resolve_checkpoint", fail_if_called)

    with pytest.raises(
        ValueError,
        match=f"{field} must be a positive integer",
    ):
        stages.create_audio_decode_executor(
            "unused",
            **{field: value},
        )
