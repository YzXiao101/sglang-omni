# SPDX-License-Identifier: Apache-2.0
"""Cache-key contract for the Ming-Omni-TTS reference encoder."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_tts import reference_encode
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.models.ming_tts.reference_encode import (
    MingTTSReferenceEncoder,
    _MingTTSReferenceEncodeHook,
    _prompt_conditioning_digest,
)
from sglang_omni.models.ming_tts.tokenizer import (
    AUDIO_PATCH_TOKEN,
    AUDIO_START_TOKEN,
    SPK_END_TOKEN,
    SPK_START_TOKEN,
    MingTTSSpecialTokenIds,
    MingTTSTokenizerBundle,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class _StubEncoder:
    """Only the attributes the hook consults; no AudioVAE/CampPlus weights."""

    sample_rate = 44100
    patch_size = 4
    dtype = torch.bfloat16

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def _encode_reference(self, ref_audio: str) -> dict:
        self.encode_calls.append(ref_audio)
        return {
            "prompt_latent_token_count": 1,
            "content": Path(ref_audio).read_bytes(),
        }


class _CountingAudioVAE:
    def __init__(self) -> None:
        self.calls = 0

    def encode_latent(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del waveform, waveform_length
        self.calls += 1
        latent = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        return latent, torch.tensor([4], dtype=torch.long)


class _CountingSpeakerEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        del waveform
        self.calls += 1
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in ("<role>HUMAN</role>", "<role>ASSISTANT</role>"):
            return [1, 2]
        special_tokens = {
            AUDIO_PATCH_TOKEN: 3,
            AUDIO_START_TOKEN: 4,
            SPK_START_TOKEN: 6,
            f"{SPK_END_TOKEN}\n": 7,
        }
        if text in special_tokens:
            return [special_tokens[text]]
        return [10] if text else []


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


def _write_wav_like(path: Path, middle: bytes) -> None:
    """Same-size payloads that differ only in the middle bytes."""
    assert len(middle) == 4
    path.write_bytes(b"RIFF" + b"\x00" * 9000 + middle + b"\x00" * 9000 + b"data")


def _hook_and_service(tmp_path) -> tuple[_StubEncoder, ReferenceEncodeService]:
    encoder = _StubEncoder()
    hook = _MingTTSReferenceEncodeHook(encoder, model_identity=str(tmp_path))
    return encoder, ReferenceEncodeService(hook, max_items=16, max_bytes=1 << 20)


def test_same_size_references_do_not_share_a_cache_entry(tmp_path) -> None:
    """Two same-size files differing only in the middle must key separately;
    a sampled head/tail hash would collide here and serve the wrong speaker."""
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav_like(a, b"AAAA")
    _write_wav_like(b, b"BBBB")
    assert a.stat().st_size == b.stat().st_size

    encoder, service = _hook_and_service(tmp_path)
    artifact_a = service.get_or_encode(str(a))
    artifact_b = service.get_or_encode(str(b))

    assert artifact_a["content"] != artifact_b["content"]
    assert encoder.encode_calls == [str(a), str(b)]
    stats = service.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0


def test_prompt_conditioning_digest_is_stable_and_value_sensitive() -> None:
    speaker = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    latent = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32)
    changed = latent.clone()
    changed[0, 0, 0] = 7.0

    digest = _prompt_conditioning_digest(speaker, latent)

    assert _prompt_conditioning_digest(speaker.clone(), latent.clone()) == digest
    assert _prompt_conditioning_digest(speaker, changed) != digest


@pytest.mark.parametrize("compute_prompt_cache_digest", [False, True])
def test_reference_cache_preserves_prompt_digest_policy(
    tmp_path,
    monkeypatch,
    compute_prompt_cache_digest: bool,
) -> None:
    ref = tmp_path / "ref.wav"
    _write_wav_like(ref, b"AAAA")
    audio_vae = _CountingAudioVAE()
    speaker_encoder = _CountingSpeakerEncoder()
    decoder = SimpleNamespace(
        audio_vae=audio_vae,
        sample_rate=44100,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    encoder = MingTTSReferenceEncoder(
        decoder,
        speaker_encoder,
        patch_size=2,
        cache_model_identity=str(tmp_path),
        compute_prompt_cache_digest=compute_prompt_cache_digest,
    )
    waveform = torch.zeros((1, 7056), dtype=torch.float32)
    monkeypatch.setattr(
        encoder,
        "_load_reference_waveform",
        lambda _path: (waveform, waveform),
    )
    digest_calls = 0
    real_digest = reference_encode._prompt_conditioning_digest

    def track_digest(
        speaker: torch.Tensor,
        prompt_latent: torch.Tensor,
    ) -> str:
        nonlocal digest_calls
        digest_calls += 1
        return real_digest(speaker, prompt_latent)

    monkeypatch.setattr(reference_encode, "_prompt_conditioning_digest", track_digest)

    def payload(request_id: str) -> StagePayload:
        state = MingTTSState(
            text="hello",
            ref_audio=str(ref),
            ref_text="reference",
            max_decode_steps=2,
        )
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello"),
            data=state.to_dict(),
        )

    first = encoder.encode_payload(
        payload("reference-1"),
        tokenizer=_tokenizer(),
        context_length=64,
    )
    second = encoder.encode_payload(
        payload("reference-2"),
        tokenizer=_tokenizer(),
        context_length=64,
    )
    first_state = MingTTSState.from_dict(first.data)
    second_state = MingTTSState.from_dict(second.data)

    assert audio_vae.calls == 1
    assert speaker_encoder.calls == 1
    assert digest_calls == int(compute_prompt_cache_digest)
    if compute_prompt_cache_digest:
        assert first_state.prompt_conditioning_digest is not None
        assert (
            second_state.prompt_conditioning_digest
            == first_state.prompt_conditioning_digest
        )
    else:
        assert first_state.prompt_conditioning_digest is None
        assert second_state.prompt_conditioning_digest is None


def test_rewritten_reference_is_not_served_stale(tmp_path) -> None:
    ref = tmp_path / "ref.wav"
    _write_wav_like(ref, b"AAAA")

    encoder, service = _hook_and_service(tmp_path)
    first = service.get_or_encode(str(ref))

    _write_wav_like(ref, b"BBBB")
    os.utime(ref, (1_700_000_000, 1_700_000_000))
    second = service.get_or_encode(str(ref))

    assert first["content"] != second["content"]
    assert len(encoder.encode_calls) == 2
