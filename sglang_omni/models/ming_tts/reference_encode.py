# SPDX-License-Identifier: Apache-2.0
"""Reference audio encoding for Ming-Omni-TTS zero-shot TTS."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from sglang_omni.models.ming_tts.audio_decode import MingAudioDecoder
from sglang_omni.models.ming_tts.payload_types import (
    MING_TTS_SAMPLE_RATE,
    MingTTSState,
    encode_prompt_latent,
    encode_speaker_embedding,
)
from sglang_omni.models.ming_tts.profile_events import (
    ming_profile_event,
    tensor_metadata,
)
from sglang_omni.models.ming_tts.prompt_builder import build_ming_tts_prompt
from sglang_omni.models.ming_tts.tokenizer import MingTTSTokenizerBundle
from sglang_omni.proto import StagePayload


class MingSpeakerEmbeddingExtractor:
    """CampPlus speaker embedding extractor matching the official reference path."""

    def __init__(self, campplus_model: str, *, target_sr: int = 16000) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "Ming-Omni-TTS reference audio requires onnxruntime for "
                "campplus.onnx speaker embedding extraction."
            ) from exc

        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(
            campplus_model,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.target_sr = int(target_sr)

    def __call__(self, waveform: Any) -> Any:
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
        except ImportError as exc:
            raise RuntimeError(
                "Ming-Omni-TTS reference audio requires torch and torchaudio "
                "kaldi compliance features for CampPlus speaker embeddings."
            ) from exc

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim != 2 or int(waveform.shape[0]) != 1:
            raise ValueError(
                "Ming-Omni-TTS speaker reference waveform must have shape [1, T], "
                f"got {tuple(waveform.shape)}"
            )
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.target_sr,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        input_name = self.session.get_inputs()[0].name
        embedding = self.session.run(None, {input_name: feat.unsqueeze(0).numpy()})[0]
        return torch.as_tensor(embedding.reshape(1, -1), dtype=torch.float32)


class MingTTSReferenceEncoder:
    """Encode a single reference audio into speaker embedding and prompt latent."""

    def __init__(
        self,
        decoder: MingAudioDecoder,
        speaker_encoder: MingSpeakerEmbeddingExtractor,
        *,
        patch_size: int,
    ) -> None:
        self.audio_vae = decoder.audio_vae
        self.sample_rate = int(decoder.sample_rate)
        self.device = decoder.device
        self.patch_size = int(patch_size)
        self.speaker_encoder = speaker_encoder
        if self.sample_rate != MING_TTS_SAMPLE_RATE:
            raise ValueError(
                "Ming-Omni-TTS reference encoder requires sample_rate "
                f"{MING_TTS_SAMPLE_RATE}, got {self.sample_rate}"
            )
        if self.patch_size <= 0:
            raise ValueError(
                f"Ming-Omni-TTS reference encoder patch_size must be > 0, got {patch_size}"
            )

    @classmethod
    def from_config(
        cls,
        audio_config: Any,
        *,
        checkpoint_dir: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        patch_size: int,
    ) -> "MingTTSReferenceEncoder":
        decoder = MingAudioDecoder.from_config(
            audio_config,
            device=device,
            dtype=dtype,
        )
        return cls(
            decoder,
            MingSpeakerEmbeddingExtractor(str(Path(checkpoint_dir) / "campplus.onnx")),
            patch_size=patch_size,
        )

    def encode_payload(
        self,
        payload: StagePayload,
        *,
        tokenizer: MingTTSTokenizerBundle,
        context_length: int,
    ) -> StagePayload:
        state = MingTTSState.from_dict(payload.data)
        if state.ref_audio is None:
            return payload
        if state.ref_text is None or not str(state.ref_text).strip():
            raise ValueError("Ming-Omni-TTS reference audio requires ref_text")
        if not isinstance(state.ref_audio, str):
            raise ValueError(
                "Ming-Omni-TTS reference audio must be a local audio path string"
            )

        with ming_profile_event(
            payload.request_id,
            "ming_reference_audio_load",
            metadata={"has_reference": True},
        ):
            prompt_waveform, speaker_waveform = self._load_reference_waveform(
                state.ref_audio
            )
        prompt_waveform = self._pad_waveform(prompt_waveform)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Ming-Omni-TTS reference audio requires torch") from exc

        with torch.inference_mode():
            waveform_length = torch.tensor(
                [int(prompt_waveform.shape[1])],
                dtype=torch.long,
                device=self.device,
            )
            prompt_waveform = self._prepare_audio_vae_waveform(prompt_waveform)
            with ming_profile_event(
                payload.request_id,
                "ming_audio_vae_encode",
                metadata={
                    "waveform": tensor_metadata(prompt_waveform),
                    "sample_rate": int(self.sample_rate),
                },
            ):
                prompt_latent, _prompt_latent_length = self.audio_vae.encode_latent(
                    prompt_waveform,
                    waveform_length,
                )
        if prompt_latent.ndim != 3 or int(prompt_latent.shape[0]) != 1:
            raise RuntimeError(
                "Ming-Omni-TTS prompt latent must have shape [1, frames, latent_dim], "
                f"got {tuple(prompt_latent.shape)}"
            )
        frames = int(prompt_latent.shape[1])
        if frames % self.patch_size != 0:
            raise RuntimeError(
                "Ming-Omni-TTS prompt latent frames must be divisible by patch_size: "
                f"frames={frames}, patch_size={self.patch_size}"
            )
        prompt_latent_token_count = frames // self.patch_size
        with ming_profile_event(
            payload.request_id,
            "ming_campplus",
            metadata={
                "waveform": tensor_metadata(speaker_waveform),
                "sample_rate": int(self.speaker_encoder.target_sr),
            },
        ):
            speaker_embedding = self.speaker_encoder(speaker_waveform)

        for field_name, value in encode_speaker_embedding(speaker_embedding).items():
            setattr(state, field_name, value)
        for field_name, value in encode_prompt_latent(prompt_latent).items():
            setattr(state, field_name, value)
        state.prompt_latent_token_count = int(prompt_latent_token_count)
        state.prompt_text = str(state.ref_text)
        state.speaker_fingerprint = self._speaker_fingerprint(
            state.ref_audio,
            state.ref_text,
        )

        plan = build_ming_tts_prompt(
            state,
            tokenizer,
            prompt_text=state.ref_text,
            speaker_count=1,
            prompt_latent_token_count=state.prompt_latent_token_count,
        )
        if plan.prompt_tokens + state.max_decode_steps > int(context_length):
            raise ValueError(
                "Ming-Omni-TTS request exceeds context length after reference encode: "
                f"prompt_tokens={plan.prompt_tokens}, "
                f"max_decode_steps={state.max_decode_steps}, "
                f"context_length={context_length}"
            )

        state.prompt = plan.effective_prompt
        state.input_ids = plan.input_ids
        state.prompt_tokens = plan.prompt_tokens
        state.spk_token_positions = plan.spk_token_positions
        state.spk_injection_positions = plan.spk_injection_positions
        state.audio_token_position = plan.audio_token_position
        state.prompt_latent_start_position = plan.prompt_latent_start_position
        state.prompt_latent_token_count = plan.prompt_latent_token_count

        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    def _load_reference_waveform(self, path: str) -> tuple[Any, Any]:
        try:
            import torchaudio
            import torchaudio.functional as F
        except ImportError as exc:
            raise RuntimeError(
                "Ming-Omni-TTS reference audio requires torchaudio to load and "
                "resample prompt wav files."
            ) from exc

        waveform, sample_rate = torchaudio.load(path)
        if waveform.ndim != 2 or int(waveform.shape[0]) != 1:
            raise ValueError(
                "Ming-Omni-TTS currently supports only mono reference audio, "
                f"got shape {tuple(waveform.shape)}"
            )
        speaker_waveform = waveform.clone()
        if int(sample_rate) != self.sample_rate:
            waveform = F.resample(
                waveform,
                orig_freq=int(sample_rate),
                new_freq=self.sample_rate,
            )
        if int(sample_rate) != self.speaker_encoder.target_sr:
            speaker_waveform = F.resample(
                speaker_waveform,
                orig_freq=int(sample_rate),
                new_freq=self.speaker_encoder.target_sr,
            )
        return waveform, speaker_waveform

    def _pad_waveform(self, waveform: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Ming-Omni-TTS reference audio requires torch") from exc

        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (int(waveform.shape[-1]) + pad_align - 1) // pad_align * pad_align
        if new_len == int(waveform.shape[-1]):
            return waveform
        padded = torch.zeros(
            1,
            new_len,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        padded[:, : int(waveform.shape[-1])] = waveform.clone()
        return padded

    def _prepare_audio_vae_waveform(self, waveform: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Ming-Omni-TTS reference audio requires torch") from exc

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        # Official generate() runs AudioVAE encode under bf16 autocast; this
        # stage is isolated, so align the waveform to the loaded AudioVAE dtype.
        return waveform.to(
            device=self.device,
            dtype=self._audio_vae_floating_dtype(),
        )

    def _audio_vae_floating_dtype(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Ming-Omni-TTS reference audio requires torch") from exc

        for parameter in self.audio_vae.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    @staticmethod
    def _speaker_fingerprint(path: str, ref_text: str | None) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(ref_text or "").encode("utf-8"))
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        else:
            hasher.update(path.encode("utf-8"))
        return hasher.hexdigest()[:24]


__all__ = [
    "MingSpeakerEmbeddingExtractor",
    "MingTTSReferenceEncoder",
]
