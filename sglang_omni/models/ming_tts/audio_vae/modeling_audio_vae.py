# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AudioVAE model wrapper."""

from __future__ import annotations

from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
    AudioVAE as TalkerAudioVAE,
)
from sglang_omni.models.ming_tts.audio_vae.configuration_audio_vae import AudioVAEconfig

_AUDIO_VAE_QWEN2_ATTN_IMPL = "sdpa"


def _set_qwen2_attention(config: AudioVAEconfig) -> None:
    for kwargs_name in ("enc_kwargs", "dec_kwargs"):
        kwargs = getattr(config, kwargs_name, None)
        if not isinstance(kwargs, dict):
            continue
        backbone = kwargs.get("backbone")
        if isinstance(backbone, dict):
            backbone["_attn_implementation"] = _AUDIO_VAE_QWEN2_ATTN_IMPL


def _sync_child_attention_configs(model: TalkerAudioVAE) -> None:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and module.__class__.__name__ == "Qwen2Model":
            config._attn_implementation = _AUDIO_VAE_QWEN2_ATTN_IMPL


class AudioVAE(TalkerAudioVAE):
    config_class = AudioVAEconfig

    def __init__(self, config: AudioVAEconfig) -> None:
        _set_qwen2_attention(config)
        super().__init__(config)
        _sync_child_attention_configs(self)


__all__ = ["AudioVAE"]
