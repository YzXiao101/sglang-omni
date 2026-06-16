# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AudioVAE model.

Adapted from the official Ming-omni-tts ``audio_tokenizer`` package.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from sglang_omni.models.ming_tts.audio_vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.models.ming_tts.audio_vae.vae_modules import Decoder, Encoder


class AudioVAE(PreTrainedModel):
    config_class = AudioVAEconfig

    def __init__(self, config: AudioVAEconfig):
        super().__init__(config)
        if getattr(config, "semantic_module_kwargs", None) is not None:
            raise NotImplementedError(
                "Ming TTS AudioVAE semantic modules are currently unsupported; "
                "the 16.8B audio_tokenizer_config sets semantic_module_kwargs=null."
            )

        self.encoder = Encoder(
            encoder_args=config.enc_kwargs["backbone"],
            input_dim=config.enc_kwargs["input_dim"],
            hop_size=config.enc_kwargs.get("hop_size", 320),
            latent_dim=config.enc_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )
        self.decoder = Decoder(
            decoder_args=config.dec_kwargs["backbone"],
            output_dim=config.dec_kwargs["output_dim"],
            latent_dim=config.dec_kwargs["latent_dim"],
            patch_size=config.patch_size,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            if self.config.init_method == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @torch.inference_mode()
    def encode_latent(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from diffusers.models.autoencoders.autoencoder_oobleck import (
                OobleckDiagonalGaussianDistribution,
            )
        except ImportError as exc:
            raise ImportError(
                "Ming TTS AudioVAE.encode_latent requires diffusers. "
                "Install project dependencies before enabling reference audio."
            ) from exc

        frame_num = torch.ceil(
            waveform_length / self.config.enc_kwargs["input_dim"]
        ).to(torch.int32)
        if self.config.patch_size != -1:
            frame_num = torch.ceil(frame_num / self.config.patch_size)
        hidden, _ = self.encoder(waveform)
        hidden = hidden.transpose(1, 2)

        posterior = OobleckDiagonalGaussianDistribution(hidden)
        latent = posterior.sample().transpose(1, 2)
        return latent, frame_num

    @torch.inference_mode()
    def encode_unified_emb_from_latent(
        self,
        latent: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any | None]:
        del latent, past_key_values, use_cache
        raise NotImplementedError(
            "Ming TTS AudioVAE semantic embedding is currently unsupported."
        )

    @torch.inference_mode()
    def encode_unified_emb_from_waveform(
        self,
        waveform: torch.Tensor,
        waveform_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, frame_num = self.encode_latent(waveform, waveform_length)
        unified_emb, _ = self.encode_unified_emb_from_latent(latent)
        return unified_emb, latent, frame_num

    @torch.inference_mode()
    def decode(
        self,
        latent: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        stream_state: tuple[Any | None, Any | None, Any | None] = (None, None, None),
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, tuple[Any | None, Any | None, Any | None], Any | None]:
        waveform, stream_state, past_key_values = self.decoder.low_level_reconstruct(
            latent,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=stream_state,
            last_chunk=last_chunk,
        )
        return waveform, stream_state, past_key_values


__all__ = ["AudioVAE"]
