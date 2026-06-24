# SPDX-License-Identifier: Apache-2.0
"""AudioVAE config for Ming-Omni-TTS.

Adapted from the official Ming-omni-tts ``audio_tokenizer`` package.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class AudioVAEconfig(PretrainedConfig):
    """Config class matching the official AudioVAE checkpoint metadata."""

    def __init__(
        self,
        sample_rate: int = 16000,
        enc_kwargs: dict[str, Any] | None = None,
        dec_kwargs: dict[str, Any] | None = None,
        hifi_gan_disc_kwargs: dict[str, Any] | None = None,
        spec_disc_kwargs: dict[str, Any] | None = None,
        lambda_disc: float = 1.0,
        lambda_mel_loss: float = 15.0,
        lambda_adv: float = 1.0,
        lambda_feat_match_loss: float = 1.0,
        init_method: str = "normal",
        patch_size: int = -1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("semantic_module_kwargs", None)
        kwargs.pop("lambda_semantic", None)
        self.sample_rate = sample_rate
        self.enc_kwargs = enc_kwargs
        self.dec_kwargs = dec_kwargs
        self.hifi_gan_disc_kwargs = hifi_gan_disc_kwargs
        self.spec_disc_kwargs = spec_disc_kwargs
        self.lambda_disc = lambda_disc
        self.lambda_mel_loss = lambda_mel_loss
        self.lambda_adv = lambda_adv
        self.lambda_feat_match_loss = lambda_feat_match_loss
        self.init_method = init_method
        self.patch_size = patch_size
        super().__init__(**kwargs)


__all__ = ["AudioVAEconfig"]
