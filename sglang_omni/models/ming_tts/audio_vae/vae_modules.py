# SPDX-License-Identifier: Apache-2.0
"""AudioVAE module wrappers for Ming-Omni-TTS."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
    Decoder as TalkerDecoder,
)
from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
    Encoder as TalkerEncoder,
)
from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
    StreamingLinearUpsample,
)


class Encoder(TalkerEncoder):
    def __init__(
        self,
        encoder_args: dict[str, Any],
        input_dim: int = 320,
        hop_size: int = 320,
        latent_dim: int = 64,
        patch_size: int = -1,
    ) -> None:
        super().__init__(
            encoder_args=encoder_args,
            input_dim=input_dim,
            hop_size=hop_size,
            latent_dim=latent_dim,
            patch_size=patch_size,
        )


class Decoder(TalkerDecoder):
    def __init__(
        self,
        decoder_args: dict[str, Any],
        output_dim: int = 320,
        latent_dim: int = 64,
        patch_size: int = -1,
    ) -> None:
        super().__init__(
            decoder_args=decoder_args,
            output_dim=output_dim,
            latent_dim=latent_dim,
            patch_size=patch_size,
        )


__all__ = ["Decoder", "Encoder", "StreamingLinearUpsample"]
