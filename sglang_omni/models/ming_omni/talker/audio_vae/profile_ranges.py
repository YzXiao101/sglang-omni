# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

import torch

_AUDIO_VAE_NVTX_ENABLED = os.environ.get("SGLANG_OMNI_AUDIO_VAE_PROFILE_NVTX") == "1"


def audio_vae_nvtx_range(name: str) -> AbstractContextManager[None]:
    if not _AUDIO_VAE_NVTX_ENABLED:
        return nullcontext()
    return torch.cuda.nvtx.range(name)
