# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

import torch

_PROFILE_NVTX_ENABLED = os.environ.get("SGLANG_OMNI_MING_TTS_PROFILE_NVTX") == "1"


def profile_nvtx_enabled() -> bool:
    return _PROFILE_NVTX_ENABLED


def profile_nvtx_range(name: str) -> AbstractContextManager[None]:
    if not _PROFILE_NVTX_ENABLED:
        return nullcontext()
    return torch.cuda.nvtx.range(name)
