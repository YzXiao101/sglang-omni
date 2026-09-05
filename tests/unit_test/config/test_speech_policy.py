# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.config import UPLOADED_VOICE_MODE_OPTIONAL, TTSRequestPolicy


def test_uploaded_voice_policy_rejects_builtin_model_restrictions() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        TTSRequestPolicy(
            uploaded_voice_mode=UPLOADED_VOICE_MODE_OPTIONAL,
            allowed_voices=("narrator",),
        )
