# SPDX-License-Identifier: Apache-2.0
"""Prompt-wav path resolution for Ming talker voice manifests."""

from __future__ import annotations

import os

import sglang_omni


def _fallback_roots() -> list[str]:
    # Checkpoint manifests may reference assets from the authoring checkout
    # (Ming-flash-omni-2.0's DB30 points at tests/data); the local checkout
    # root is the only stand-in for that.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(sglang_omni.__file__)))
    return [pkg_root]


def resolve_prompt_wav_path(raw_path: str, talker_dir: str) -> str | None:
    """Return an existing absolute path for a manifest prompt-wav entry.

    Tries the checkpoint-relative join first, then re-resolves path suffixes
    (longest first) against the talker dir and the checkout root, so absolute
    paths baked in on the authoring machine keep working. Returns None when
    no candidate exists.
    """
    primary = os.path.join(talker_dir, raw_path)
    if os.path.isfile(primary):
        return primary
    parts = [p for p in raw_path.replace("\\", "/").split("/") if p]
    roots = [talker_dir, *_fallback_roots()]
    for start in range(len(parts)):
        tail = os.path.join(*parts[start:])
        for root in roots:
            candidate = os.path.join(root, tail)
            if os.path.isfile(candidate):
                return candidate
    return None
