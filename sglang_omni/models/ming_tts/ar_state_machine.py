# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AR recurrence state machine."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from sglang_omni.models.ming_tts.ar_state import (
    get_ming_ar_state,
    sync_ming_ar_state_to_legacy,
    update_ming_ar_latent_history_,
)
from sglang_omni.models.ming_tts.profile_events import ming_profile_event


def normalize_ming_ar_hidden_states(
    hidden: Any,
    *,
    request_count: int,
) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError("Ming TTS model output did not include hidden states")
    if hidden.ndim == 2:
        z_diff = hidden.unsqueeze(1)
    elif hidden.ndim == 3 and int(hidden.shape[1]) == 1:
        z_diff = hidden
    else:
        raise RuntimeError(
            "Ming TTS hidden states must have shape [batch, hidden] or "
            f"[batch, 1, hidden], got {tuple(hidden.shape)}"
        )
    if int(z_diff.shape[0]) != int(request_count):
        raise RuntimeError(
            "Ming TTS hidden batch does not match request batch: "
            f"{int(z_diff.shape[0])} != {int(request_count)}"
        )
    return z_diff


class MingARStateMachine:
    """Eager Ming recurrence after the SGLang backbone hidden-state forward."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def step_batch(self, hidden: torch.Tensor, requests: list[Any]) -> torch.Tensor:
        if not requests:
            return torch.empty(0, dtype=torch.long, device=hidden.device)

        z_diff = normalize_ming_ar_hidden_states(
            hidden,
            request_count=len(requests),
        )
        device = z_diff.device
        next_ids = []

        if device.type == "cuda":
            dtype = self.model._decode_input_embedding.weight.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                dtype = torch.bfloat16
            context = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            context = nullcontext()

        with context:
            for row_idx, sched_req in enumerate(requests):
                data = sched_req.data
                ar_state = get_ming_ar_state(data)
                step = int(ar_state.generation_steps)
                history = ar_state.ensure_latent_history(
                    device=device,
                    history_patch_size=int(self.model.history_patch_size),
                    latent_dim=int(self.model.latent_dim),
                )
                row_hidden = z_diff[row_idx : row_idx + 1]
                row_metadata = {
                    "batch_size": len(requests),
                    "row_idx": int(row_idx),
                    "generation_step": int(step),
                }

                with ming_profile_event(
                    sched_req.request_id,
                    "ming_flowloss_sample",
                    row_metadata,
                ):
                    sampled, _trajectory = self.model.flowloss.sample(
                        row_hidden,
                        history,
                        cfg=float(ar_state.cfg),
                        patch_size=int(self.model.patch_size),
                        sigma=float(ar_state.sigma),
                        temperature=float(ar_state.flow_temperature),
                    )
                if not isinstance(sampled, torch.Tensor):
                    sampled = torch.as_tensor(sampled)
                if sampled.ndim == 2:
                    sampled = sampled.unsqueeze(0)
                expected = (
                    1,
                    int(self.model.patch_size),
                    int(self.model.latent_dim),
                )
                if tuple(sampled.shape) != expected:
                    raise RuntimeError(
                        f"Ming TTS sampled latent must have shape {expected}, "
                        f"got {tuple(sampled.shape)}"
                    )
                ar_state.generated_latents.append(sampled.squeeze(0).detach())

                with ming_profile_event(
                    sched_req.request_id,
                    "ming_stop_head",
                    row_metadata,
                ):
                    stop_prob = self.model.stop_head(row_hidden).softmax(dim=-1)[
                        0,
                        0,
                        1,
                    ]
                    stop = bool(stop_prob.item() > 0.5 and step > 3)
                ar_state.generated_last_chunk.append(stop)
                if stop:
                    ar_state.stop_step = step
                    next_ids.append(int(ar_state.audio_eos_token_id))
                    sync_ming_ar_state_to_legacy(data, ar_state)
                    continue

                if ar_state.latent_history is None:
                    raise RuntimeError("Ming TTS latent_history must be initialized")
                update_ming_ar_latent_history_(ar_state.latent_history, sampled)

                next_ids.append(int(ar_state.audio_patch_token_id))
                will_finish_by_length = step + 1 >= int(ar_state.max_decode_steps)
                if not will_finish_by_length:
                    with ming_profile_event(
                        sched_req.request_id,
                        "ming_feedback_proj",
                        row_metadata,
                    ):
                        feedback = self.model.linear_proj_audio(sampled)
                    ar_state.pending_feedback_queue.append(
                        feedback.reshape(-1).detach()
                    )
                sync_ming_ar_state_to_legacy(data, ar_state)

        return torch.tensor(next_ids, dtype=torch.long, device=device)


__all__ = [
    "MingARStateMachine",
    "normalize_ming_ar_hidden_states",
]
