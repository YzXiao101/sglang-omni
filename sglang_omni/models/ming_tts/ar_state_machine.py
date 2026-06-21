# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AR recurrence state machine."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.ming_tts.ar_state import (
    get_ming_ar_state,
    sync_ming_ar_state_to_legacy,
    update_ming_ar_latent_history_,
)
from sglang_omni.models.ming_tts.profile_events import ming_profile_event


@dataclass
class MingARStepResult:
    """Rank-synchronized output of one Ming AR recurrence step."""

    next_token_ids: torch.Tensor
    feedback_embeddings: torch.Tensor
    feedback_mask: torch.Tensor
    stop_flags: torch.Tensor
    length_finish_flags: torch.Tensor
    generation_steps: torch.Tensor
    request_ids: list[str]
    generated_latents: list[torch.Tensor | None] = field(default_factory=list)

    @classmethod
    def empty_for_broadcast(
        cls,
        *,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        feedback_dtype: torch.dtype,
        request_ids: list[str],
    ) -> "MingARStepResult":
        return cls(
            next_token_ids=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            feedback_embeddings=torch.zeros(
                int(batch_size),
                int(hidden_size),
                dtype=feedback_dtype,
                device=device,
            ),
            feedback_mask=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            stop_flags=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            length_finish_flags=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            generation_steps=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            request_ids=list(request_ids),
            generated_latents=[None for _ in range(int(batch_size))],
        )


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

    def step_batch(self, hidden: torch.Tensor, requests: list[Any]) -> MingARStepResult:
        weight = self.model._decode_input_embedding.weight
        if not requests:
            return MingARStepResult.empty_for_broadcast(
                batch_size=0,
                hidden_size=int(weight.shape[1]),
                device=hidden.device,
                feedback_dtype=weight.dtype,
                request_ids=[],
            )

        z_diff = normalize_ming_ar_hidden_states(
            hidden,
            request_count=len(requests),
        )
        device = z_diff.device
        batch_size = len(requests)
        hidden_size = int(weight.shape[1])
        result = MingARStepResult.empty_for_broadcast(
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            feedback_dtype=weight.dtype,
            request_ids=[str(sched_req.request_id) for sched_req in requests],
        )
        next_ids = []

        if device.type == "cuda":
            dtype = weight.dtype
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
                result.generation_steps[row_idx] = step
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
                sampled_chunk = sampled.squeeze(0).detach()
                ar_state.generated_latents.append(sampled_chunk)
                result.generated_latents[row_idx] = sampled_chunk

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
                result.stop_flags[row_idx] = stop
                if stop:
                    ar_state.stop_step = step
                    next_ids.append(int(ar_state.audio_eos_token_id))
                    ar_state.generation_steps = step + 1
                    sync_ming_ar_state_to_legacy(data, ar_state)
                    continue

                if ar_state.latent_history is None:
                    raise RuntimeError("Ming TTS latent_history must be initialized")
                update_ming_ar_latent_history_(ar_state.latent_history, sampled)

                next_ids.append(int(ar_state.audio_patch_token_id))
                will_finish_by_length = step + 1 >= int(ar_state.max_decode_steps)
                result.length_finish_flags[row_idx] = will_finish_by_length
                if not will_finish_by_length:
                    with ming_profile_event(
                        sched_req.request_id,
                        "ming_feedback_proj",
                        row_metadata,
                    ):
                        feedback = self.model.linear_proj_audio(sampled)
                    feedback = feedback.reshape(-1).detach()
                    if int(feedback.shape[0]) != hidden_size:
                        raise RuntimeError(
                            "Ming TTS feedback projection hidden size mismatch: "
                            f"{int(feedback.shape[0])} != {hidden_size}"
                        )
                    result.feedback_embeddings[row_idx].copy_(
                        feedback.to(
                            device=result.feedback_embeddings.device,
                            dtype=result.feedback_embeddings.dtype,
                        )
                    )
                    result.feedback_mask[row_idx] = True
                    ar_state.pending_feedback_queue.append(feedback)
                ar_state.generation_steps = step + 1
                sync_ming_ar_state_to_legacy(data, ar_state)

        result.next_token_ids.copy_(
            torch.tensor(next_ids, dtype=torch.long, device=device)
        )
        return result


def apply_follower_ming_ar_step_result(
    step_result: MingARStepResult,
    requests: list[Any],
) -> None:
    """Apply rank0-owned step metadata needed by TP follower ranks."""

    if int(step_result.next_token_ids.numel()) != len(requests):
        raise RuntimeError(
            "Ming TTS follower step result batch does not match requests: "
            f"{int(step_result.next_token_ids.numel())} != {len(requests)}"
        )

    for row_idx, sched_req in enumerate(requests):
        ar_state = get_ming_ar_state(sched_req.data)
        step = int(step_result.generation_steps[row_idx].item())
        ar_state.generation_steps = step + 1
        if bool(step_result.stop_flags[row_idx].item()):
            ar_state.stop_step = step
        if bool(step_result.feedback_mask[row_idx].item()):
            ar_state.pending_feedback_queue.append(
                step_result.feedback_embeddings[row_idx].detach().clone()
            )
        sync_ming_ar_state_to_legacy(sched_req.data, ar_state)


__all__ = [
    "MingARStepResult",
    "MingARStateMachine",
    "apply_follower_ming_ar_step_result",
    "normalize_ming_ar_hidden_states",
]
