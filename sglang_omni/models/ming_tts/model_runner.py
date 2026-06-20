# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.ming_tts.profile_events import (
    emit_ming_event,
    ming_profile_event,
    tensor_metadata,
)


class MingTTSModelRunner(ModelRunner):
    """Runs Ming-Omni-TTS AR steps and samples continuous acoustic latents."""

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        for sched_req in requests:
            self._ensure_latent_history(
                sched_req.data,
                device=self.model._decode_input_embedding.weight.device,
            )

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        projected = [
            getattr(sched_req.data, "prefill_input_embeds", None)
            for sched_req in requests
        ]
        if all(item is None for item in projected):
            return None
        if any(item is None for item in projected):
            raise RuntimeError(
                "Ming TTS cannot mix projected and token-embedding prefill rows"
            )

        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            prompt_embeds = data.prefill_input_embeds
            if prompt_embeds is None:
                raise RuntimeError("Ming TTS prefill requires prefill_input_embeds")
            current = prompt_embeds[prefix_len : prefix_len + req_len]
            if int(current.shape[0]) != req_len:
                raise RuntimeError(
                    "Ming TTS projected prefill row mismatch: "
                    f"have {int(current.shape[0])}, need {req_len}"
                )
            pieces.append(current)
        input_embeds = torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=self.model._decode_input_embedding.weight.dtype,
        )

        model_runner = self.tp_worker.model_runner
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        metadata = {
            "batch_size": len(requests),
            "input_embeds": tensor_metadata(input_embeds),
        }
        for sched_req in requests:
            emit_ming_event(
                sched_req.request_id,
                "ming_custom_prefill_start",
                metadata,
            )
        try:
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_embeds_are_projected=True,
            )
        finally:
            for sched_req in requests:
                emit_ming_event(
                    sched_req.request_id,
                    "ming_custom_prefill_end",
                    metadata,
                )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch
        if is_lookahead:
            raise RuntimeError("Ming TTS async lookahead is currently unsupported")
        batch_size = len(requests)
        if batch_size == 0:
            return

        embedding = self.model._decode_input_embedding
        weight = embedding.weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "Ming TTS decode input_ids must contain one row id per request"
            )
        if batch_size > int(weight.shape[0]):
            raise RuntimeError(
                "Ming TTS decode batch exceeds staged decode-embedding rows "
                f"({batch_size} > {int(weight.shape[0])})"
            )

        rows = []
        for sched_req in requests:
            data = sched_req.data
            queue = data.pending_feedback_queue
            if not queue:
                raise RuntimeError(
                    f"Ming TTS request {sched_req.request_id} is missing "
                    "decode feedback embedding"
                )
            feedback = queue.popleft() if hasattr(queue, "popleft") else queue.pop(0)
            if not isinstance(feedback, torch.Tensor):
                feedback = torch.as_tensor(feedback)
            if feedback.ndim == 2 and int(feedback.shape[0]) == 1:
                feedback = feedback.reshape(-1)
            if feedback.ndim != 1 or int(feedback.shape[0]) != int(weight.shape[1]):
                raise RuntimeError(
                    "Ming TTS decode feedback must have shape [hidden] or "
                    f"[1, hidden], got {tuple(feedback.shape)}"
                )
            rows.append(feedback.to(device=weight.device, dtype=weight.dtype))

        stacked = torch.stack(rows, dim=0).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            weight[:batch_size].copy_(stacked)

        row_ids = torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
        forward_batch.input_ids[:batch_size].copy_(row_ids)

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._run_tts_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._run_tts_step(result, forward_batch, schedule_batch, requests)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        batch = getattr(scheduler_output, "batch_data", None)
        if bool(getattr(batch, "is_prefill_only", False)):
            return {sched_req.request_id for sched_req in scheduler_output.requests}
        return set()

    def _run_tts_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return

        hidden = getattr(result.logits_output, "hidden_states", None)
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
        if int(z_diff.shape[0]) != len(requests):
            raise RuntimeError(
                "Ming TTS hidden batch does not match request batch: "
                f"{int(z_diff.shape[0])} != {len(requests)}"
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

        batch_metadata = {
            "batch_size": len(requests),
            "hidden": tensor_metadata(z_diff),
        }
        for sched_req in requests:
            emit_ming_event(
                sched_req.request_id,
                "ming_tts_step_start",
                batch_metadata,
            )
        try:
            with context:
                for row_idx, sched_req in enumerate(requests):
                    data = sched_req.data
                    step = int(data.generation_steps)
                    history = self._ensure_latent_history(data, device=device)
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
                            cfg=float(data.cfg),
                            patch_size=int(self.model.patch_size),
                            sigma=float(data.sigma),
                            temperature=float(data.flow_temperature),
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
                    data.generated_latents.append(sampled.squeeze(0).detach())

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
                    data.generated_last_chunk.append(stop)
                    if stop:
                        data.stop_step = step
                        next_ids.append(int(data.audio_eos_token_id))
                        continue

                    if data.latent_history is None:
                        raise RuntimeError(
                            "Ming TTS latent_history must be initialized"
                        )
                    patch = int(sampled.shape[1])
                    history_len = int(data.latent_history.shape[1])
                    if patch >= history_len:
                        data.latent_history.copy_(
                            sampled[:, -history_len:, :].to(
                                device=data.latent_history.device,
                                dtype=data.latent_history.dtype,
                            )
                        )
                    else:
                        data.latent_history[:, :-patch, :].copy_(
                            data.latent_history[:, patch:, :].clone()
                        )
                        data.latent_history[:, -patch:, :].copy_(
                            sampled.to(
                                device=data.latent_history.device,
                                dtype=data.latent_history.dtype,
                            )
                        )

                    next_ids.append(int(data.audio_patch_token_id))
                    will_finish_by_length = step + 1 >= int(data.max_decode_steps)
                    if not will_finish_by_length:
                        with ming_profile_event(
                            sched_req.request_id,
                            "ming_feedback_proj",
                            row_metadata,
                        ):
                            feedback = self.model.linear_proj_audio(sampled)
                        data.pending_feedback_queue.append(
                            feedback.reshape(-1).detach()
                        )

            next_token_ids = torch.tensor(next_ids, dtype=torch.long, device=device)
            result.next_token_ids = next_token_ids
            result.can_run_cuda_graph = False
            schedule_batch.output_ids = next_token_ids
        finally:
            for sched_req in requests:
                emit_ming_event(
                    sched_req.request_id,
                    "ming_tts_step_end",
                    batch_metadata,
                )

    def _ensure_latent_history(
        self, data: Any, *, device: torch.device
    ) -> torch.Tensor:
        history = getattr(data, "latent_history", None)
        if history is None:
            history = torch.zeros(
                1,
                int(self.model.history_patch_size),
                int(self.model.latent_dim),
                device=device,
                dtype=torch.float32,
            )
            prompt_latent = getattr(data, "prompt_latent_for_history", None)
            if prompt_latent is not None:
                if not isinstance(prompt_latent, torch.Tensor):
                    prompt_latent = torch.as_tensor(prompt_latent)
                if prompt_latent.ndim == 2:
                    prompt_latent = prompt_latent.unsqueeze(0)
                expected_tail = (1, int(self.model.latent_dim))
                if (
                    prompt_latent.ndim != 3
                    or int(prompt_latent.shape[0]) != expected_tail[0]
                    or int(prompt_latent.shape[2]) != expected_tail[1]
                ):
                    raise RuntimeError(
                        "Ming TTS prompt latent history must have shape "
                        f"[1, frames, {expected_tail[1]}], "
                        f"got {tuple(prompt_latent.shape)}"
                    )
                prompt_latent = prompt_latent.to(device=device, dtype=torch.float32)
                history_len = int(history.shape[1])
                prompt_len = int(prompt_latent.shape[1])
                if prompt_len <= 0:
                    raise RuntimeError(
                        "Ming TTS prompt latent history requires at least one frame"
                    )
                if prompt_len >= history_len:
                    history.copy_(prompt_latent[:, -history_len:, :])
                else:
                    history[:, -prompt_len:, :].copy_(prompt_latent)
            data.latent_history = history
            return history

        if not isinstance(history, torch.Tensor):
            history = torch.as_tensor(history)
        if history.ndim == 2:
            history = history.unsqueeze(0)
        expected = (
            1,
            int(self.model.history_patch_size),
            int(self.model.latent_dim),
        )
        if tuple(history.shape) != expected:
            raise RuntimeError(
                f"Ming TTS latent_history must have shape {expected}, "
                f"got {tuple(history.shape)}"
            )
        history = history.to(device=device, dtype=torch.float32)
        data.latent_history = history
        return history
