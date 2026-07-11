# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.ming_tts.sglang_model import MingTTSTailInputs


@dataclass
class MingTTSTPStepUpdate:
    """Rank-synchronized output of one Ming AR recurrence step.

    Only the entry rank owns generated acoustic latents for serialization.
    Follower ranks consume the synchronized token, stop, and feedback fields
    only to keep their next backbone decode input aligned.
    """

    control_tensor: torch.Tensor
    feedback_embeddings: torch.Tensor

    @classmethod
    def empty_for_broadcast(
        cls,
        *,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        feedback_dtype: torch.dtype,
    ) -> "MingTTSTPStepUpdate":
        return cls(
            control_tensor=torch.zeros(
                3,
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
        )

    @property
    def next_token_ids(self) -> torch.Tensor:
        return self.control_tensor[0]

    @property
    def feedback_mask(self) -> torch.Tensor:
        return self.control_tensor[1]

    @property
    def stop_flags(self) -> torch.Tensor:
        return self.control_tensor[2]


class MingTTSModelRunner(ModelRunner):
    """Runs Ming-Omni-TTS AR steps and samples continuous acoustic latents."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        server_args = getattr(tp_worker, "server_args", None)
        self._tp_rank = int(getattr(tp_worker, "tp_rank", 0) or 0)
        self._tp_size = int(getattr(server_args, "tp_size", 1) or 1)

    def before_prefill(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch, schedule_batch
        device = self.model._decode_input_embedding.weight.device
        for sched_req in requests:
            sched_req.data.decode_state.ensure_latent_history(
                device=device,
                history_patch_size=int(self.model.history_patch_size),
                latent_dim=int(self.model.latent_dim),
            )

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        batch_parts = []
        dtype = self.model._decode_input_embedding.weight.dtype
        device = forward_batch.input_ids.device
        embedding = self.model.get_input_embeddings()
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_input_len)
            end = prefix_len + extend_len
            prompt_ids = data.prompt_input_ids
            prompt_len = int(prompt_ids.shape[0])
            req_parts = []

            prompt_start = min(prefix_len, prompt_len)
            prompt_stop = min(end, prompt_len)
            if prompt_stop > prompt_start:
                if data.prefill_input_embeds is None:
                    prompt_rows = embedding(
                        prompt_ids[prompt_start:prompt_stop].to(device=device)
                    ).to(dtype=dtype)
                else:
                    prompt_rows = data.prefill_input_embeds[
                        prompt_start:prompt_stop
                    ].to(device=device, dtype=dtype)
                req_parts.append(prompt_rows)

            # Note (yzxiao): Retraction may re-prefill generated audio tokens,
            # whose rows live in feedback embeddings rather than token embeds.
            gen_start = max(prefix_len, prompt_len) - prompt_len
            gen_end = max(end - prompt_len, 0)
            if gen_end > gen_start:
                feedback_history = data.decode_input_embeds or []
                if len(feedback_history) < gen_end:
                    raise RuntimeError(
                        "Ming TTS generated feedback history is shorter than "
                        "request output_ids during retraction prefill"
                    )
                feedback_rows = [
                    feedback.to(device=device, dtype=dtype)
                    for feedback in feedback_history[gen_start:gen_end]
                ]
                req_parts.append(torch.stack(feedback_rows, dim=0))

            req_embeds = torch.cat(req_parts, dim=0)
            batch_parts.append(req_embeds)
        return torch.cat(batch_parts, dim=0)

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=self.model._decode_input_embedding.weight.dtype,
        )

        model_runner = self.tp_worker.model_runner
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        logits_output = self.model(
            input_ids=forward_batch.input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
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

        rows = []
        weight = self.model._decode_input_embedding.weight
        for sched_req in requests:
            data = sched_req.data
            rows.append(
                data.decode_input_embeds[-1].to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )

        row_ids = self.model.stage_decode_feedback(torch.stack(rows, dim=0))
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
        self._collect_ming_tts_step(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._collect_ming_tts_step(result, forward_batch, schedule_batch, requests)

    def finalize_skip_rids(self, scheduler_output: Any) -> set[str]:
        batch = getattr(scheduler_output, "batch_data", None)
        if bool(getattr(batch, "is_prefill_only", False)):
            return {sched_req.request_id for sched_req in scheduler_output.requests}
        return set()

    def _collect_ming_tts_step(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        del forward_batch
        if not requests:
            return

        hidden = result.logits_output.hidden_states
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        hidden_states = hidden

        if self._is_entry_rank:
            step_update = self._run_entry_tail_step(hidden_states, requests)
        else:
            weight = self.model._decode_input_embedding.weight
            step_update = MingTTSTPStepUpdate.empty_for_broadcast(
                batch_size=len(requests),
                hidden_size=int(weight.shape[1]),
                device=hidden_states.device,
                feedback_dtype=weight.dtype,
            )
        self._broadcast_tp_step_update(step_update)
        if not self._is_entry_rank:
            self._apply_follower_step_update(step_update, requests)

        next_token_ids = step_update.next_token_ids
        result.next_token_ids = next_token_ids
        schedule_batch.output_ids = next_token_ids

    def _run_entry_tail_step(
        self,
        hidden_states: torch.Tensor,
        requests: list[Any],
    ) -> MingTTSTPStepUpdate:
        weight = self.model._decode_input_embedding.weight
        device = hidden_states.device
        batch_size = len(requests)
        hidden_size = int(weight.shape[1])
        step_update = MingTTSTPStepUpdate.empty_for_broadcast(
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            feedback_dtype=weight.dtype,
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
            decode_states = [req.data.decode_state for req in requests]
            steps = [int(req.data.generation_steps) for req in requests]
            max_steps = [int(state.max_decode_steps) for state in decode_states]
            histories = [
                state.ensure_latent_history(
                    device=device,
                    history_patch_size=int(self.model.history_patch_size),
                    latent_dim=int(self.model.latent_dim),
                )
                for state in decode_states
            ]
            history_batch = torch.cat(histories, dim=0)
            steps_tensor = torch.tensor(steps, dtype=torch.long, device=device)
            max_steps_tensor = torch.tensor(
                max_steps,
                dtype=torch.long,
                device=device,
            )
            cfg_tensor = torch.tensor(
                [float(state.cfg) for state in decode_states],
                dtype=torch.float32,
                device=device,
            )
            sigma_tensor = torch.tensor(
                [float(state.sigma) for state in decode_states],
                dtype=torch.float32,
                device=device,
            )
            temperature_tensor = torch.tensor(
                [float(state.temperature) for state in decode_states],
                dtype=torch.float32,
                device=device,
            )

            tail_outputs = self.model.run_tail_step(
                MingTTSTailInputs(
                    hidden_states=hidden_states,
                    latent_history=history_batch,
                    cfg=cfg_tensor,
                    sigma=sigma_tensor,
                    temperature=temperature_tensor,
                )
            )
            sampled = tail_outputs.sampled
            stop_prob = tail_outputs.stop_prob
            feedback_embeddings = tail_outputs.feedback_embeddings
            stop_flags = (stop_prob > 0.5) & (steps_tensor > 3)
            length_flags = steps_tensor + 1 >= max_steps_tensor
            feedback_mask = ~(stop_flags | length_flags)
            step_update.stop_flags.copy_(stop_flags)
            step_update.feedback_mask.copy_(feedback_mask)
            decision_rows = torch.stack((stop_flags, length_flags)).cpu().tolist()
            stop_list, length_list = decision_rows
            for row_idx, decode_state in enumerate(decode_states):
                step = steps[row_idx]
                sampled_row = sampled[row_idx : row_idx + 1]
                sampled_chunk = sampled_row.squeeze(0).detach()
                decode_state.generated_latents.append(sampled_chunk)

                stop = stop_list[row_idx]
                length = length_list[row_idx]
                decode_state.generated_last_chunk.append(stop or length)
                if stop:
                    decode_state.stop_step = step
                    next_ids.append(int(decode_state.audio_eos_token_id))
                    continue

                self._advance_latent_history(
                    decode_state.latent_history,
                    sampled_row,
                )
                next_ids.append(int(decode_state.audio_patch_token_id))
                if not length:
                    feedback = feedback_embeddings[row_idx].detach()
                    data = requests[row_idx].data
                    data.decode_input_embeds.append(feedback)
                    step_update.feedback_embeddings[row_idx].copy_(
                        feedback.to(
                            device=step_update.feedback_embeddings.device,
                            dtype=step_update.feedback_embeddings.dtype,
                        )
                    )

        step_update.next_token_ids.copy_(
            torch.tensor(next_ids, dtype=torch.long, device=device)
        )
        return step_update

    @staticmethod
    def _advance_latent_history(
        latent_history: torch.Tensor,
        sampled_row: torch.Tensor,
    ) -> None:
        patch = int(sampled_row.shape[1])
        history_len = int(latent_history.shape[1])
        sampled_row = sampled_row.to(
            device=latent_history.device,
            dtype=latent_history.dtype,
        )
        if patch >= history_len:
            latent_history.copy_(sampled_row[:, -history_len:, :])
            return
        latent_history[:, :-patch, :].copy_(latent_history[:, patch:, :].clone())
        latent_history[:, -patch:, :].copy_(sampled_row)

    def _apply_follower_step_update(
        self,
        step_update: MingTTSTPStepUpdate,
        requests: list[Any],
    ) -> None:
        feedback_list, stop_list = step_update.control_tensor[1:].cpu().tolist()
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            decode_state = data.decode_state
            step = int(data.generation_steps)
            if stop_list[row_idx]:
                decode_state.stop_step = step
            if feedback_list[row_idx]:
                feedback = step_update.feedback_embeddings[row_idx].detach().clone()
                data.decode_input_embeds.append(feedback)

    @property
    def _is_entry_rank(self) -> bool:
        # Note (yzxiao): FlowLoss is not tensor-parallel, so rank 0 owns
        # acoustic sampling while followers only mirror the next AR input.
        return self._tp_rank == 0

    def _broadcast_tp_step_update(self, step_update: MingTTSTPStepUpdate) -> None:
        if self._tp_size <= 1:
            return
        for tensor in (
            step_update.control_tensor,
            step_update.feedback_embeddings,
        ):
            self._broadcast_tensor_from_entry(tensor)

    def _broadcast_tensor_from_entry(self, tensor: torch.Tensor) -> None:
        import torch.distributed as dist

        tp_group = self._get_tp_group()
        if tp_group is None:
            raise RuntimeError("Ming TTS TP broadcast requires a TP group")
        ranks = getattr(tp_group, "ranks", None)
        src_rank = int(ranks[0]) if ranks else int(getattr(tp_group, "first_rank", 0))
        dist_group = getattr(tp_group, "device_group", None)
        if dist_group is None:
            dist_group = getattr(tp_group, "group", None)
        dist.broadcast(tensor, src=src_rank, group=dist_group)

    def _get_tp_group(self) -> Any:
        getter = getattr(self.tp_worker, "get_tp_group", None)
        if callable(getter):
            return getter()
        model_runner = getattr(self.tp_worker, "model_runner", None)
        return getattr(model_runner, "tp_group", None)
