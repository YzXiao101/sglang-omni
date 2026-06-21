# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.ming_tts.ar_state import (
    get_ming_ar_device_state_pool,
    get_ming_ar_state,
    sync_ming_ar_state_to_legacy,
)
from sglang_omni.models.ming_tts.ar_state_machine import (
    MingARStateMachine,
    MingARStepResult,
    apply_follower_ming_ar_step_result,
    normalize_ming_ar_hidden_states,
)
from sglang_omni.models.ming_tts.profile_events import emit_ming_event, tensor_metadata


class MingTTSModelRunner(ModelRunner):
    """Runs Ming-Omni-TTS AR steps and samples continuous acoustic latents."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._ar_state_machine = MingARStateMachine(self.model)
        server_args = getattr(tp_worker, "server_args", None)
        self._tp_rank = int(getattr(tp_worker, "tp_rank", 0) or 0)
        self._tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        self._feedback_buffer_contract = self._capture_feedback_buffer_contract()

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
        if any(
            bool(getattr(sched_req.data, "row_prefill_radix_cache_enabled", False))
            and getattr(sched_req.data, "prefill_input_embeds", None) is None
            for sched_req in requests
        ):
            raise RuntimeError(
                "Ming TTS row-prefill radix cache requires prefill_input_embeds; "
                "synthetic row-hash ids must not be used for embedding lookup"
            )
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
            self._validate_projected_prefill_embeds(prompt_embeds)
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

        state_pool = get_ming_ar_device_state_pool(self.model)
        self._validate_feedback_buffer_contract(state_pool)
        weight = state_pool.feedback_weight
        if forward_batch.input_ids.numel() < batch_size:
            raise RuntimeError(
                "Ming TTS decode input_ids must contain one row id per request"
            )
        state_pool.validate_batch_size(batch_size)

        rows = []
        for sched_req in requests:
            ar_state = get_ming_ar_state(sched_req.data)
            queue = ar_state.pending_feedback_queue
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
        state_pool.stage_feedback(stacked)

        row_ids = state_pool.row_ids(
            batch_size,
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
        z_diff = normalize_ming_ar_hidden_states(
            hidden,
            request_count=len(requests),
        )
        self._validate_backbone_hidden_invariants(z_diff, len(requests))

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
            if self._is_entry_rank:
                step_result = self._ar_state_machine.step_batch(z_diff, requests)
            else:
                step_result = self._empty_step_result_for_broadcast(
                    z_diff,
                    requests,
                )
            self._broadcast_step_result(step_result)
            if not self._is_entry_rank:
                apply_follower_ming_ar_step_result(step_result, requests)

            next_token_ids = step_result.next_token_ids
            result.next_token_ids = next_token_ids
            # Preserve SGLang's AR forward graph-hit flag. FlowLoss and feedback
            # staging run eager after replay, but they do not invalidate whether
            # the already-completed decode forward used CUDA graph.
            schedule_batch.output_ids = next_token_ids
        finally:
            for sched_req in requests:
                emit_ming_event(
                    sched_req.request_id,
                    "ming_tts_step_end",
                    batch_metadata,
                )

    @property
    def _is_entry_rank(self) -> bool:
        return self._tp_rank == 0

    def _empty_step_result_for_broadcast(
        self,
        hidden: torch.Tensor,
        requests: list[Any],
    ) -> MingARStepResult:
        weight = self.model._decode_input_embedding.weight
        return MingARStepResult.empty_for_broadcast(
            batch_size=len(requests),
            hidden_size=int(weight.shape[1]),
            device=hidden.device,
            feedback_dtype=weight.dtype,
            request_ids=[str(sched_req.request_id) for sched_req in requests],
        )

    def _broadcast_step_result(self, step_result: MingARStepResult) -> None:
        if self._tp_size <= 1:
            return
        for tensor in (
            step_result.next_token_ids,
            step_result.feedback_embeddings,
            step_result.feedback_mask,
            step_result.stop_flags,
            step_result.length_finish_flags,
            step_result.generation_steps,
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

    def _expected_hidden_size(self) -> int:
        weight = self.model._decode_input_embedding.weight
        expected = int(weight.shape[1])
        model_hidden_size = getattr(self.model, "hidden_size", expected)
        if int(model_hidden_size) != expected:
            raise RuntimeError(
                "Ming TTS model hidden size does not match decode feedback "
                f"embedding width ({int(model_hidden_size)} != {expected})"
            )
        return expected

    def _capture_feedback_buffer_contract(
        self,
        state_pool: Any | None = None,
    ) -> tuple[int, tuple[int, ...], torch.dtype, torch.device]:
        if state_pool is None:
            state_pool = get_ming_ar_device_state_pool(self.model)
        weight = state_pool.feedback_weight
        return (
            int(weight.data_ptr()),
            tuple(int(item) for item in weight.shape),
            weight.dtype,
            weight.device,
        )

    def _validate_feedback_buffer_contract(self, state_pool: Any) -> None:
        expected = getattr(self, "_feedback_buffer_contract", None)
        if expected is None:
            raise RuntimeError("Ming TTS feedback buffer contract is not initialized")
        actual = self._capture_feedback_buffer_contract(state_pool)
        if actual != expected:
            raise RuntimeError(
                "Ming TTS decode feedback buffer changed after runner setup; "
                "CUDA graph replay and TP row mapping require a stable "
                "feedback_weight tensor"
            )

    def _validate_projected_prefill_embeds(
        self,
        prompt_embeds: torch.Tensor,
    ) -> None:
        if prompt_embeds.ndim != 2:
            raise RuntimeError(
                "Ming TTS projected prefill embeddings must have shape "
                f"[tokens, hidden], got {tuple(prompt_embeds.shape)}"
            )
        hidden_size = int(prompt_embeds.shape[1])
        expected = self._expected_hidden_size()
        if hidden_size != expected:
            raise RuntimeError(
                "Ming TTS projected prefill hidden size mismatch: "
                f"{hidden_size} != {expected}"
            )

    def _validate_backbone_hidden_invariants(
        self,
        hidden: torch.Tensor,
        request_count: int,
    ) -> None:
        expected = self._expected_hidden_size()
        if hidden.ndim != 3:
            raise RuntimeError(
                "Ming TTS AR backbone hidden states must have shape "
                f"[batch, 1, hidden], got {tuple(hidden.shape)}"
            )
        if int(hidden.shape[0]) != int(request_count):
            raise RuntimeError(
                "Ming TTS AR backbone hidden batch mismatch: "
                f"{int(hidden.shape[0])} != {int(request_count)}"
            )
        if int(hidden.shape[1]) != 1:
            raise RuntimeError(
                "Ming TTS AR backbone must return exactly one sampled hidden "
                f"row per request, got {int(hidden.shape[1])}"
            )
        if int(hidden.shape[2]) != expected:
            raise RuntimeError(
                "Ming TTS AR backbone must return full hidden states for the "
                "rank0-owned tail; got hidden size "
                f"{int(hidden.shape[2])}, expected {expected}"
            )

    def _ensure_latent_history(
        self, data: Any, *, device: torch.device
    ) -> torch.Tensor:
        ar_state = get_ming_ar_state(data)
        history = ar_state.ensure_latent_history(
            device=device,
            history_patch_size=int(self.model.history_patch_size),
            latent_dim=int(self.model.latent_dim),
        )
        sync_ming_ar_state_to_legacy(data, ar_state)
        return history
