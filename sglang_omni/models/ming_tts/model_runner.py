# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.ming_tts.ar_runtime import (
    get_ming_ar_feedback_state_pool,
    get_ming_ar_state,
    update_ming_ar_latent_history_,
)
from sglang_omni.models.ming_tts.fm.cfm import build_cfm_sampling_schedule
from sglang_omni.models.ming_tts.profile_events import (
    emit_ming_event,
    ming_profile_event,
    tensor_metadata,
)

_MING_TTS_CFM_STEPS = 10


@dataclass
class MingARStepResult:
    """Rank-synchronized output of one Ming AR recurrence step.

    Only the entry rank owns generated acoustic latents for serialization.
    Follower ranks consume the synchronized token, stop, and feedback fields
    only to keep their next backbone decode input aligned.
    """

    next_token_ids: torch.Tensor
    feedback_embeddings: torch.Tensor
    feedback_mask: torch.Tensor
    stop_flags: torch.Tensor
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
            generation_steps=torch.zeros(
                int(batch_size),
                dtype=torch.long,
                device=device,
            ),
            request_ids=list(request_ids),
            generated_latents=[None for _ in range(int(batch_size))],
        )


@dataclass
class MingARTailTensorOutputs:
    """Pure tensor outputs produced by the Ming AR tail compute boundary."""

    sampled: torch.Tensor
    feedback_embeddings: torch.Tensor
    stop_prob: torch.Tensor


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


def build_ming_tts_cfm_sampling_inputs(
    *,
    batch_size: int,
    device: torch.device,
    latent_dim: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_rows: list[torch.Tensor] = []
    sde_rows: list[torch.Tensor] = []
    timesteps = None
    for _ in range(int(batch_size)):
        row_noise = torch.randn(
            1,
            int(latent_dim),
            int(patch_size),
            device=device,
        )
        row_timesteps, row_sde_random = build_cfm_sampling_schedule(
            steps=_MING_TTS_CFM_STEPS,
            device=device,
            dtype=row_noise.dtype,
            batch_size=1,
            patch_size=int(patch_size),
            latent_dim=int(latent_dim),
        )
        noise_rows.append(row_noise)
        sde_rows.append(row_sde_random)
        if timesteps is None:
            timesteps = row_timesteps

    if timesteps is None:
        raise RuntimeError("Ming TTS CFM sampling inputs require batch_size > 0")

    noise = torch.cat(noise_rows, dim=0)
    sde_random = torch.cat(sde_rows, dim=1)
    return noise, timesteps, sde_random


def _validate_ming_tail_cfg_values(cfg: torch.Tensor | list[float]) -> None:
    if isinstance(cfg, torch.Tensor):
        invalid = torch.logical_or(cfg < 1e-5, cfg == 1.0)
        is_invalid = bool(torch.any(invalid).detach().cpu().item())
    else:
        is_invalid = any(value < 1e-5 or value == 1.0 for value in cfg)
    if is_invalid:
        raise NotImplementedError(
            "Ming-Omni-TTS tail requires guided CFM sampling "
            "with cfg >= 1e-5 and cfg != 1.0"
        )


class MingARTailGraphExecutor:
    """Exact-batch CUDA graph for the pure Ming AR tail tensor compute."""

    def __init__(self, model: Any, batch_size: int) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.initialized = False
        self.graph: torch.cuda.CUDAGraph | None = None
        self.z_diff_placeholder: torch.Tensor | None = None
        self.history_placeholder: torch.Tensor | None = None
        self.noise_placeholder: torch.Tensor | None = None
        self.timesteps_placeholder: torch.Tensor | None = None
        self.sde_random_placeholder: torch.Tensor | None = None
        self.cfg_placeholder: torch.Tensor | None = None
        self.sigma_placeholder: torch.Tensor | None = None
        self.temperature_placeholder: torch.Tensor | None = None
        self.sampled_placeholder: torch.Tensor | None = None
        self.feedback_placeholder: torch.Tensor | None = None
        self.stop_prob_placeholder: torch.Tensor | None = None

    def execute(
        self,
        *,
        z_diff: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
        validate_cfg: bool = True,
    ) -> MingARTailTensorOutputs:
        self._validate_inputs(
            z_diff=z_diff,
            latent_history=latent_history,
            noise=noise,
            timesteps=timesteps,
            sde_random=sde_random,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        )
        if validate_cfg:
            _validate_ming_tail_cfg_values(cfg)
        if not self.initialized:
            self._initialize_graph(
                z_diff=z_diff,
                latent_history=latent_history,
                noise=noise,
                timesteps=timesteps,
                sde_random=sde_random,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
            )

        self._copy_inputs(
            z_diff=z_diff,
            latent_history=latent_history,
            noise=noise,
            timesteps=timesteps,
            sde_random=sde_random,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        )
        if self.graph is None:
            raise RuntimeError("Ming TTS tail CUDA graph is not initialized")
        self.graph.replay()

        sampled = torch.empty_like(self._required_tensor(self.sampled_placeholder))
        sampled.copy_(self._required_tensor(self.sampled_placeholder))
        feedback = torch.empty_like(self._required_tensor(self.feedback_placeholder))
        feedback.copy_(self._required_tensor(self.feedback_placeholder))
        stop_prob = torch.empty_like(self._required_tensor(self.stop_prob_placeholder))
        stop_prob.copy_(self._required_tensor(self.stop_prob_placeholder))
        return MingARTailTensorOutputs(
            sampled=sampled,
            feedback_embeddings=feedback,
            stop_prob=stop_prob,
        )

    def _initialize_graph(
        self,
        *,
        z_diff: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        self.z_diff_placeholder = torch.empty_like(z_diff)
        self.history_placeholder = torch.empty_like(latent_history)
        self.noise_placeholder = torch.empty_like(noise)
        self.timesteps_placeholder = torch.empty_like(timesteps)
        self.sde_random_placeholder = torch.empty_like(sde_random)
        self.cfg_placeholder = torch.empty_like(cfg)
        self.sigma_placeholder = torch.empty_like(sigma)
        self.temperature_placeholder = torch.empty_like(temperature)
        self._copy_inputs(
            z_diff=z_diff,
            latent_history=latent_history,
            noise=noise,
            timesteps=timesteps,
            sde_random=sde_random,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        )

        self._run_tail_compute()
        torch.cuda.synchronize(device=z_diff.device)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                (
                    self.sampled_placeholder,
                    self.feedback_placeholder,
                    self.stop_prob_placeholder,
                ) = self._run_tail_compute()
        except BaseException:
            self.graph = None
            self.initialized = False
            self.sampled_placeholder = None
            self.feedback_placeholder = None
            self.stop_prob_placeholder = None
            raise

        self.graph = graph
        self.initialized = True

    def _run_tail_compute(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_diff = self._required_tensor(self.z_diff_placeholder)
        sampled = self.model.flowloss.sample_final_with_noise(
            z=z_diff,
            latent_history=self._required_tensor(self.history_placeholder),
            noise=self._required_tensor(self.noise_placeholder),
            cfg=self._required_tensor(self.cfg_placeholder),
            patch_size=int(self.model.patch_size),
            sigma=self._required_tensor(self.sigma_placeholder),
            temperature=self._required_tensor(self.temperature_placeholder),
            timesteps=self._required_tensor(self.timesteps_placeholder),
            sde_random=self._required_tensor(self.sde_random_placeholder),
            validate_cfg=False,
        )
        feedback = self.model.linear_proj_audio(sampled).reshape(self.batch_size, -1)
        stop_prob = self.model.stop_head(z_diff).softmax(dim=-1)[:, 0, 1]
        return sampled, feedback, stop_prob

    def _copy_inputs(
        self,
        *,
        z_diff: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        self._required_tensor(self.z_diff_placeholder).copy_(z_diff)
        self._required_tensor(self.history_placeholder).copy_(latent_history)
        self._required_tensor(self.noise_placeholder).copy_(noise)
        self._required_tensor(self.timesteps_placeholder).copy_(timesteps)
        self._required_tensor(self.sde_random_placeholder).copy_(sde_random)
        self._required_tensor(self.cfg_placeholder).copy_(cfg)
        self._required_tensor(self.sigma_placeholder).copy_(sigma)
        self._required_tensor(self.temperature_placeholder).copy_(temperature)

    def _validate_inputs(
        self,
        *,
        z_diff: torch.Tensor,
        latent_history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sde_random: torch.Tensor,
        cfg: torch.Tensor,
        sigma: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        tensors = {
            "z_diff": z_diff,
            "latent_history": latent_history,
            "noise": noise,
            "timesteps": timesteps,
            "sde_random": sde_random,
            "cfg": cfg,
            "sigma": sigma,
            "temperature": temperature,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Ming TTS tail graph input {name} is not a tensor")
            if tensor.device.type != "cuda":
                raise RuntimeError("Ming TTS tail CUDA graph requires CUDA tensors")
            if tensor.device != z_diff.device:
                raise RuntimeError(
                    "Ming TTS tail graph inputs must be on one CUDA device"
                )

        batch_size = self.batch_size
        expected_noise = (
            batch_size,
            int(self.model.latent_dim),
            int(self.model.patch_size),
        )
        expected_history = (
            batch_size,
            int(self.model.history_patch_size),
            int(self.model.latent_dim),
        )
        expected_sde = (
            _MING_TTS_CFM_STEPS,
            batch_size,
            int(self.model.patch_size),
            int(self.model.latent_dim),
        )
        expected_hidden = self._expected_hidden_size()
        if (
            int(z_diff.shape[0]) != batch_size
            or z_diff.ndim != 3
            or int(z_diff.shape[1]) != 1
            or int(z_diff.shape[2]) != expected_hidden
        ):
            raise RuntimeError(
                "Ming TTS tail graph hidden shape mismatch: "
                f"got {tuple(z_diff.shape)}, expected "
                f"({batch_size}, 1, {expected_hidden})"
            )
        if tuple(latent_history.shape) != expected_history:
            raise RuntimeError(
                "Ming TTS tail graph history shape mismatch: "
                f"expected {expected_history}, got {tuple(latent_history.shape)}"
            )
        if tuple(noise.shape) != expected_noise:
            raise RuntimeError(
                "Ming TTS tail graph noise shape mismatch: "
                f"expected {expected_noise}, got {tuple(noise.shape)}"
            )
        if tuple(sde_random.shape) != expected_sde:
            raise RuntimeError(
                "Ming TTS tail graph sde_random shape mismatch: "
                f"expected {expected_sde}, got {tuple(sde_random.shape)}"
            )
        if tuple(timesteps.shape) != (_MING_TTS_CFM_STEPS + 1,):
            raise RuntimeError(
                "Ming TTS tail graph timesteps shape mismatch: "
                f"got {tuple(timesteps.shape)}"
            )
        for name, tensor in (
            ("cfg", cfg),
            ("sigma", sigma),
            ("temperature", temperature),
        ):
            if tuple(tensor.shape) != (batch_size,):
                raise RuntimeError(
                    f"Ming TTS tail graph {name} must have shape "
                    f"({batch_size},), got {tuple(tensor.shape)}"
                )

    @staticmethod
    def _required_tensor(tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError("Ming TTS tail CUDA graph tensor is not initialized")
        return tensor

    def _expected_hidden_size(self) -> int:
        embedding = getattr(self.model, "_decode_input_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return int(weight.shape[1])
        return int(getattr(self.model, "hidden_size"))


class MingARTailGraphExecutorCache:
    """Lazy exact-batch CUDA graph cache for Ming AR tail compute."""

    def __init__(self, model: Any, *, max_batch_size: int | None = None) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self._executors: dict[int, MingARTailGraphExecutor] = {}

    def execute(self, **kwargs) -> MingARTailTensorOutputs:
        z_diff = kwargs["z_diff"]
        batch_size = int(z_diff.shape[0])
        if self.max_batch_size is not None and batch_size > int(self.max_batch_size):
            raise RuntimeError(
                "Ming TTS tail CUDA graph batch exceeds configured capacity: "
                f"{batch_size} > {int(self.max_batch_size)}"
            )
        executor = self._executors.get(batch_size)
        if executor is None:
            executor = MingARTailGraphExecutor(self.model, batch_size)
            self._executors[batch_size] = executor
        return executor.execute(**kwargs)


class MingARTailExecutor:
    """Ming AR tail compute after the SGLang backbone hidden-state forward."""

    def __init__(
        self,
        model: Any,
        *,
        enable_cuda_graph: bool = False,
        cuda_graph_max_bs: int | None = None,
    ) -> None:
        self.model = model
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self._tail_graph_cache = (
            MingARTailGraphExecutorCache(
                model,
                max_batch_size=cuda_graph_max_bs,
            )
            if self.enable_cuda_graph
            else None
        )

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

        batch_event_id = str(requests[0].request_id)
        with context:
            with ming_profile_event(
                batch_event_id,
                "ming_tail_gather",
                {
                    "batch_size": int(batch_size),
                    "hidden": tensor_metadata(z_diff),
                },
            ):
                ar_states = [get_ming_ar_state(req.data) for req in requests]
                steps = [int(state.generation_steps) for state in ar_states]
                max_steps = [int(state.max_decode_steps) for state in ar_states]
                cfg_values = [float(state.cfg) for state in ar_states]
                _validate_ming_tail_cfg_values(cfg_values)
                histories = [
                    state.ensure_latent_history(
                        device=device,
                        history_patch_size=int(self.model.history_patch_size),
                        latent_dim=int(self.model.latent_dim),
                    )
                    for state in ar_states
                ]
                history_batch = torch.cat(histories, dim=0)
                steps_tensor = torch.tensor(steps, dtype=torch.long, device=device)
                max_steps_tensor = torch.tensor(
                    max_steps,
                    dtype=torch.long,
                    device=device,
                )
                cfg_tensor = torch.tensor(
                    cfg_values,
                    dtype=torch.float32,
                    device=device,
                )
                sigma_tensor = torch.tensor(
                    [float(state.sigma) for state in ar_states],
                    dtype=torch.float32,
                    device=device,
                )
                temperature_tensor = torch.tensor(
                    [float(state.flow_temperature) for state in ar_states],
                    dtype=torch.float32,
                    device=device,
                )

            noise, timesteps, sde_random = build_ming_tts_cfm_sampling_inputs(
                batch_size=batch_size,
                device=device,
                latent_dim=int(self.model.latent_dim),
                patch_size=int(self.model.patch_size),
            )
            feedback_all: torch.Tensor | None = None
            if self.enable_cuda_graph:
                if self._tail_graph_cache is None:
                    raise RuntimeError("Ming TTS tail graph cache is not initialized")
                with ming_profile_event(
                    batch_event_id,
                    "ming_tail_graph_replay",
                    {
                        "batch_size": int(batch_size),
                        "hidden": tensor_metadata(z_diff),
                        "history": tensor_metadata(history_batch),
                    },
                ):
                    graph_outputs = self._tail_graph_cache.execute(
                        z_diff=z_diff,
                        latent_history=history_batch,
                        noise=noise,
                        timesteps=timesteps,
                        sde_random=sde_random,
                        cfg=cfg_tensor,
                        sigma=sigma_tensor,
                        temperature=temperature_tensor,
                        validate_cfg=False,
                    )
                sampled = graph_outputs.sampled
                stop_prob = graph_outputs.stop_prob
                feedback_all = graph_outputs.feedback_embeddings
            else:
                with ming_profile_event(
                    batch_event_id,
                    "ming_flowloss_batch_sample",
                    {
                        "batch_size": int(batch_size),
                        "hidden": tensor_metadata(z_diff),
                        "history": tensor_metadata(history_batch),
                    },
                ):
                    sampled, _trajectory = self.model.flowloss.sample_with_noise(
                        z=z_diff,
                        latent_history=history_batch,
                        noise=noise,
                        cfg=cfg_tensor,
                        patch_size=int(self.model.patch_size),
                        sigma=sigma_tensor,
                        temperature=temperature_tensor,
                        timesteps=timesteps,
                        sde_random=sde_random,
                        validate_cfg=False,
                    )
            if not isinstance(sampled, torch.Tensor):
                raise RuntimeError("Ming TTS sampled latent must be a tensor")
            expected = (
                int(batch_size),
                int(self.model.patch_size),
                int(self.model.latent_dim),
            )
            if tuple(sampled.shape) != expected:
                raise RuntimeError(
                    f"Ming TTS sampled latent must have shape {expected}, "
                    f"got {tuple(sampled.shape)}"
                )

            if feedback_all is not None:
                expected_feedback = (int(batch_size), int(hidden_size))
                if tuple(feedback_all.shape) != expected_feedback:
                    raise RuntimeError(
                        "Ming TTS tail graph feedback shape mismatch: "
                        f"expected {expected_feedback}, got {tuple(feedback_all.shape)}"
                    )
            else:
                with ming_profile_event(
                    batch_event_id,
                    "ming_stop_head_batch",
                    {
                        "batch_size": int(batch_size),
                        "hidden": tensor_metadata(z_diff),
                    },
                ):
                    stop_prob = self.model.stop_head(z_diff).softmax(dim=-1)[:, 0, 1]
            if tuple(stop_prob.shape) != (int(batch_size),):
                raise RuntimeError(
                    "Ming TTS stop probability shape mismatch: "
                    f"got {tuple(stop_prob.shape)}"
                )
            with ming_profile_event(
                batch_event_id,
                "ming_stop_decision_batch",
                {
                    "batch_size": int(batch_size),
                    "stop_prob": tensor_metadata(stop_prob),
                },
            ):
                stop_flags = (stop_prob > 0.5) & (steps_tensor > 3)

            length_flags = steps_tensor + 1 >= max_steps_tensor
            continuation_flags = torch.logical_not(
                torch.logical_or(stop_flags, length_flags)
            )
            feedback_rows = torch.nonzero(
                continuation_flags,
                as_tuple=False,
            ).flatten()
            feedback_row_count = int(feedback_rows.numel())
            feedback_by_row: dict[int, torch.Tensor] = {}
            feedback_rows_list: list[int] = []
            if feedback_all is not None:
                feedback_rows_list = [
                    int(row) for row in feedback_rows.detach().cpu().tolist()
                ]
            elif feedback_row_count:
                with ming_profile_event(
                    batch_event_id,
                    "ming_feedback_proj_batch",
                    {
                        "batch_size": int(batch_size),
                        "continuation_count": int(feedback_row_count),
                        "sampled": tensor_metadata(sampled),
                    },
                ):
                    feedback_batch = self.model.linear_proj_audio(
                        sampled.index_select(0, feedback_rows)
                    )
                feedback_batch = feedback_batch.reshape(feedback_row_count, -1)
                if int(feedback_batch.shape[1]) != hidden_size:
                    raise RuntimeError(
                        "Ming TTS feedback projection hidden size mismatch: "
                        f"{int(feedback_batch.shape[1])} != {hidden_size}"
                    )
                feedback_rows_list = [
                    int(row) for row in feedback_rows.detach().cpu().tolist()
                ]
                for local_idx, row_idx in enumerate(feedback_rows_list):
                    feedback_by_row[row_idx] = feedback_batch[local_idx].detach()

            stop_list = [bool(value) for value in stop_flags.detach().cpu().tolist()]
            length_list = [
                bool(value) for value in length_flags.detach().cpu().tolist()
            ]
            with ming_profile_event(
                batch_event_id,
                "ming_tail_scatter",
                {
                    "batch_size": int(batch_size),
                    "stop_count": int(sum(stop_list)),
                    "length_count": int(sum(length_list)),
                    "continuation_count": int(len(feedback_rows_list)),
                },
            ):
                for row_idx, ar_state in enumerate(ar_states):
                    step = steps[row_idx]
                    result.generation_steps[row_idx] = step
                    sampled_row = sampled[row_idx : row_idx + 1]
                    sampled_chunk = sampled_row.squeeze(0).detach()
                    ar_state.generated_latents.append(sampled_chunk)
                    result.generated_latents[row_idx] = sampled_chunk

                    stop = stop_list[row_idx]
                    ar_state.generated_last_chunk.append(stop)
                    result.stop_flags[row_idx] = stop
                    if stop:
                        ar_state.stop_step = step
                        next_ids.append(int(ar_state.audio_eos_token_id))
                        ar_state.generation_steps = step + 1
                        continue

                    if ar_state.latent_history is None:
                        raise RuntimeError(
                            "Ming TTS latent_history must be initialized"
                        )
                    update_ming_ar_latent_history_(
                        ar_state.latent_history,
                        sampled_row,
                    )

                    next_ids.append(int(ar_state.audio_patch_token_id))
                    if not length_list[row_idx]:
                        if feedback_all is not None:
                            feedback = feedback_all[row_idx].detach()
                        else:
                            feedback = feedback_by_row[row_idx]
                        result.feedback_embeddings[row_idx].copy_(
                            feedback.to(
                                device=result.feedback_embeddings.device,
                                dtype=result.feedback_embeddings.dtype,
                            )
                        )
                        result.feedback_mask[row_idx] = True
                        ar_state.pending_feedback_queue.append(feedback)
                    ar_state.generation_steps = step + 1

        result.next_token_ids.copy_(
            torch.tensor(next_ids, dtype=torch.long, device=device)
        )
        return result


def apply_follower_ming_ar_step_result(
    step_result: MingARStepResult,
    requests: list[Any],
) -> None:
    """Apply entry-rank step metadata needed by TP follower ranks.

    Follower ranks intentionally do not mirror generated acoustic latents.
    Serialization must stay on the entry rank.
    """

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


class MingTTSModelRunner(ModelRunner):
    """Runs Ming-Omni-TTS AR steps and samples continuous acoustic latents."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        server_args = getattr(tp_worker, "server_args", None)
        tail_graph_max_bs = getattr(
            server_args,
            "ming_ar_tail_cuda_graph_max_bs",
            None,
        )
        if tail_graph_max_bs is None:
            tail_graph_max_bs = getattr(server_args, "max_running_requests", None)
        self._ar_tail = MingARTailExecutor(
            self.model,
            enable_cuda_graph=bool(
                getattr(server_args, "enable_ming_ar_tail_cuda_graph", False)
            ),
            cuda_graph_max_bs=tail_graph_max_bs,
        )
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

        state_pool = get_ming_ar_feedback_state_pool(self.model)
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
                step_result = self._ar_tail.step_batch(z_diff, requests)
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
        # Entry rank owns FlowLoss sampling and serialized acoustic output.
        # TP followers keep only the metadata needed for the next backbone step.
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
        # FIXME: Pack small TP metadata tensors into one collective if profiling
        # shows per-step NCCL launch overhead dominates low-batch decode.
        for tensor in (
            step_result.next_token_ids,
            step_result.feedback_embeddings,
            step_result.feedback_mask,
            step_result.stop_flags,
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
            state_pool = get_ming_ar_feedback_state_pool(self.model)
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
        return history
