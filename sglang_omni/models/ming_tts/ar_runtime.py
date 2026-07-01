# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AR runtime contracts.

This module owns request-local AR recurrence state and fixed-address feedback
buffers shared by request builders, the model runner, and result serialization.
Cross-stage serializable payload fields belong in payload_types.py; AR backbone
math and weights belong in sglang_model.py/model_runner.py.

The Ming-TTS AR loop is not a normal token-logits-sampler loop. The
SGLang-managed backbone produces hidden states, and the Ming AR tail turns
those hidden states into continuous acoustic latents and feedback embeddings.

Current step contract:

prefill:
  input: prompt input ids plus optional projected prompt embeddings
  output: prompt KV cache plus first hidden state
  side state: latent_history initialized from the prompt latent tail

decode before forward:
  input: pending feedback embedding per active request
  output: fixed feedback buffer updated in-place
  side state: input_ids rewritten to row ids
  graph padding rows: intentionally untouched and ignored by SGLang metadata

backbone forward:
  input: positions, KV cache, row feedback embedding
  output: hidden state only

state machine step:
  input: hidden state, latent_history, generation params
  output: sampled latent, stop flag, next feedback embedding, next token id
  side state: latent_history/generated_latents/pending feedback updated
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MingARRequestState:
    """Per-request Ming AR recurrence state.

    This object is the single owner of Ming AR recurrence state.  Scheduler
    request data may keep generic counters for SGLang bookkeeping, but Ming
    generation must not read AR state from those generic fields.
    """

    generation_steps: int = 0
    max_decode_steps: int = 0
    cfg: float = 2.0
    sigma: float = 0.25
    flow_temperature: float = 0.0
    audio_patch_token_id: int = 0
    audio_eos_token_id: int = 0
    audio_token_id: int = 0
    prompt_latent_for_history: Any = None
    latent_history: Any = None
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    generated_latents: list[Any] = field(default_factory=list)
    generated_last_chunk: list[bool] = field(default_factory=list)
    stop_step: int | None = None
    engine_start_s: float = 0.0

    def ensure_latent_history(
        self,
        *,
        device: torch.device,
        history_patch_size: int,
        latent_dim: int,
    ) -> torch.Tensor:
        history = self.latent_history
        if history is None:
            history = torch.zeros(
                1,
                int(history_patch_size),
                int(latent_dim),
                device=device,
                dtype=torch.float32,
            )
            prompt_latent = self.prompt_latent_for_history
            if prompt_latent is not None:
                if not isinstance(prompt_latent, torch.Tensor):
                    prompt_latent = torch.as_tensor(prompt_latent)
                if prompt_latent.ndim == 2:
                    prompt_latent = prompt_latent.unsqueeze(0)
                expected_tail = (1, int(latent_dim))
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
            self.latent_history = history
            return history

        if not isinstance(history, torch.Tensor):
            history = torch.as_tensor(history)
        if history.ndim == 2:
            history = history.unsqueeze(0)
        expected = (1, int(history_patch_size), int(latent_dim))
        if tuple(history.shape) != expected:
            raise RuntimeError(
                f"Ming TTS latent_history must have shape {expected}, "
                f"got {tuple(history.shape)}"
            )
        history = history.to(device=device, dtype=torch.float32)
        self.latent_history = history
        return history

    def release_runtime_tensors(self) -> None:
        """Drop GPU-heavy runtime references after serialization."""

        self.latent_history = None
        self.prompt_latent_for_history = None
        if hasattr(self.pending_feedback_queue, "clear"):
            self.pending_feedback_queue.clear()
        else:
            self.pending_feedback_queue = collections.deque()
        self.generated_latents.clear()


class MingARFeedbackStatePool:
    """Fixed-address active-row feedback buffer for Ming AR CUDA graph replay.

    CUDA graph replay can use a larger captured bucket than the active request
    count. This pool only writes active rows [0, batch_size); padded rows are
    intentionally left untouched and must be ignored by SGLang forward metadata.
    """

    def __init__(self, decode_input_embedding: Any) -> None:
        self.decode_input_embedding = decode_input_embedding

    @property
    def feedback_weight(self) -> torch.Tensor:
        weight = getattr(self.decode_input_embedding, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("Ming AR decode input embedding has no tensor weight")
        return weight

    @property
    def capacity(self) -> int:
        return int(self.feedback_weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.feedback_weight.shape[1])

    @property
    def device(self) -> torch.device:
        return self.feedback_weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feedback_weight.dtype

    def validate_batch_size(self, batch_size: int) -> None:
        if int(batch_size) > self.capacity:
            raise RuntimeError(
                "Ming TTS decode batch exceeds staged decode-embedding rows "
                f"({int(batch_size)} > {self.capacity})"
            )

    def stage_feedback(self, row_embeddings: torch.Tensor) -> None:
        if not isinstance(row_embeddings, torch.Tensor):
            row_embeddings = torch.as_tensor(row_embeddings)
        if row_embeddings.ndim != 2:
            raise RuntimeError(
                "Ming TTS decode feedback batch must have shape [batch, hidden], "
                f"got {tuple(row_embeddings.shape)}"
            )
        batch_size = int(row_embeddings.shape[0])
        hidden_size = int(row_embeddings.shape[1])
        self.validate_batch_size(batch_size)
        if hidden_size != self.hidden_size:
            raise RuntimeError(
                "Ming TTS decode feedback hidden size mismatch: "
                f"{hidden_size} != {self.hidden_size}"
            )
        with torch.no_grad():
            self.feedback_weight[:batch_size].copy_(
                row_embeddings.to(device=self.device, dtype=self.dtype)
            )

    def row_ids(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        self.validate_batch_size(batch_size)
        return torch.arange(
            int(batch_size),
            dtype=torch.long,
            device=device if device is not None else self.device,
        )


def get_ming_ar_state(data: Any) -> MingARRequestState:
    ar_state = getattr(data, "ar_state", None)
    if not isinstance(ar_state, MingARRequestState):
        raise RuntimeError("Ming TTS request data is missing MingARRequestState")
    return ar_state


def update_ming_ar_latent_history_(
    latent_history: torch.Tensor,
    sampled: torch.Tensor,
) -> None:
    patch = int(sampled.shape[1])
    history_len = int(latent_history.shape[1])
    if patch >= history_len:
        latent_history.copy_(
            sampled[:, -history_len:, :].to(
                device=latent_history.device,
                dtype=latent_history.dtype,
            )
        )
        return
    latent_history[:, :-patch, :].copy_(latent_history[:, patch:, :].clone())
    latent_history[:, -patch:, :].copy_(
        sampled.to(device=latent_history.device, dtype=latent_history.dtype)
    )


def release_ming_ar_runtime_tensors(data: Any) -> None:
    ar_state = getattr(data, "ar_state", None)
    if isinstance(ar_state, MingARRequestState):
        ar_state.release_runtime_tensors()


def get_ming_ar_feedback_state_pool(model: Any) -> MingARFeedbackStatePool:
    pool = getattr(model, "_ming_ar_feedback_state_pool", None)
    if isinstance(pool, MingARFeedbackStatePool):
        return pool
    embedding = getattr(model, "_decode_input_embedding", None)
    if embedding is None:
        raise RuntimeError("Ming TTS model is missing _decode_input_embedding")
    pool = MingARFeedbackStatePool(embedding)
    setattr(model, "_ming_ar_feedback_state_pool", pool)
    return pool


__all__ = [
    "MingARFeedbackStatePool",
    "MingARRequestState",
    "get_ming_ar_feedback_state_pool",
    "get_ming_ar_state",
    "release_ming_ar_runtime_tensors",
    "update_ming_ar_latent_history_",
]
