# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS AR state contracts.

The Ming-TTS AR loop is not a normal token-logits-sampler loop.  The
SGLang-managed backbone produces hidden states, and a Ming-specific recurrent
state machine turns those hidden states into continuous acoustic latents and
feedback embeddings.

Current step contract:

prefill:
  input: prompt input ids plus optional projected prompt embeddings
  output: prompt KV cache plus first hidden state
  side state: latent_history initialized from the prompt latent tail

decode before forward:
  input: pending feedback embedding per active request
  output: fixed feedback buffer updated in-place
  side state: input_ids rewritten to row ids

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

    This mirrors the legacy fields on ``MingTTSSGLangRequestData`` during the
    first refactor stage.  Callers should use ``get_ming_ar_state`` and
    ``sync_ming_ar_state_to_legacy`` so old and new storage stay compatible
    while the runner is migrated.
    """

    generation_steps: int = 0
    max_decode_steps: int = 0
    cfg: float = 2.0
    sigma: float = 0.25
    flow_temperature: float = 0.0
    seed: int | None = None
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

    @classmethod
    def from_legacy(cls, data: Any) -> "MingARRequestState":
        queue = getattr(data, "pending_feedback_queue", None)
        if queue is None:
            queue = collections.deque()

        generated_latents = getattr(data, "generated_latents", None)
        if generated_latents is None:
            generated_latents = []

        generated_last_chunk = getattr(data, "generated_last_chunk", None)
        if generated_last_chunk is None:
            generated_last_chunk = []

        return cls(
            generation_steps=int(getattr(data, "generation_steps", 0) or 0),
            max_decode_steps=int(getattr(data, "max_decode_steps", 0) or 0),
            cfg=float(getattr(data, "cfg", 2.0)),
            sigma=float(getattr(data, "sigma", 0.25)),
            flow_temperature=float(getattr(data, "flow_temperature", 0.0)),
            seed=getattr(data, "seed", None),
            audio_patch_token_id=int(getattr(data, "audio_patch_token_id", 0) or 0),
            audio_eos_token_id=int(getattr(data, "audio_eos_token_id", 0) or 0),
            audio_token_id=int(getattr(data, "audio_token_id", 0) or 0),
            prompt_latent_for_history=getattr(
                data,
                "prompt_latent_for_history",
                None,
            ),
            latent_history=getattr(data, "latent_history", None),
            pending_feedback_queue=queue,
            generated_latents=generated_latents,
            generated_last_chunk=generated_last_chunk,
            stop_step=getattr(data, "stop_step", None),
            engine_start_s=float(getattr(data, "engine_start_s", 0.0) or 0.0),
        )

    def sync_generation_step_from_legacy(self, data: Any) -> None:
        self.generation_steps = int(
            getattr(data, "generation_steps", self.generation_steps) or 0
        )

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


class MingARDeviceStatePool:
    """Fixed-address device state visible to Ming AR CUDA graph replay."""

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
        ar_state = MingARRequestState.from_legacy(data)
        setattr(data, "ar_state", ar_state)
    ar_state.sync_generation_step_from_legacy(data)
    sync_ming_ar_state_to_legacy(data, ar_state)
    return ar_state


def sync_ming_ar_state_to_legacy(
    data: Any,
    ar_state: MingARRequestState,
) -> None:
    data.generation_steps = int(ar_state.generation_steps)
    data.max_decode_steps = int(ar_state.max_decode_steps)
    data.cfg = float(ar_state.cfg)
    data.sigma = float(ar_state.sigma)
    data.flow_temperature = float(ar_state.flow_temperature)
    data.seed = ar_state.seed
    data.audio_patch_token_id = int(ar_state.audio_patch_token_id)
    data.audio_eos_token_id = int(ar_state.audio_eos_token_id)
    data.audio_token_id = int(ar_state.audio_token_id)
    data.prompt_latent_for_history = ar_state.prompt_latent_for_history
    data.latent_history = ar_state.latent_history
    data.pending_feedback_queue = ar_state.pending_feedback_queue
    data.generated_latents = ar_state.generated_latents
    data.generated_last_chunk = ar_state.generated_last_chunk
    data.stop_step = ar_state.stop_step
    data.engine_start_s = float(ar_state.engine_start_s)
    data.ar_state = ar_state


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
        sync_ming_ar_state_to_legacy(data, ar_state)
        return

    data.latent_history = None
    data.prompt_latent_for_history = None
    if hasattr(getattr(data, "pending_feedback_queue", None), "clear"):
        data.pending_feedback_queue.clear()
    data.generated_latents = []


def get_ming_ar_device_state_pool(model: Any) -> MingARDeviceStatePool:
    pool = getattr(model, "_ming_ar_device_state_pool", None)
    if isinstance(pool, MingARDeviceStatePool):
        return pool
    embedding = getattr(model, "_decode_input_embedding", None)
    if embedding is None:
        raise RuntimeError("Ming TTS model is missing _decode_input_embedding")
    pool = MingARDeviceStatePool(embedding)
    setattr(model, "_ming_ar_device_state_pool", pool)
    return pool


__all__ = [
    "MingARDeviceStatePool",
    "MingARRequestState",
    "get_ming_ar_device_state_pool",
    "get_ming_ar_state",
    "release_ming_ar_runtime_tensors",
    "sync_ming_ar_state_to_legacy",
    "update_ming_ar_latent_history_",
]
