# SPDX-License-Identifier: Apache-2.0
"""SGLang model class for BailingMoeV2 (Ming-Omni thinker).

This implements the BailingMoeV2ForCausalLM as a native SGLang model,
enabling paged KV cache, RadixAttention, and FusedMoE support.

Architecture (from config.json):
  - 32 layers, 32 attention heads, 4 KV heads (GQA)
  - Hidden 4096, intermediate 9216, MoE intermediate 1024
  - 256 experts, 8/token, 1 shared expert, MultiRouter
  - partial_rotary_factor=0.5, rope_theta=2.4M
  - use_qk_norm=True, use_expert_bias=True
  - first_k_dense_replace=1 (layer 0 is dense, rest are MoE)
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
from torch import nn

from sglang_omni.models.ming_omni.configuration import (
    BailingMM2Config,
    BailingMoeV2Config,
)
from sglang_omni.models.ming_omni.tp_utils import validate_attention_tp_config
from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang_omni.vendor.sglang.layers import (
    LayerCommunicator,
    LayerScatterModes,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizationConfig,
    RadixAttention,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_moe_impl_class,
    get_rope,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)

__all__ = [
    "BailingMM2Config",
    "BailingMoeV2Config",
    "BailingMoeV2ForCausalLM",
]


@dataclass
class _WeightLoadCategoryStats:
    count: int = 0
    num_bytes: int = 0
    seconds: float = 0.0

    def add(self, tensor: Any, elapsed_s: float = 0.0) -> None:
        self.count += 1
        self.num_bytes += _tensor_nbytes(tensor)
        self.seconds += elapsed_s


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.numel() * tensor.element_size())
    except Exception:
        return 0


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f}GiB"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_moe_layer_id(name: str) -> int | None:
    parts = name.split(".", 3)
    if len(parts) >= 2 and parts[0] == "layers":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _gib_per_second(stats: _WeightLoadCategoryStats) -> float:
    if stats.seconds <= 0.0:
        return 0.0
    return stats.num_bytes / (1024**3) / stats.seconds


def _avg_ms_per_tensor(stats: _WeightLoadCategoryStats) -> float:
    if stats.count <= 0:
        return 0.0
    return stats.seconds * 1000.0 / stats.count


def _format_moe_bucket_key(key: Any) -> str:
    if isinstance(key, tuple):
        return "/".join(str(part) for part in key)
    return str(key)


def _device_label(obj: Any) -> str:
    try:
        device = getattr(obj, "device", None)
        if device is None and hasattr(obj, "data"):
            device = getattr(obj.data, "device", None)
        return str(device) if device is not None else "unknown"
    except Exception:
        return "unknown"


def _format_top_weight_load_buckets(
    stats_by_key: dict[Any, _WeightLoadCategoryStats],
    *,
    limit: int,
) -> str:
    top_items = sorted(
        stats_by_key.items(),
        key=lambda item: item[1].seconds,
        reverse=True,
    )[:limit]
    return (
        "["
        + ", ".join(
            (
                f"{_format_moe_bucket_key(key)}:"
                f"count={stats.count},bytes={_format_gib(stats.num_bytes)},"
                f"s={stats.seconds:.2f},avg_ms={_avg_ms_per_tensor(stats):.2f},"
                f"gib_s={_gib_per_second(stats):.2f}"
            )
            for key, stats in top_items
        )
        + "]"
    )


def _format_moe_layer_device_placement(
    stats_by_key: dict[tuple[int, str, str], _WeightLoadCategoryStats],
) -> str:
    if not stats_by_key:
        return "[]"

    by_layer: dict[int, list[tuple[str, str, _WeightLoadCategoryStats]]] = {}
    for (layer_id, param_device, loaded_device), stats in stats_by_key.items():
        by_layer.setdefault(layer_id, []).append((param_device, loaded_device, stats))

    parts = []
    for layer_id in sorted(by_layer):
        device_parts = []
        for param_device, loaded_device, stats in sorted(
            by_layer[layer_id],
            key=lambda item: item[2].num_bytes,
            reverse=True,
        ):
            device_parts.append(
                f"{param_device}/{loaded_device}:count={stats.count},"
                f"bytes={_format_gib(stats.num_bytes)},s={stats.seconds:.2f}"
            )
        parts.append(f"{layer_id}:" + "|".join(device_parts))
    return "[" + ", ".join(parts) + "]"


# ============================================================================
# Attention Layer
# ============================================================================


class BailingMoeV2Attention(nn.Module):
    """Multi-head attention with GQA, partial RoPE, and QK normalization."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.use_qk_norm = config.use_qk_norm

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        validate_attention_tp_config(
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            tp_size=attn_tp_size,
            context=f"BailingMoeV2Attention(layer_id={layer_id})",
        )

        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_size = attn_tp_size
        self.num_heads_per_tp = self.num_heads // attn_tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // attn_tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        # QK normalization layers (per-head)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE - using partial rotary factor
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Radix attention for paged KV cache
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            1.0 / math.sqrt(self.head_dim),
            self.num_kv_heads_per_tp,
            layer_id=layer_id,
        )

    def forward_prepare(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV projection + QK norm + RoPE."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Reshape for multi-head
        q = q.view(-1, self.num_heads_per_tp, self.head_dim)
        k = k.view(-1, self.num_kv_heads_per_tp, self.head_dim)
        v = v.view(-1, self.num_kv_heads_per_tp, self.head_dim)

        # QK normalization
        if self.use_qk_norm:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)

        # Partial RoPE: only apply to first rotary_dim dimensions
        q_rot = q[..., : self.rotary_dim]
        q_pass = q[..., self.rotary_dim :]
        k_rot = k[..., : self.rotary_dim]
        k_pass = k[..., self.rotary_dim :]

        q_rot, k_rot = self.rotary_emb(forward_batch.positions, q_rot, k_rot)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        return q, k, v

    def forward_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Attention computation with paged KV cache."""
        attn_output = self.attn(q, k, v, forward_batch)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q, k, v = self.forward_prepare(hidden_states, forward_batch)
        attn_output = self.forward_core(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


# ============================================================================
# MLP (for dense layers and shared experts)
# ============================================================================


class BailingMoeV2MLP(nn.Module):
    """Standard SwiGLU MLP implemented with SGLang tensor parallel layers."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return hidden_states


# ============================================================================
# Sparse MoE Block
# ============================================================================


class BailingMoeV2SparseMoeBlock(nn.Module):
    """Sparse MoE with group-limited top-k routing and optional shared expert.

    Routing: group_limited_topk
      1. Divide 256 experts into n_group=8 groups of 32
      2. Score each group by sum of top-2 expert scores within group
      3. Select topk_group=4 groups
      4. From selected groups, pick top num_experts_per_tok=8 experts
      5. Apply sigmoid gating with routed_scaling_factor=2.5
    """

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = get_tensor_model_parallel_world_size()

        # Gate: linear projection for router scores
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)

        # Expert bias for load balancing
        if config.use_expert_bias:
            self.expert_bias = nn.Parameter(
                torch.zeros(config.num_experts), requires_grad=False
            )
        else:
            self.expert_bias = None

        # Routed and shared experts produce TP-partial outputs, combine first,
        # then reduce once here or via LayerCommunicator all-reduce fusion.
        FusedMoE = get_moe_impl_class(quant_config)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("experts", prefix),
        )

        # Shared expert
        if config.num_shared_experts and config.num_shared_experts > 0:
            shared_intermediate = (
                config.moe_intermediate_size * config.num_shared_experts
            )
            if should_use_flashinfer_cutlass_moe_fp4_allgather():
                shared_tp_rank, shared_tp_size = 0, 1
            else:
                shared_tp_rank, shared_tp_size = None, None
            self.shared_experts = BailingMoeV2MLP(
                config,
                shared_intermediate,
                quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
            )
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        del forward_batch
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_input = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        # Router scores via sigmoid (not softmax like standard MoE)
        router_logits, _ = self.gate(hidden_states)
        router_logits = router_logits.float()
        scores = torch.sigmoid(router_logits)

        # Add expert bias for load balancing
        if self.expert_bias is not None:
            scores_for_routing = scores + self.expert_bias
        else:
            scores_for_routing = scores

        # Group-limited top-k selection
        topk_weights, topk_ids = self._group_limited_topk(scores_for_routing)

        # Gather actual scores (without bias) for the selected experts
        topk_weights = torch.gather(scores, dim=1, index=topk_ids)

        # Normalize and scale
        if self.num_experts_per_tok > 1:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * self.routed_scaling_factor

        # FusedMoE forward — wrap in StandardTopKOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )
        routed_output = self.experts(hidden_states, topk_output)

        # Add shared expert output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(shared_input)
            final_hidden_states = routed_output + shared_output
        else:
            final_hidden_states = routed_output

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def _group_limited_topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Group-limited top-k expert selection.

        1. Reshape scores to [tokens, n_group, experts_per_group]
        2. Score each group by sum of top-2 within group
        3. Select top topk_group groups
        4. Mask non-selected groups to -inf
        5. Select top num_experts_per_tok from unmasked experts
        """
        num_tokens = scores.shape[0]
        experts_per_group = self.num_experts // self.n_group

        # Group scores: sum of top-2 experts per group
        group_scores = (
            scores.view(num_tokens, self.n_group, experts_per_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        # Select top groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Expand group mask to expert-level
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, experts_per_group)
            .reshape(num_tokens, -1)
        )

        # Mask and select top-k
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(
            masked_scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )

        return topk_weights, topk_ids


# ============================================================================
# Decoder Layer
# ============================================================================


class BailingMoeV2DecoderLayer(nn.Module):
    """Single transformer decoder layer with attention + MoE/MLP."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.is_dense = layer_id < config.first_k_dense_replace

        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BailingMoeV2Attention(
            config,
            layer_id,
            quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        # FFN: dense MLP for first_k_dense_replace layers, MoE for rest
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.is_dense:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = BailingMoeV2MLP(
                config,
                config.intermediate_size,
                quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )
        else:
            self.mlp = BailingMoeV2SparseMoeBlock(
                config,
                layer_id,
                quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        is_layer_sparse = not self.is_dense
        is_previous_layer_sparse = layer_id - 1 >= config.first_k_dense_replace
        is_next_layer_sparse = layer_id + 1 >= config.first_k_dense_replace

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=None,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(hidden_states, forward_batch)

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states,
            forward_batch=forward_batch,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states,
                residual,
                forward_batch,
            )

        return hidden_states, residual


# ============================================================================
# Full Model
# ============================================================================


class BailingMoeV2TextModel(nn.Module):
    """BailingMoeV2 text model body (no LM head)."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix="": BailingMoeV2DecoderLayer(
                config, idx, quant_config, prefix=prefix
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, forward_batch, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights with prefix-based selection and MoE mapping."""
        text_loader_start_s = time.perf_counter()

        from sglang_omni.models.qwen3_omni.components.thinker_model import (
            extract_fused_experts,
        )

        # Fused QKV: attention q/k/v -> qkv_proj
        _attn_fused_map = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
        }

        params_dict = dict(self.named_parameters())
        _loaded_weight_count = 0
        _skipped_weight_count = 0
        _unmatched_weight_names: list[str] = []
        _gate_up_fused_shards: dict[str, set[int]] = {}
        _attn_qkv_stats = _WeightLoadCategoryStats()
        _moe_stats = _WeightLoadCategoryStats()
        _gate_up_stats = _WeightLoadCategoryStats()
        _direct_stats = _WeightLoadCategoryStats()
        _unmatched_stats = _WeightLoadCategoryStats()
        _moe_detail_enabled = _env_flag("SGLANG_OMNI_MING_MOE_PROFILE_DETAIL")
        _moe_shard_stats: dict[str, _WeightLoadCategoryStats] = {}
        _moe_layer_stats: dict[int, _WeightLoadCategoryStats] = {}
        _moe_layer_shard_stats: dict[tuple[int, str], _WeightLoadCategoryStats] = {}
        _moe_device_stats: dict[tuple[str, str], _WeightLoadCategoryStats] = {}
        _moe_layer_device_stats: dict[
            tuple[int, str, str], _WeightLoadCategoryStats
        ] = {}
        _moe_layer_device_shard_stats: dict[
            tuple[int, str, str, str], _WeightLoadCategoryStats
        ] = {}

        for name, loaded_weight in weights:
            # Strip common prefixes from Ming checkpoint
            for prefix in ("model.model.", "model.", "thinker.model.", "thinker."):
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break

            # 0. Remap checkpoint naming conventions to our model's naming
            # gate.expert_bias -> expert_bias (MoE routing bias)
            if ".mlp.gate.expert_bias" in name:
                name = name.replace(".mlp.gate.expert_bias", ".mlp.expert_bias")
            # word_embeddings -> embed_tokens
            if name == "word_embeddings.weight":
                name = "embed_tokens.weight"

            # 1a. Handle checkpoint naming: attention.X -> self_attn.Y
            # Ming checkpoint uses "attention.query_key_value" / "attention.dense"
            # Our model uses "self_attn.qkv_proj" / "self_attn.o_proj"
            if ".attention." in name:
                name = name.replace(
                    ".attention.query_key_value.", ".self_attn.qkv_proj."
                )
                name = name.replace(".attention.dense.", ".self_attn.o_proj.")
                name = name.replace(".attention.q_norm.", ".self_attn.q_norm.")
                name = name.replace(".attention.k_norm.", ".self_attn.k_norm.")

            # 1b. Handle separate q/k/v -> fused qkv_proj (if checkpoint has them)
            matched_attn = False
            for shard_name, (fused_name, shard_id) in _attn_fused_map.items():
                if shard_name in name and "self_attn" in name:
                    fused_key = name.replace(shard_name, fused_name)
                    if fused_key in params_dict:
                        param = params_dict[fused_key]
                        load_start_s = time.perf_counter()
                        param.weight_loader(param, loaded_weight, shard_id)
                        _attn_qkv_stats.add(
                            loaded_weight, time.perf_counter() - load_start_s
                        )
                        _loaded_weight_count += 1
                        matched_attn = True
                        break
            if matched_attn:
                continue

            # 2. Handle MoE expert weights via FusedMoE weight_loader
            if ".mlp.experts." in name:
                res = extract_fused_experts(
                    name=name,
                    ckpt_gate_proj_name="gate_proj",
                    ckpt_down_proj_name="down_proj",
                    ckpt_up_proj_name="up_proj",
                    num_experts=self.config.num_experts,
                )
                if res:
                    param_name, weight_name, expert_id, shard_id = res
                    # extract_fused_experts returns param_name like "experts.w13_"
                    # and weight_name like "experts.42.gate_proj"
                    # Checkpoint name: "layers.X.mlp.experts.42.gate_proj.weight"
                    # FusedMoE param:  "layers.X.mlp.experts.w13_weight"
                    # Replace "experts.42.gate_proj.weight" -> "experts.w13_weight"
                    fused_key = name.replace(
                        weight_name + ".weight", param_name + "weight"
                    )
                    if fused_key in params_dict:
                        param = params_dict[fused_key]
                        load_start_s = time.perf_counter()
                        param.weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        elapsed_s = time.perf_counter() - load_start_s
                        _moe_stats.add(loaded_weight, elapsed_s)
                        if _moe_detail_enabled:
                            shard_key = str(shard_id)
                            param_device = _device_label(param)
                            loaded_device = _device_label(loaded_weight)
                            _moe_shard_stats.setdefault(
                                shard_key, _WeightLoadCategoryStats()
                            ).add(loaded_weight, elapsed_s)
                            _moe_device_stats.setdefault(
                                (param_device, loaded_device),
                                _WeightLoadCategoryStats(),
                            ).add(loaded_weight, elapsed_s)
                            layer_id = _extract_moe_layer_id(name)
                            if layer_id is not None:
                                _moe_layer_stats.setdefault(
                                    layer_id, _WeightLoadCategoryStats()
                                ).add(loaded_weight, elapsed_s)
                                _moe_layer_shard_stats.setdefault(
                                    (layer_id, shard_key),
                                    _WeightLoadCategoryStats(),
                                ).add(loaded_weight, elapsed_s)
                                _moe_layer_device_stats.setdefault(
                                    (
                                        layer_id,
                                        param_device,
                                        loaded_device,
                                    ),
                                    _WeightLoadCategoryStats(),
                                ).add(loaded_weight, elapsed_s)
                                _moe_layer_device_shard_stats.setdefault(
                                    (
                                        layer_id,
                                        shard_key,
                                        param_device,
                                        loaded_device,
                                    ),
                                    _WeightLoadCategoryStats(),
                                ).add(loaded_weight, elapsed_s)
                        _loaded_weight_count += 1
                        continue

            # 3. Handle gate/up -> fused gate_up_proj
            # Applies to both shared_experts and dense MLP (layer 0)
            # Ming ckpt: *.gate_proj.weight / *.up_proj.weight
            # Our model: *.gate_up_proj.weight (MergedColumnParallelLinear)
            matched_mlp_gate_up = False
            for weight_name, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
                if weight_name not in name:
                    continue
                fused_key = name.replace(weight_name, ".gate_up_proj.")
                if fused_key in params_dict:
                    param = params_dict[fused_key]
                    load_start_s = time.perf_counter()
                    param.weight_loader(param, loaded_weight, shard_id)
                    _gate_up_stats.add(
                        loaded_weight, time.perf_counter() - load_start_s
                    )
                    _loaded_weight_count += 1
                    _gate_up_fused_shards.setdefault(fused_key, set()).add(shard_id)
                    matched_mlp_gate_up = True
                    break
            if matched_mlp_gate_up:
                continue

            # 4. Handle shared expert gate_up_proj / down_proj directly
            # (already fused in checkpoint)
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                load_start_s = time.perf_counter()
                weight_loader(param, loaded_weight)
                _direct_stats.add(loaded_weight, time.perf_counter() - load_start_s)
                _loaded_weight_count += 1
                continue

            _skipped_weight_count += 1
            _unmatched_stats.add(loaded_weight)
            _unmatched_weight_names.append(name)

        incomplete_gate_up_pairs = {
            key: sorted({0, 1} - shards)
            for key, shards in _gate_up_fused_shards.items()
            if shards != {0, 1}
        }
        if incomplete_gate_up_pairs:
            raise ValueError(
                "Incomplete Ming gate/up fused weights: "
                f"missing_pairs={incomplete_gate_up_pairs}"
            )
        if _unmatched_weight_names:
            logger.warning(
                "Ming thinker text loader skipped unmatched weights: "
                "count=%d sample=%s",
                len(_unmatched_weight_names),
                _unmatched_weight_names[:10],
            )
        logger.info(
            "Ming thinker text loader summary: loaded=%d skipped=%d unmatched=%d",
            _loaded_weight_count,
            _skipped_weight_count,
            len(_unmatched_weight_names),
        )
        text_loader_total_s = time.perf_counter() - text_loader_start_s
        accounted_load_s = (
            _attn_qkv_stats.seconds
            + _moe_stats.seconds
            + _gate_up_stats.seconds
            + _direct_stats.seconds
        )
        logger.info(
            "Ming text weight profile: total_s=%.2f accounted_load_s=%.2f "
            "unaccounted_s=%.2f loaded=%d skipped=%d unmatched=%d "
            "attn_qkv_count=%d attn_qkv_bytes=%s attn_qkv_s=%.2f "
            "moe_count=%d moe_bytes=%s moe_s=%.2f "
            "gate_up_count=%d gate_up_bytes=%s gate_up_s=%.2f "
            "direct_count=%d direct_bytes=%s direct_s=%.2f "
            "unmatched_bytes=%s",
            text_loader_total_s,
            accounted_load_s,
            max(0.0, text_loader_total_s - accounted_load_s),
            _loaded_weight_count,
            _skipped_weight_count,
            len(_unmatched_weight_names),
            _attn_qkv_stats.count,
            _format_gib(_attn_qkv_stats.num_bytes),
            _attn_qkv_stats.seconds,
            _moe_stats.count,
            _format_gib(_moe_stats.num_bytes),
            _moe_stats.seconds,
            _gate_up_stats.count,
            _format_gib(_gate_up_stats.num_bytes),
            _gate_up_stats.seconds,
            _direct_stats.count,
            _format_gib(_direct_stats.num_bytes),
            _direct_stats.seconds,
            _format_gib(_unmatched_stats.num_bytes),
        )
        if _moe_detail_enabled:
            w1_stats = _moe_shard_stats.get("w1", _WeightLoadCategoryStats())
            w2_stats = _moe_shard_stats.get("w2", _WeightLoadCategoryStats())
            w3_stats = _moe_shard_stats.get("w3", _WeightLoadCategoryStats())
            logger.info(
                "Ming MoE detail profile: total_count=%d total_bytes=%s "
                "total_s=%.2f avg_ms_per_tensor=%.2f throughput_gib_s=%.2f "
                "w1_count=%d w1_bytes=%s w1_s=%.2f w1_avg_ms=%.2f "
                "w1_gib_s=%.2f w2_count=%d w2_bytes=%s w2_s=%.2f "
                "w2_avg_ms=%.2f w2_gib_s=%.2f w3_count=%d w3_bytes=%s "
                "w3_s=%.2f w3_avg_ms=%.2f w3_gib_s=%.2f "
                "top_layers=%s top_layer_shards=%s",
                _moe_stats.count,
                _format_gib(_moe_stats.num_bytes),
                _moe_stats.seconds,
                _avg_ms_per_tensor(_moe_stats),
                _gib_per_second(_moe_stats),
                w1_stats.count,
                _format_gib(w1_stats.num_bytes),
                w1_stats.seconds,
                _avg_ms_per_tensor(w1_stats),
                _gib_per_second(w1_stats),
                w2_stats.count,
                _format_gib(w2_stats.num_bytes),
                w2_stats.seconds,
                _avg_ms_per_tensor(w2_stats),
                _gib_per_second(w2_stats),
                w3_stats.count,
                _format_gib(w3_stats.num_bytes),
                w3_stats.seconds,
                _avg_ms_per_tensor(w3_stats),
                _gib_per_second(w3_stats),
                _format_top_weight_load_buckets(_moe_layer_stats, limit=5),
                _format_top_weight_load_buckets(_moe_layer_shard_stats, limit=10),
            )
            logger.info(
                "Ming MoE device detail profile: device_buckets=%s "
                "top_layer_device_shards=%s",
                _format_top_weight_load_buckets(_moe_device_stats, limit=10),
                _format_top_weight_load_buckets(
                    _moe_layer_device_shard_stats, limit=10
                ),
            )
            logger.info(
                "Ming MoE layer device placement profile: layers=%s",
                _format_moe_layer_device_placement(_moe_layer_device_stats),
            )


# ============================================================================
# ForCausalLM Wrapper (top-level SGLang model class)
# ============================================================================


class BailingMoeV2ForCausalLM(nn.Module):
    """Top-level SGLang model class for BailingMoeV2.

    Wraps BailingMoeV2TextModel with LM head and LogitsProcessor.

    SGLang runtime expects:
      - model.model.embed_tokens  (embedding table)
      - model.model(...)          (text body forward)
      - model.lm_head             (output projection)
      - model.logits_processor    (logits post-processing)

    The config passed by SGLang is the top-level BailingMM2Config (from
    AutoConfig). This class extracts llm_config for model construction
    and patches token IDs on the HF config for the runtime's multimodal
    embedding injection.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # Keep the original HF config reference so SGLang runtime can read
        # patched attributes (audio_token_id, etc.) from the same object.
        self.config = config

        # Extract LLM sub-config (Ming's BailingMM2Config has .llm_config)
        llm_cfg = getattr(config, "llm_config", config)
        adapted = (
            BailingMoeV2Config(
                **(llm_cfg.to_dict() if hasattr(llm_cfg, "to_dict") else {}),
            )
            if not isinstance(llm_cfg, BailingMoeV2Config)
            else llm_cfg
        )

        # Build model body
        self.model = BailingMoeV2TextModel(adapted, quant_config)

        # Build LM head
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            adapted.vocab_size,
            adapted.hidden_size,
            quant_config=quant_config,
        )

        # Build logits processor
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(adapted)

        # ------------------------------------------------------------------
        # Vision encoder + projector — NOT loaded here.
        # The pipeline's IMAGE_STAGE (MingImageEncoder) handles vision
        # encoding independently and injects pre-computed image_embeds
        # via SGLang's _inject_multimodal_embeds().  Loading a duplicate
        # copy here would waste ~1.2 GB of GPU memory.
        # Vision/projector weights are silently skipped in load_weights().
        # ------------------------------------------------------------------
        self.visual = None
        self.linear_proj = None

        # ------------------------------------------------------------------
        # Patch token IDs on the HF config for SGLang runtime's
        # _inject_multimodal_embeds() which reads config.audio_token_id etc.
        # This runs during model loading, BEFORE SGLangModelScheduler reads
        # the config, so the patched values will be visible.
        # ------------------------------------------------------------------
        self._patch_token_ids(config, llm_cfg)

    @staticmethod
    def _patch_token_ids(config: Any, llm_cfg: Any) -> None:
        """Set image/video/audio token IDs on the HF config."""
        if not hasattr(config, "image_token_id"):
            config.image_token_id = getattr(llm_cfg, "image_patch_token", None)
        if not hasattr(config, "video_token_id"):
            config.video_token_id = getattr(llm_cfg, "video_patch_token", None)
        if not hasattr(config, "audio_token_id"):
            # audio_patch_token is NOT in config.json — resolve from tokenizer
            model_path = getattr(config, "_name_or_path", None)
            if model_path:
                try:
                    from sglang_omni.models.ming_omni.components.common import (
                        load_ming_tokenizer,
                    )

                    tok = load_ming_tokenizer(model_path)
                    audio_id = tok.convert_tokens_to_ids("<audioPatch>")
                    # convert_tokens_to_ids returns the UNK id if not found
                    unk_id = getattr(tok, "unk_token_id", None)
                    if isinstance(audio_id, int) and audio_id != unk_id:
                        config.audio_token_id = audio_id
                    else:
                        config.audio_token_id = None
                        logger.warning(
                            "Could not resolve <audioPatch> token ID from %s",
                            model_path,
                        )
                except Exception:
                    config.audio_token_id = None
                    logger.warning(
                        "Failed to load tokenizer for audio_token_id resolution"
                    )
            else:
                config.audio_token_id = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights from Ming-Omni checkpoint.

        Routes weights to sub-modules:
        - lm_head.*       → self.lm_head
        - vision.*, linear_proj.* → skipped (vision handled by IMAGE_STAGE)
        - audio.*, linear_proj_audio.* → skipped (audio handled by AUDIO_STAGE)
        - everything else → self.model (BailingMoeV2TextModel)
        """
        top_level_start_s = time.perf_counter()
        model_weights = []
        _loaded_lm_head_count = 0
        _skipped_tower_count = 0
        lm_head_params = dict(self.lm_head.named_parameters())
        _lm_head_stats = _WeightLoadCategoryStats()
        _tower_skipped_stats = _WeightLoadCategoryStats()
        _text_candidate_stats = _WeightLoadCategoryStats()

        route_start_s = time.perf_counter()
        for name, tensor in weights:
            # Strip top-level "model." prefix from checkpoint names.
            # Checkpoint uses "model.vision.*", "model.linear_proj.*", etc.
            # but NOT "model.model.*" (that stripping is in TextModel).
            stripped = name
            if stripped.startswith("model.") and not stripped.startswith(
                "model.model."
            ):
                stripped = stripped[len("model.") :]

            # Route lm_head weights
            if stripped in ("lm_head.weight",) or name in ("model.lm_head.weight",):
                param = lm_head_params.get("weight")
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    load_start_s = time.perf_counter()
                    weight_loader(param, tensor)
                    _lm_head_stats.add(tensor, time.perf_counter() - load_start_s)
                    _loaded_lm_head_count += 1
                continue

            # Skip vision encoder + projector weights (handled by IMAGE_STAGE)
            if stripped.startswith("vision."):
                _tower_skipped_stats.add(tensor)
                _skipped_tower_count += 1
                continue
            if stripped.startswith("linear_proj.") and not stripped.startswith(
                "linear_proj_audio."
            ):
                _tower_skipped_stats.add(tensor)
                _skipped_tower_count += 1
                continue

            # Skip audio weights (handled by AUDIO_STAGE)
            if stripped.startswith("audio.") or stripped.startswith(
                "linear_proj_audio."
            ):
                _tower_skipped_stats.add(tensor)
                _skipped_tower_count += 1
                continue

            # Pass original name to text model (it does its own prefix stripping)
            _text_candidate_stats.add(tensor)
            model_weights.append((name, tensor))
        route_and_collect_s = time.perf_counter() - route_start_s

        # Load text model weights
        text_model_load_start_s = time.perf_counter()
        self.model.load_weights(iter(model_weights))
        text_model_load_s = time.perf_counter() - text_model_load_start_s
        logger.info(
            "Ming top-level loader summary: lm_head_loaded=%d tower_skipped=%d "
            "text_weight_candidates=%d",
            _loaded_lm_head_count,
            _skipped_tower_count,
            len(model_weights),
        )

        # Handle weight tying
        tie_word_embeddings_s = 0.0
        llm_cfg = getattr(self.config, "llm_config", self.config)
        if getattr(llm_cfg, "tie_word_embeddings", False):
            tie_start_s = time.perf_counter()
            lm_weight = lm_head_params.get("weight")
            if lm_weight is not None:
                lm_weight.data = self.model.embed_tokens.weight.data
            tie_word_embeddings_s = time.perf_counter() - tie_start_s
        logger.info(
            "Ming top-level weight profile: total_s=%.2f route_and_collect_s=%.2f "
            "text_model_load_s=%.2f tie_word_embeddings_s=%.2f "
            "lm_head_count=%d lm_head_bytes=%s lm_head_load_s=%.2f "
            "tower_skipped_count=%d tower_skipped_bytes=%s "
            "text_candidate_count=%d text_candidate_bytes=%s",
            time.perf_counter() - top_level_start_s,
            route_and_collect_s,
            text_model_load_s,
            tie_word_embeddings_s,
            _lm_head_stats.count,
            _format_gib(_lm_head_stats.num_bytes),
            _lm_head_stats.seconds,
            _tower_skipped_stats.count,
            _format_gib(_tower_skipped_stats.num_bytes),
            _text_candidate_stats.count,
            _format_gib(_text_candidate_stats.num_bytes),
        )
        if _env_flag("SGLANG_OMNI_MING_MOE_SYNC_AFTER_LOAD"):
            sync_start_s = time.perf_counter()
            sync_status = "cuda_unavailable"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                sync_status = "ok"
            logger.info(
                "Ming top-level post-load sync profile: sync_s=%.2f status=%s",
                time.perf_counter() - sync_start_s,
                sync_status,
            )
        if _env_flag("SGLANG_OMNI_MING_MODEL_WEIGHTS_RELEASE_PROFILE"):
            release_count = len(model_weights)
            release_bytes = _text_candidate_stats.num_bytes
            release_start_s = time.perf_counter()
            model_weights.clear()
            logger.info(
                "Ming top-level model_weights release profile: clear_s=%.2f "
                "count=%d bytes=%s remaining_count=%d",
                time.perf_counter() - release_start_s,
                release_count,
                _format_gib(release_bytes),
                len(model_weights),
            )
