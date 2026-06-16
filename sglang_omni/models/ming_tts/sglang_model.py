# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Ming-Omni-TTS 16.8B AR model wrapper."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import nn

from sglang_omni.models.ming_tts.weight_loading import (
    MING_TTS_LM_HEAD_SKIP_REASON,
    MING_TTS_ROTARY_BUFFER_SKIP_REASON,
    OWNER_AR_MODEL,
    OWNER_AUDIO_VAE,
    OWNER_INTENTIONAL_SKIP,
    OWNER_TTS_HEADS,
    OWNER_UNKNOWN,
    MingTTSWeightReport,
    assert_ming_tts_weight_coverage,
    classify_ming_tts_weight,
)
from sglang_omni.models.weight_loader import default_weight_loader
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    MRotaryEmbedding,
    QKVParallelLinear,
    QuantizationConfig,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
    StandardTopKOutput,
    VocabParallelEmbedding,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_moe_impl_class,
    get_rope,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang_omni.vendor.sglang.server_args import get_global_server_args
from sglang_omni.vendor.sglang.utils import add_prefix

logger = logging.getLogger(__name__)


class MingBailingMoeAttention(nn.Module):
    """BailingMoe attention with SGLang-managed paged KV cache."""

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        if self.num_heads % attn_tp_size != 0:
            raise ValueError(
                "Ming BailingMoe attention heads must be divisible by "
                f"attention TP size, got heads={self.num_heads}, "
                f"tp_size={attn_tp_size}"
            )
        if self.num_kv_heads >= attn_tp_size:
            if self.num_kv_heads % attn_tp_size != 0:
                raise ValueError(
                    "Ming BailingMoe KV heads must be divisible by attention "
                    f"TP size, got kv_heads={self.num_kv_heads}, "
                    f"tp_size={attn_tp_size}"
                )
        elif attn_tp_size % self.num_kv_heads != 0:
            raise ValueError(
                "Ming BailingMoe KV heads must either divide or be divided by "
                f"attention TP size, got kv_heads={self.num_kv_heads}, "
                f"tp_size={attn_tp_size}"
            )

        self.num_heads_per_tp = self.num_heads // attn_tp_size
        self.num_kv_heads_per_tp = max(1, self.num_kv_heads // attn_tp_size)
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=bool(getattr(config, "use_qkv_bias", False)),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("query_key_value", prefix),
        )
        self.dense = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "use_bias", False)),
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("dense", prefix),
        )

        rope_scaling = getattr(config, "runtime_rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=float(config.rope_theta),
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads_per_tp,
            self.head_dim,
            1.0 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_kv_heads_per_tp,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def _prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        if isinstance(self.rotary_emb, MRotaryEmbedding):
            if positions.dim() == 1:
                return positions.unsqueeze(0).expand(3, -1)
            return positions
        if positions.dim() == 2:
            return positions[0]
        return positions

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(self._prepare_positions(positions), q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.dense(attn_output)
        return output


class MingBailingMoeMLP(nn.Module):
    """BailingMoe SwiGLU MLP using SGLang tensor-parallel layers."""

    def __init__(
        self,
        config: Any,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            int(config.hidden_size),
            [int(intermediate_size), int(intermediate_size)],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            int(intermediate_size),
            int(config.hidden_size),
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class MingBailingMoeGate(nn.Module):
    """Replicated BailingMoe router weight with official softmax top-k semantics."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(int(config.num_experts), int(config.hidden_size))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )


class MingBailingMoeSparseMoeBlock(nn.Module):
    """BailingMoe sparse block: official routing, SGLang FusedMoE execution."""

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_experts = int(config.num_experts)
        self.num_experts_per_tok = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.multi_gate = bool(getattr(config, "multi_gate", False))
        self.tp_size = get_tensor_model_parallel_world_size()

        self.gate = MingBailingMoeGate(config)
        if self.multi_gate:
            self.image_gate = MingBailingMoeGate(config)
            self.audio_gate = MingBailingMoeGate(config)

        FusedMoE = get_moe_impl_class(quant_config)
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("experts", prefix),
        )

        num_shared_experts = int(getattr(config, "num_shared_experts", 0) or 0)
        if num_shared_experts > 0:
            self.shared_experts = MingBailingMoeMLP(
                config,
                int(config.moe_intermediate_size) * num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    def _topk(
        self,
        hidden_states: torch.Tensor,
        gate: MingBailingMoeGate,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = gate(hidden_states).float()
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(
            scores,
            k=self.num_experts_per_tok,
            dim=-1,
        )
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_ids, topk_weights, router_logits

    def _route(
        self,
        hidden_states: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> StandardTopKOutput:
        topk_ids, topk_weights, router_logits = self._topk(hidden_states, self.gate)
        if not self.multi_gate:
            return StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )

        for mask, gate in (
            (image_mask, getattr(self, "image_gate", None)),
            (audio_mask, getattr(self, "audio_gate", None)),
        ):
            if mask is None or gate is None:
                continue
            flat_mask = mask.reshape(-1, 1).to(device=topk_ids.device, dtype=torch.bool)
            alt_ids, alt_weights, alt_logits = self._topk(hidden_states, gate)
            topk_ids = torch.where(flat_mask, alt_ids, topk_ids)
            topk_weights = torch.where(flat_mask, alt_weights, topk_weights)
            router_logits = torch.where(flat_mask, alt_logits, router_logits)

        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        shared_input = (
            hidden_states.clone() if self.shared_experts is not None else hidden_states
        )

        topk_output = self._route(hidden_states, image_mask, audio_mask)
        hidden_states = self.experts(hidden_states, topk_output)
        if self.shared_experts is not None:
            hidden_states = hidden_states + self.shared_experts(shared_input)

        if self.tp_size > 1 and not should_use_flashinfer_cutlass_moe_fp4_allgather():
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return hidden_states.view(original_shape)


class MingBailingMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attention = MingBailingMoeAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        if getattr(config, "num_experts", None) is not None and layer_id >= int(
            getattr(config, "first_k_dense_replace", 0) or 0
        ):
            self.mlp = MingBailingMoeSparseMoeBlock(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = MingBailingMoeMLP(
                config=config,
                intermediate_size=int(config.intermediate_size),
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )
        self.post_attention_layernorm = RMSNorm(
            int(config.hidden_size),
            eps=float(config.rms_norm_eps),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MingBailingMoeTextModel(nn.Module):
    """BailingMoe decoder body with SGLang KV-cache ownership."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("word_embeddings", prefix),
        )
        self.layers = nn.ModuleList(
            [
                MingBailingMoeDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.start_layer = 0
        self.end_layer = int(config.num_hidden_layers)
        self.norm = RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))

    def get_input_embeddings(self) -> nn.Module:
        return self.word_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer_id in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_id](
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MingTTSSGLangModel(nn.Module):
    """Ming-Omni-TTS AR backbone plus weighted TTS heads."""

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".query_key_value.",
        ".dense.",
    ]

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.llm_config = getattr(config, "llm_config", config)
        self.quant_config = quant_config
        self.model = MingBailingMoeTextModel(
            self.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.hidden_size = int(self.llm_config.hidden_size)
        self.vocab_size = int(self.llm_config.vocab_size)

        max_batch_size = 1
        try:
            max_batch_size = int(get_global_server_args().max_running_requests)
        except Exception:
            logger.debug("Falling back to one Ming TTS decode row")

        weight = self.model.word_embeddings.weight
        self._decode_input_embedding = nn.Embedding(
            max_batch_size,
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)

        if not hasattr(self.config, "audio_tokenizer_config"):
            raise ValueError(
                "MingTTSSGLangModel requires the top-level bailingmm config, "
                "not only llm_config, because FlowLoss/Aggregator shapes live "
                "in audio_tokenizer_config, ditar_config, and aggregator_config."
            )

        try:
            from sglang_omni.models.ming_tts.fm.dit import Aggregator
            from sglang_omni.models.ming_tts.fm.flowloss import FlowLoss
        except ImportError as exc:
            raise ImportError(
                "MingTTSSGLangModel requires sglang_omni.models.ming_tts.fm "
                "for Aggregator and FlowLoss, including its x-transformers "
                "dependency. Install project dependencies or run "
                "`pip install x-transformers` before loading this model."
            ) from exc

        audio_config = self.config.audio_tokenizer_config
        self.latent_dim = int(audio_config.enc_kwargs["latent_dim"])
        self.patch_size = int(self.config.ditar_config["patch_size"])
        self.history_patch_size = int(
            self.config.ditar_config.get("history_patch_size", self.patch_size)
        )
        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=self.hidden_size,
            **self.config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim,
            llm_cond_dim=self.hidden_size,
            **self.config.ditar_config,
        )
        self.stop_head = nn.Linear(self.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.hidden_size, bias=True)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del pp_proxy_tensors, input_embeds_are_projected

        if input_embeds is None:
            input_embeds = getattr(forward_batch, "input_embeds", None)

        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and bool(forward_mode.is_decode())
        )
        if input_embeds is None and is_decode:
            input_embeds = self._decode_input_embedding(input_ids)
            input_ids = None

        mrope_positions = getattr(forward_batch, "mrope_positions", None)
        if mrope_positions is not None:
            positions = mrope_positions

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        if is_extend:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            if extend_seq_lens is None:
                sample_hidden_states = hidden_states[-1:].contiguous()
            else:
                last_index = (
                    torch.cumsum(
                        extend_seq_lens.to(
                            device=hidden_states.device,
                            dtype=torch.long,
                        ),
                        dim=0,
                    )
                    - 1
                )
                sample_hidden_states = hidden_states[last_index]
        else:
            sample_hidden_states = hidden_states
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        report = MingTTSWeightReport(
            loaded={OWNER_AR_MODEL: 0, OWNER_TTS_HEADS: 0},
            skipped={MING_TTS_LM_HEAD_SKIP_REASON: []},
            deferred={OWNER_AUDIO_VAE: []},
        )
        loaded_param_names: set[str] = set()
        num_experts = int(self.llm_config.num_experts)

        def load_param(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        def load_gate_up_weight(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> tuple[str, str] | None:
            if ".experts." in name:
                return None
            for weight_name, shard_id in ((".gate_proj.", 0), (".up_proj.", 1)):
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, ".gate_up_proj.")
                param = params_dict.get(mapped_name)
                if param is None:
                    return None
                param.weight_loader(param, loaded_weight, shard_id)
                return mapped_name, str(shard_id)
            return None

        def load_fused_expert_weight(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> tuple[str, str] | None:
            if ".experts." not in name:
                return None
            match = re.search(r"experts\.(\d+)\.(gate_proj|down_proj|up_proj)", name)
            if match is None:
                return None

            expert_id = int(match.group(1))
            if expert_id >= num_experts:
                return None

            weight_type = match.group(2)
            param_name = "experts.w2_weight"
            shard_id = "w2"
            if weight_type == "gate_proj":
                param_name = "experts.w13_weight"
                shard_id = "w1"
            elif weight_type == "up_proj":
                param_name = "experts.w13_weight"
                shard_id = "w3"

            weight_name = f"experts.{expert_id}.{weight_type}.weight"
            if weight_name not in name:
                return None
            mapped_name = name.replace(weight_name, param_name)
            param = params_dict.get(mapped_name)
            if param is None:
                return None
            param.weight_loader(
                param,
                loaded_weight,
                mapped_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return mapped_name, f"{shard_id}:{expert_id}"

        for name in params_dict:
            if name.endswith("gate_up_proj.weight"):
                report.add_required_shards(name, ("0", "1"))
            elif name.endswith("experts.w13_weight"):
                shards = []
                for expert_id in range(num_experts):
                    shards.append(f"w1:{expert_id}")
                    shards.append(f"w3:{expert_id}")
                report.add_required_shards(name, shards)
            elif name.endswith("experts.w2_weight"):
                report.add_required_shards(
                    name,
                    [f"w2:{expert_id}" for expert_id in range(num_experts)],
                )

        for original_name, loaded_weight in weights:
            owner = classify_ming_tts_weight(original_name)
            if owner == OWNER_INTENTIONAL_SKIP:
                report.skipped.setdefault(
                    MING_TTS_LM_HEAD_SKIP_REASON,
                    [],
                ).append(original_name)
                continue
            if owner == OWNER_AUDIO_VAE:
                report.deferred.setdefault(OWNER_AUDIO_VAE, []).append(original_name)
                continue
            if owner == OWNER_UNKNOWN:
                report.leftovers.append(original_name)
                continue
            if "rotary_emb." in original_name or original_name.endswith(
                ".rotary_embed.inv_freq"
            ):
                report.skipped.setdefault(
                    MING_TTS_ROTARY_BUFFER_SKIP_REASON,
                    [],
                ).append(original_name)
                continue

            name = original_name
            if name.startswith("model.model."):
                name = "model." + name[len("model.model.") :]
            elif name.startswith(("word_embeddings.", "layers.", "norm.")):
                name = "model." + name

            packed = load_fused_expert_weight(name, loaded_weight)
            if packed is not None:
                target_param, shard_id = packed
                loaded_param_names.add(target_param)
                report.add_loaded(owner, original_name, target_param=target_param)
                report.add_loaded_shard(target_param, shard_id)
                continue

            packed = load_gate_up_weight(name, loaded_weight)
            if packed is not None:
                target_param, shard_id = packed
                loaded_param_names.add(target_param)
                report.add_loaded(owner, original_name, target_param=target_param)
                report.add_loaded_shard(target_param, shard_id)
                continue

            param = params_dict.get(name)
            if param is not None:
                load_param(param, loaded_weight)
                loaded_param_names.add(name)
                report.add_loaded(owner, original_name, target_param=name)
            else:
                report.leftovers.append(original_name)

        runtime_params = {"_decode_input_embedding.weight"}
        missing_params = sorted(set(params_dict) - loaded_param_names - runtime_params)
        if missing_params:
            report.missing["model_params"] = missing_params

        assert_ming_tts_weight_coverage(report)
        self._weight_load_report = report
        logger.info("%s", report.summary())


EntryClass = MingTTSSGLangModel
