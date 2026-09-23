# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Ming-Omni talker kernel integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from sglang_omni.models import weight_loader
from sglang_omni.models.ming_omni.talker import (
    modeling_ming_omni_talker as talker_model,
)
from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
)
from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT
from sglang_omni.models.ming_omni.talker.talker_module.execution import (
    TalkerExecutionConfig,
)
from sglang_omni.models.ming_omni.talker.talker_module.modules import (
    Attention,
    PackedQKVLinear,
)
from sglang_omni.models.ming_omni.talker.talker_module.rotary import (
    CachedRotaryEmbedding,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_talker_loader_selects_kernels_only_for_cuda(monkeypatch, device: str) -> None:
    config = MingOmniTalkerConfig(
        flowmodel={"attn_backend": "torch"},
        aggregator={"attn_backend": "torch"},
        patch_size=4,
        history_patch_size=3,
    )
    monkeypatch.setattr(
        MingOmniTalkerConfig,
        "from_pretrained_dir",
        classmethod(lambda cls, model_path: config),
    )
    monkeypatch.setattr(
        weight_loader, "load_weights_by_prefix", lambda *args, **kwargs: {}
    )
    kernel = Mock()
    provider = Mock(return_value=kernel)
    monkeypatch.setattr(
        talker_model,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, get_joint_rope_inplace_kernel=provider),
    )

    class RecordingTalker(talker_model.MingOmniTalker):
        def __init__(
            self,
            model_config: MingOmniTalkerConfig,
            *,
            dit_execution_config: TalkerExecutionConfig | None,
            aggregator_execution_config: TalkerExecutionConfig | None,
        ) -> None:
            nn.Module.__init__(self)
            self.model_config = model_config
            self.dit_execution_config = dit_execution_config
            self.aggregator_execution_config = aggregator_execution_config

        def load_weights(self, weights) -> None:
            assert list(weights) == []

        def to(self, *, device: torch.device, dtype: torch.dtype) -> RecordingTalker:
            assert device.type in ("cpu", "cuda")
            assert dtype == torch.bfloat16
            return self

        def eval(self) -> RecordingTalker:
            return self

    model = RecordingTalker.from_pretrained("unused", device=device)
    assert model.model_config is config
    if device == "cpu":
        assert model.dit_execution_config is None
        assert model.aggregator_execution_config is None
        provider.assert_not_called()
    else:
        dit_config = model.dit_execution_config
        aggregator_config = model.aggregator_execution_config
        assert dit_config.rope_kernel is kernel
        assert dit_config.rope_seq_len == 8
        assert dit_config.rope_max_batch_size == 2
        assert aggregator_config.rope_kernel is kernel
        assert aggregator_config.rope_seq_len == 5
        assert aggregator_config.rope_max_batch_size == talker_model._MAX_CACHE_LEN
        assert dit_config.qkv_layer is PackedQKVLinear
        assert aggregator_config.qkv_layer is PackedQKVLinear
        assert dit_config.norm_layer is aggregator_config.norm_layer
        assert dit_config.norm_layer.keywords == {"cast_x_before_out_mul": True}
        provider.assert_called_once_with()


def test_packed_qkv_loads_checkpoint_shards_and_matches_native_attention() -> None:
    class TinyTalker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aggregator = nn.Module()
            self.aggregator.attn = Attention(
                dim=4,
                heads=2,
                dim_head=2,
                attn_backend="torch",
                qkv_layer=PackedQKVLinear,
            )

    packed_talker = TinyTalker()
    native_attention = Attention(dim=4, heads=2, dim_head=2, attn_backend="torch")
    weights: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        packed_talker.aggregator.attn.to_out.load_state_dict(
            native_attention.to_out.state_dict()
        )
        for index, shard in enumerate(("q", "k", "v"), start=1):
            native_projection = getattr(native_attention, f"to_{shard}")
            weight = torch.arange(16, dtype=torch.float32).reshape(4, 4) * index / 16
            bias = torch.arange(4, dtype=torch.float32) + index
            native_projection.weight.copy_(weight)
            native_projection.bias.copy_(bias)
            weights.extend(
                (
                    (f"aggregator.attn.to_{shard}.weight", weight),
                    (f"aggregator.attn.to_{shard}.bias", bias),
                )
            )
        talker_model.MingOmniTalker.load_weights(packed_talker, weights)
        inputs = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
        torch.testing.assert_close(
            packed_talker.aggregator.attn(inputs), native_attention(inputs)
        )

    with pytest.raises(ValueError, match="to_qkv.bias.*k"):
        talker_model.MingOmniTalker.load_weights(
            TinyTalker(),
            [
                (name, value)
                for name, value in weights
                if name != "aggregator.attn.to_k.bias"
            ],
        )


def test_dit_forward_uses_configured_norm_packing_and_joint_rope() -> None:
    norm_calls: list[torch.Size] = []
    rope_positions: list[list[int]] = []

    class RecordingNorm(nn.Module):
        def __init__(self, dim: int, eps: float) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(dim, eps=eps)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            norm_calls.append(inputs.shape)
            return self.norm(inputs)

    def record_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None:
        assert query.shape == key.shape == (12, 2, 4)
        assert cache.dtype == torch.float32
        assert is_neox is False
        rope_positions.append(positions.tolist())

    dit = DiT(
        in_channels=4,
        hidden_size=8,
        depth=2,
        num_heads=2,
        llm_cond_dim=6,
        cfg_dropout_prob=0,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch",
            rope_kernel=record_rope,
            rope_seq_len=6,
            rope_max_batch_size=2,
            norm_layer=RecordingNorm,
            qkv_layer=PackedQKVLinear,
        ),
    ).eval()
    with torch.no_grad():
        output = dit.forward_with_cfg(
            torch.randn(1, 2, 4),
            torch.tensor(0.5),
            torch.randn(1, 1, 6),
            torch.randn(1, 3, 4),
        )

    assert output.shape == (2, 2, 4)
    assert len(norm_calls) == 5
    assert rope_positions == [list(range(6)) * 2] * 2
    assert all(isinstance(block.attn.to_qkv, PackedQKVLinear) for block in dit.blocks)


def test_reference_and_decode_share_cache_but_keep_resident_positions() -> None:
    rope_positions: list[list[int]] = []

    def record_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None:
        assert query.shape == key.shape
        assert cache.dtype == torch.float32
        assert is_neox is False
        rope_positions.append(positions.tolist())

    aggregator = Aggregator(
        in_channels=4,
        hidden_size=8,
        depth=2,
        num_heads=2,
        llm_input_dim=6,
        execution_config=TalkerExecutionConfig(
            attn_backend="torch",
            rope_kernel=record_rope,
            rope_seq_len=3,
            rope_max_batch_size=4,
            norm_layer=nn.RMSNorm,
            qkv_layer=PackedQKVLinear,
        ),
    ).eval()
    rotary = aggregator.rotary_embed
    assert isinstance(rotary, CachedRotaryEmbedding)
    rotary.to(dtype=torch.bfloat16)
    assert rotary.cos_sin_cache.dtype == torch.float32
    positions_pointer = rotary.positions.data_ptr()
    cache_pointer = rotary.cos_sin_cache.data_ptr()

    with torch.no_grad():
        reference_output = aggregator(torch.randn(4, 2, 4))
        decode_output = aggregator(torch.randn(1, 2, 4))

    assert reference_output.shape == (4, 1, 6)
    assert decode_output.shape == (1, 1, 6)
    assert rope_positions == [list(range(3)) * 4] * 2 + [list(range(3))] * 2
    assert rotary.positions.data_ptr() == positions_pointer
    assert rotary.cos_sin_cache.data_ptr() == cache_pointer
