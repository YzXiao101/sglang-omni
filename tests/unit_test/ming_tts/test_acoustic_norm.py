# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import partial

import pytest
import torch
from sglang.kernels import fused_op

from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT
from sglang_omni.models.ming_omni.talker.talker_module.execution import (
    TalkerExecutionConfig,
)
from sglang_omni.models.ming_omni.talker.talker_module.modules import (
    RMSNorm as MingRMSNorm,
)
from sglang_omni.models.ming_tts.sglang_model import RMSNorm as SGLangRMSNorm


def test_ming_tts_acoustic_rms_norm_preserves_bf16_semantics() -> None:
    hidden_size = 8
    eps = 1e-6
    legacy = MingRMSNorm(hidden_size, eps).to(dtype=torch.bfloat16)
    optimized = SGLangRMSNorm(
        hidden_size,
        eps=eps,
        cast_x_before_out_mul=True,
    ).to(dtype=torch.bfloat16)
    weight = torch.linspace(0.5, 1.5, hidden_size, dtype=torch.float32).to(
        torch.bfloat16
    )
    inputs = torch.linspace(-3.0, 3.0, 3 * 5 * hidden_size).reshape(3, 5, hidden_size)
    inputs = inputs.to(torch.bfloat16)

    with torch.no_grad():
        legacy.weight.copy_(weight)
        optimized.weight.copy_(weight)
        expected = legacy(inputs)
        actual = optimized.forward_native(inputs)

    assert type(optimized) is SGLangRMSNorm
    assert optimized.hidden_size == hidden_size
    assert optimized.variance_epsilon == eps
    assert optimized.cast_x_before_out_mul is True
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("component_name", ["aggregator", "dit"])
def test_acoustic_components_execute_injected_norms_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    component_name: str,
) -> None:
    monkeypatch.setattr(fused_op, "_platform_key", lambda: "cpu")
    common = dict(
        in_channels=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        qk_norm="rms_norm",
    )
    native_execution = TalkerExecutionConfig(attn_backend="torch")
    optimized_execution = TalkerExecutionConfig(
        attn_backend="torch",
        rms_norm_factory=partial(
            SGLangRMSNorm,
            cast_x_before_out_mul=True,
        ),
    )
    if component_name == "aggregator":
        native = Aggregator(
            **common,
            llm_input_dim=6,
            execution_config=native_execution,
        )
        optimized = Aggregator(
            **common,
            llm_input_dim=6,
            execution_config=optimized_execution,
        )
        forward_args = (torch.randn(2, 2, 4),)
        expected_shape = (2, 1, 6)
    else:
        native = DiT(
            **common,
            llm_cond_dim=6,
            cfg_dropout_prob=0,
            execution_config=native_execution,
        )
        optimized = DiT(
            **common,
            llm_cond_dim=6,
            cfg_dropout_prob=0,
            execution_config=optimized_execution,
        )
        forward_args = (
            torch.randn(2, 2, 4),
            torch.rand(2),
            torch.randn(2, 1, 6),
            torch.randn(2, 3, 4),
        )
        expected_shape = (2, 6, 4)

    native_hidden_norms = {
        "blocks.0.norm1": native.blocks[0].norm1,
        "blocks.0.norm2": native.blocks[0].norm2,
        "final_layer.norm_final": native.final_layer.norm_final,
    }
    optimized_hidden_norms = {
        "blocks.0.norm1": optimized.blocks[0].norm1,
        "blocks.0.norm2": optimized.blocks[0].norm2,
        "final_layer.norm_final": optimized.final_layer.norm_final,
    }
    assert all(type(norm) is MingRMSNorm for norm in native_hidden_norms.values())
    assert all(type(norm) is SGLangRMSNorm for norm in optimized_hidden_norms.values())
    assert all(
        norm.cast_x_before_out_mul is True for norm in optimized_hidden_norms.values()
    )
    assert type(optimized.blocks[0].attn.q_norm) is MingRMSNorm
    assert type(optimized.blocks[0].attn.k_norm) is MingRMSNorm

    native_state = native.state_dict()
    optimized_state = optimized.state_dict()
    assert native_state.keys() == optimized_state.keys()
    assert {name: value.shape for name, value in native_state.items()} == {
        name: value.shape for name, value in optimized_state.items()
    }
    with torch.no_grad():
        projection = native.final_layer.linear
        projection.weight.copy_(
            torch.linspace(-0.2, 0.2, projection.weight.numel()).reshape_as(
                projection.weight
            )
        )
        projection.bias.copy_(torch.linspace(-0.1, 0.1, projection.bias.numel()))
    optimized.load_state_dict(native.state_dict(), strict=True)
    native.eval()
    optimized.eval()

    calls: list[str] = []
    handles = [
        norm.register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.append(name)
        )
        for name, norm in optimized_hidden_norms.items()
    ]
    try:
        with torch.no_grad():
            expected = native(*forward_args)
            actual = optimized(*forward_args)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == list(optimized_hidden_norms)
    assert actual.shape == expected.shape == expected_shape
    assert actual.abs().sum() > 0
    torch.testing.assert_close(actual, expected)
