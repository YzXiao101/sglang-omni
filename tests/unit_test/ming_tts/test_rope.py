# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming_omni.talker.talker_module import modules
from sglang_omni.models.ming_omni.talker.talker_module.aggregator import Aggregator
from sglang_omni.models.ming_omni.talker.talker_module.dit import DiT
from sglang_omni.models.ming_omni.talker.talker_module.modules import Attention
from sglang_omni.models.ming_tts.rope import SGLKernelRoPEProvider


def _attention(*, rope_provider=None, **kwargs) -> Attention:
    config = {
        "dim": 128,
        "heads": 2,
        "dim_head": 64,
        "dropout": 0.0,
        "attn_backend": "torch",
        "attn_mask_enabled": False,
        "rope_provider": rope_provider,
    }
    config.update(kwargs)
    return Attention(**config).eval()


def _rope(sequence_length: int) -> tuple[torch.Tensor, float]:
    return torch.zeros((1, sequence_length, 64)), 1.0


def test_attention_provider_rejection_runs_the_unchanged_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Size, torch.Size]] = []

    def reject(query, key, rope):
        del rope
        calls.append((query.shape, key.shape))
        return None

    torch.manual_seed(0)
    native = _attention()
    candidate = _attention(rope_provider=reject)
    candidate.load_state_dict(native.state_dict())
    inputs = torch.randn((2, 3, 128))
    rope = _rope(sequence_length=3)

    with torch.no_grad():
        expected = native(inputs, rope=rope)

    native_apply = modules.apply_rotary_pos_emb
    native_rope_calls = 0

    def record_native_rope(*args, **kwargs):
        nonlocal native_rope_calls
        native_rope_calls += 1
        return native_apply(*args, **kwargs)

    monkeypatch.setattr(modules, "apply_rotary_pos_emb", record_native_rope)
    with torch.no_grad():
        actual = candidate(inputs, rope=rope)

    assert torch.equal(actual, expected)
    assert calls == [(torch.Size((2, 3, 2, 64)),) * 2]
    assert native_rope_calls == 2
    assert list(candidate.state_dict()) == list(native.state_dict())


def test_attention_provider_success_skips_native_rope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def provide(query, key, rope):
        nonlocal calls
        del rope
        calls += 1
        return query, key

    monkeypatch.setattr(
        modules,
        "apply_rotary_pos_emb",
        lambda *args, **kwargs: pytest.fail("native RoPE ran after provider success"),
    )
    attention = _attention(rope_provider=provide)

    with torch.no_grad():
        output = attention(torch.randn((2, 3, 128)), rope=_rope(3))

    assert output.shape == (2, 3, 128)
    assert calls == 1


def test_attention_propagates_provider_launch_failure_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LaunchFailure(RuntimeError):
        pass

    def fail_after_launch(query, key, rope):
        del query, key, rope
        raise LaunchFailure

    monkeypatch.setattr(
        modules,
        "apply_rotary_pos_emb",
        lambda *args, **kwargs: pytest.fail("native RoPE ran after launch failure"),
    )
    attention = _attention(rope_provider=fail_after_launch)

    with pytest.raises(LaunchFailure):
        attention(torch.randn((1, 2, 128)), rope=_rope(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"qk_norm": "rms_norm"}, id="qk-norm"),
        pytest.param({"pe_attn_head": 1}, id="partial-head"),
        pytest.param({"dim_head": 32}, id="head-dim"),
    ],
)
def test_attention_keeps_statically_incompatible_semantics_native(kwargs: dict) -> None:
    provider = lambda query, key, rope: (query, key)

    attention = _attention(rope_provider=provider, **kwargs)

    assert attention.rope_provider is None


def test_attention_keeps_non_torch_backend_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(modules, "is_flash_attn_available", lambda: True)
    provider = lambda query, key, rope: (query, key)

    attention = Attention(
        dim=128,
        heads=2,
        dim_head=64,
        attn_backend="flash_attn",
        rope_provider=provider,
    )

    assert attention.rope_provider is None


def test_sgl_kernel_provider_rejects_cpu_before_cache_or_mutation() -> None:
    provider = SGLKernelRoPEProvider(
        lambda *args, **kwargs: pytest.fail("kernel ran for a CPU input")
    )
    query = torch.randn((2, 3, 2, 64), dtype=torch.bfloat16)
    key = torch.randn_like(query)
    query_before = query.clone()
    key_before = key.clone()
    freqs = torch.randn((1, 3, 64), dtype=torch.bfloat16)

    result = provider(query, key, (freqs, 1.0))

    assert result is None
    assert torch.equal(query, query_before)
    assert torch.equal(key, key_before)
    assert provider._cos_sin_caches == {}
    assert provider._positions == {}


def test_sgl_kernel_cache_preserves_x_transformers_bf16_pair_rounding() -> None:
    angles = torch.tensor(
        [[[0.0, 0.0, 0.25, 0.25, -0.5, -0.5, 1.0, 1.0]]],
        dtype=torch.bfloat16,
    )

    cache = SGLKernelRoPEProvider._build_cos_sin_cache(
        angles,
        sequence_length=1,
    )
    cos, sin = cache.chunk(2, dim=-1)

    assert cache.dtype == torch.bfloat16
    assert cache.is_contiguous()
    assert torch.equal(cos.repeat_interleave(2, dim=-1), angles[0].cos())
    assert torch.equal(sin.repeat_interleave(2, dim=-1), angles[0].sin())


def test_dit_and_aggregator_own_distinct_source_local_providers() -> None:
    kernel = lambda *args, **kwargs: None
    aggregator_provider = SGLKernelRoPEProvider(kernel)
    dit_provider = SGLKernelRoPEProvider(kernel)
    aggregator = Aggregator(
        in_channels=8,
        hidden_size=128,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        llm_input_dim=16,
        attn_backend="torch",
        rope_provider=aggregator_provider,
    )
    dit = DiT(
        in_channels=8,
        hidden_size=128,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        llm_cond_dim=16,
        cfg_dropout_prob=0.0,
        attn_backend="torch",
        rope_provider=dit_provider,
    )
    native_aggregator = Aggregator(
        in_channels=8,
        hidden_size=128,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        llm_input_dim=16,
        attn_backend="torch",
    )
    native_dit = DiT(
        in_channels=8,
        hidden_size=128,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        llm_cond_dim=16,
        cfg_dropout_prob=0.0,
        attn_backend="torch",
    )

    assert aggregator_provider is not dit_provider
    assert all(
        block.attn.rope_provider is aggregator_provider for block in aggregator.blocks
    )
    assert all(block.attn.rope_provider is dit_provider for block in dit.blocks)
    assert list(aggregator.state_dict()) == list(native_aggregator.state_dict())
    assert list(dit.state_dict()) == list(native_dit.state_dict())
