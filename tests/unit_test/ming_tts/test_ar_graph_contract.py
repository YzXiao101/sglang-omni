# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from sglang_omni.models.ming_tts.payload_types import (
    MingTTSState,
    encode_speaker_embedding,
)
from sglang_omni.models.ming_tts.tokenizer import (
    MingTTSSpecialTokenIds,
    MingTTSTokenizerBundle,
)
from sglang_omni.proto import OmniRequest, StagePayload


def _install_fake_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReq:
        def __init__(
            self,
            *,
            rid,
            origin_input_text,
            origin_input_ids,
            sampling_params,
            eos_token_ids=None,
            vocab_size=None,
            extra_key=None,
            **kwargs,
        ) -> None:
            del kwargs
            self.rid = rid
            self.origin_input_text = origin_input_text
            self.origin_input_ids = list(origin_input_ids)
            self.sampling_params = sampling_params
            self.eos_token_ids = eos_token_ids
            self.vocab_size = vocab_size
            self.extra_key = extra_key
            self.output_ids = []
            self.prefix_indices = []
            self.extend_input_len = len(origin_input_ids)

    class FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def normalize(self, tokenizer) -> None:
            del tokenizer

        def verify(self, vocab_size) -> None:
            self.vocab_size = vocab_size

    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.scheduler": types.ModuleType(
            "sglang.srt.managers.scheduler"
        ),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.sampling": types.ModuleType("sglang.srt.sampling"),
        "sglang.srt.sampling.sampling_params": types.ModuleType(
            "sglang.srt.sampling.sampling_params"
        ),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.sampling"):
        modules[name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].managers = modules["sglang.srt.managers"]
    modules["sglang.srt"].sampling = modules["sglang.srt.sampling"]
    modules["sglang.srt.managers"].scheduler = modules["sglang.srt.managers.scheduler"]
    modules["sglang.srt.managers"].schedule_batch = modules[
        "sglang.srt.managers.schedule_batch"
    ]
    modules["sglang.srt.sampling"].sampling_params = modules[
        "sglang.srt.sampling.sampling_params"
    ]
    modules["sglang.srt.managers.scheduler"].GenerationBatchResult = SimpleNamespace
    modules["sglang.srt.managers.schedule_batch"].Req = FakeReq
    modules["sglang.srt.sampling.sampling_params"].SamplingParams = FakeSamplingParams
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _fake_tokenizer_bundle() -> MingTTSTokenizerBundle:
    special = MingTTSSpecialTokenIds(
        bos=0,
        eos=1,
        pad=1,
        role_start=2,
        role_end=3,
        audio_patch=4,
        audio_start=5,
        end_of_audio=6,
        spk_start=7,
        spk_end=8,
    )
    return MingTTSTokenizerBundle(tokenizer=SimpleNamespace(), special=special)


def _fake_ming_tts_model():
    torch = pytest.importorskip("torch")

    class FakeMingTTSModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(vocab_size=32)
            self.patch_size = 2
            self.latent_dim = 3
            self.embedding = torch.nn.Embedding(32, 4)
            self.spk_head = torch.nn.Linear(192, 4, bias=False)
            torch.nn.init.ones_(self.spk_head.weight)

        def get_input_embeddings(self):
            return self.embedding

        def linear_proj_audio(self, latent):
            return torch.zeros(
                latent.shape[0],
                4,
                device=latent.device,
                dtype=self.embedding.weight.dtype,
            )

    return FakeMingTTSModel()


def _text_only_payload(request_id: str = "ming-text-1") -> StagePayload:
    state = MingTTSState(
        text="hello",
        input_ids=[0, 5, 6],
        audio_token_position=1,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={}),
        data=state.to_dict(),
    )


def _reference_payload(
    *,
    request_id: str = "ming-ref-1",
    speaker_value: float = 1.0,
) -> StagePayload:
    torch = pytest.importorskip("torch")
    state = MingTTSState(
        text="hello",
        input_ids=[0, 7, 8, 5, 6],
        audio_token_position=3,
        spk_token_positions=[1],
    )
    state_dict = state.to_dict()
    state_dict.update(
        encode_speaker_embedding(torch.full((1, 192), float(speaker_value)))
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={}),
        data=state_dict,
    )


def test_ming_prefill_row_cache_key_ids_are_stable_and_row_local() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.radix_cache_key import (
        build_ming_prefill_row_cache_key_ids,
        build_ming_row_prefill_extra_key,
    )

    rows = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=torch.bfloat16,
    )

    first = build_ming_prefill_row_cache_key_ids(rows)
    second = build_ming_prefill_row_cache_key_ids(rows.clone())

    assert first == second
    assert first[0] == first[2]
    assert first[0] != first[1]
    assert all(0 <= item < 2**63 for item in first)
    with pytest.raises(ValueError, match="2-D tensor"):
        build_ming_prefill_row_cache_key_ids(rows.reshape(1, 3, 3))

    namespace = build_ming_row_prefill_extra_key(
        model_identity="fake-model",
        input_dtype=torch.bfloat16,
        hidden_size=3,
        patch_size=2,
        latent_dim=3,
        audio_start_token_id=5,
        audio_patch_token_id=4,
        audio_eos_token_id=6,
    )
    changed = build_ming_row_prefill_extra_key(
        model_identity="fake-model",
        input_dtype=torch.float16,
        hidden_size=3,
        patch_size=2,
        latent_dim=3,
        audio_start_token_id=5,
        audio_patch_token_id=4,
        audio_eos_token_id=6,
    )

    assert namespace.startswith("ming_tts:row-prefill:v1:")
    assert namespace != changed
    assert "request" not in namespace


def test_projected_reference_prefill_rejects_radix_cache_in_graph_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=False,
    )

    with pytest.raises(RuntimeError, match="disable_radix_cache=True"):
        request_builder(_reference_payload())


def test_projected_reference_prefill_allowed_when_radix_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=True,
    )

    data = request_builder(_reference_payload())

    assert data.prefill_input_embeds is not None
    assert data.req._input_embeds_are_projected is True
    assert data.ar_state is not None
    assert not hasattr(data, "input_embeds_are_projected")
    assert not hasattr(data, "pending_feedback_queue")
    assert not hasattr(data, "generated_latents")
    assert not hasattr(data, "generated_last_chunk")
    assert int(data.state.max_decode_steps) == int(data.ar_state.max_decode_steps)


def test_text_only_prefill_is_not_marked_projected_without_row_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=True,
    )

    data = request_builder(_text_only_payload())

    assert data.prefill_input_embeds is None
    assert data.req._input_embeds_are_projected is False


def test_row_prefill_radix_cache_uses_content_ids_for_text_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=False,
        projected_prefill_radix_cache_enabled=True,
        model_cache_identity="fake-ming-model",
    )

    first = request_builder(_text_only_payload("text-a"))
    second = request_builder(_text_only_payload("text-b"))

    assert first.prefill_input_embeds is not None
    assert second.prefill_input_embeds is not None
    assert first.req._input_embeds_are_projected is True
    assert second.req._input_embeds_are_projected is True
    assert first.row_prefill_radix_cache_enabled is True
    assert first.req.origin_input_ids == second.req.origin_input_ids
    assert first.req.origin_input_ids != [0, 5, 6]
    assert first.req.extra_key == second.req.extra_key
    assert "text-a" not in first.req.extra_key
    assert "text-b" not in first.req.extra_key
    assert torch.equal(first.prompt_input_ids, torch.tensor([0, 5, 6]))
    assert first.row_prefill_input_ids.tolist() == first.req.origin_input_ids


def test_row_prefill_radix_cache_hashes_projected_speaker_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, _ = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=False,
        projected_prefill_radix_cache_enabled=True,
        model_cache_identity="fake-ming-model",
    )

    first = request_builder(_reference_payload(request_id="ref-a", speaker_value=1.0))
    second = request_builder(_reference_payload(request_id="ref-b", speaker_value=2.0))

    assert torch.equal(first.prompt_input_ids, second.prompt_input_ids)
    assert first.req.origin_input_ids != second.req.origin_input_ids
    assert len(first.req.origin_input_ids) == len(second.req.origin_input_ids)
    differing_rows = [
        idx
        for idx, (left, right) in enumerate(
            zip(first.req.origin_input_ids, second.req.origin_input_ids)
        )
        if left != right
    ]
    assert differing_rows == [2]
    assert first.req.extra_key == second.req.extra_key


def test_ming_ar_feedback_state_pool_stages_active_rows_in_place() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.ar_runtime import MingARFeedbackStatePool

    embedding = torch.nn.Embedding(3, 2)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.tensor(
                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
                dtype=torch.float32,
            )
        )
    pool = MingARFeedbackStatePool(embedding)
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    padding_row_before = embedding.weight[2].detach().clone()

    before_ptr = embedding.weight.data_ptr()
    pool.stage_feedback(rows)

    assert embedding.weight.data_ptr() == before_ptr
    assert torch.allclose(embedding.weight[:2], rows)
    assert torch.equal(embedding.weight[2], padding_row_before)
    assert torch.equal(pool.row_ids(2), torch.tensor([0, 1]))
    with pytest.raises(RuntimeError, match="decode batch exceeds"):
        pool.validate_batch_size(4)
    with pytest.raises(RuntimeError, match="hidden size mismatch"):
        pool.stage_feedback(torch.ones(1, 3))


def test_ming_ar_latent_history_update_matches_shift_contract() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.ar_runtime import update_ming_ar_latent_history_

    history = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    sampled = torch.tensor(
        [[[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]]],
        dtype=torch.float32,
    )

    update_ming_ar_latent_history_(history, sampled)

    expected = torch.tensor(
        [
            [
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [100.0, 101.0, 102.0],
                [200.0, 201.0, 202.0],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.equal(history, expected)


def _fake_ar_model(torch_module, *, stop: bool = False):
    class FakeFlowLoss:
        def __init__(self, sample_value: float = 1.0) -> None:
            self.sample_value = sample_value

        def sample(self, hidden, history, *, cfg, patch_size, sigma, temperature):
            del hidden, history, cfg, sigma, temperature
            return (
                torch_module.full((1, int(patch_size), 3), self.sample_value),
                None,
            )

    class FakeStopHead(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, hidden):
            batch = int(hidden.shape[0])
            if stop:
                row = torch_module.tensor(
                    [[-10.0, 10.0]],
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
            else:
                row = torch_module.tensor(
                    [[10.0, -10.0]],
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
            return row.reshape(1, 1, 2).expand(batch, 1, 2)

    class FakeARModel:
        def __init__(self) -> None:
            self.patch_size = 2
            self.latent_dim = 3
            self.history_patch_size = 4
            self.flowloss = FakeFlowLoss()
            self.stop_head = FakeStopHead()
            self._decode_input_embedding = SimpleNamespace(
                weight=torch_module.empty(1, 4, dtype=torch_module.float32)
            )

        def linear_proj_audio(self, sampled):
            return torch_module.full(
                (int(sampled.shape[0]), 4),
                7.0,
                dtype=sampled.dtype,
                device=sampled.device,
            )

    return FakeARModel()


def _fake_ar_request(*, step: int = 0, max_decode_steps: int = 5):
    from sglang_omni.models.ming_tts.ar_runtime import MingARRequestState

    ar_state = MingARRequestState(
        generation_steps=step,
        max_decode_steps=max_decode_steps,
        cfg=2.0,
        sigma=0.25,
        flow_temperature=0.0,
        audio_patch_token_id=4,
        audio_eos_token_id=6,
        audio_token_id=5,
    )
    data = SimpleNamespace(ar_state=ar_state, generation_steps=step)
    return SimpleNamespace(request_id="req-1", data=data)


def test_ming_ar_tail_appends_feedback_for_non_final_step() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.model_runner import MingARTailExecutor

    request = _fake_ar_request(step=0, max_decode_steps=5)
    ar_tail = MingARTailExecutor(_fake_ar_model(torch, stop=False))

    step_result = ar_tail.step_batch(torch.ones(1, 1, 4), [request])

    assert torch.equal(step_result.next_token_ids.cpu(), torch.tensor([4]))
    assert torch.equal(step_result.generation_steps.cpu(), torch.tensor([0]))
    assert torch.equal(step_result.feedback_mask.cpu(), torch.tensor([1]))
    assert torch.equal(step_result.stop_flags.cpu(), torch.tensor([0]))
    assert torch.allclose(
        step_result.feedback_embeddings.cpu(),
        torch.full((1, 4), 7.0),
    )
    assert len(step_result.generated_latents) == 1
    assert len(request.data.ar_state.generated_latents) == 1
    assert request.data.ar_state.generated_last_chunk == [False]
    assert len(request.data.ar_state.pending_feedback_queue) == 1
    assert request.data.ar_state.generation_steps == 1
    assert request.data.ar_state.latent_history is not None


def test_ming_ar_tail_does_not_append_feedback_on_length_end() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.model_runner import MingARTailExecutor

    request = _fake_ar_request(step=0, max_decode_steps=1)
    ar_tail = MingARTailExecutor(_fake_ar_model(torch, stop=False))

    step_result = ar_tail.step_batch(torch.ones(1, 1, 4), [request])

    assert torch.equal(step_result.next_token_ids.cpu(), torch.tensor([4]))
    assert torch.equal(step_result.feedback_mask.cpu(), torch.tensor([0]))
    assert request.data.ar_state.generated_last_chunk == [False]
    assert len(request.data.ar_state.pending_feedback_queue) == 0
    assert request.data.ar_state.generation_steps == 1


def test_ming_ar_tail_stop_path_uses_eos_without_feedback() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.model_runner import MingARTailExecutor

    request = _fake_ar_request(step=4, max_decode_steps=8)
    ar_tail = MingARTailExecutor(_fake_ar_model(torch, stop=True))

    step_result = ar_tail.step_batch(torch.ones(1, 1, 4), [request])

    assert torch.equal(step_result.next_token_ids.cpu(), torch.tensor([6]))
    assert torch.equal(step_result.feedback_mask.cpu(), torch.tensor([0]))
    assert torch.equal(step_result.stop_flags.cpu(), torch.tensor([1]))
    assert request.data.ar_state.generated_last_chunk == [True]
    assert request.data.ar_state.stop_step == 4
    assert len(request.data.ar_state.pending_feedback_queue) == 0
    assert request.data.ar_state.generation_steps == 5


def test_ming_ar_follower_applies_feedback_without_latent_collection() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_tts.model_runner import (
        MingARStepResult,
        apply_follower_ming_ar_step_result,
    )

    request = _fake_ar_request(step=0, max_decode_steps=5)
    feedback = torch.full((1, 4), 9.0)
    step_result = MingARStepResult(
        next_token_ids=torch.tensor([4]),
        feedback_embeddings=feedback,
        feedback_mask=torch.tensor([1]),
        stop_flags=torch.tensor([0]),
        generation_steps=torch.tensor([0]),
        request_ids=["req-1"],
    )

    apply_follower_ming_ar_step_result(step_result, [request])

    assert request.data.ar_state.generation_steps == 1
    assert len(request.data.ar_state.pending_feedback_queue) == 1
    assert torch.equal(request.data.ar_state.pending_feedback_queue[0], feedback[0])
    assert request.data.ar_state.generated_latents == []
    assert request.data.ar_state.generated_last_chunk == []


def test_ming_tts_runner_broadcasts_step_result_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.model_runner import (
        MingARStepResult,
        MingTTSModelRunner,
    )

    calls: list[tuple[tuple[int, ...], torch.dtype, int, object]] = []

    def fake_broadcast(tensor, *, src, group):
        calls.append((tuple(tensor.shape), tensor.dtype, int(src), group))

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    group = SimpleNamespace(ranks=[10, 11], device_group=object())
    runner = object.__new__(MingTTSModelRunner)
    runner._tp_size = 2
    runner.tp_worker = SimpleNamespace(get_tp_group=lambda: group)
    step_result = MingARStepResult(
        next_token_ids=torch.tensor([4]),
        feedback_embeddings=torch.ones(1, 4),
        feedback_mask=torch.tensor([1]),
        stop_flags=torch.tensor([0]),
        generation_steps=torch.tensor([0]),
        request_ids=["req-1"],
    )

    runner._broadcast_step_result(step_result)

    assert calls == [
        ((1,), torch.long, 10, group.device_group),
        ((1, 4), torch.float32, 10, group.device_group),
        ((1,), torch.long, 10, group.device_group),
        ((1,), torch.long, 10, group.device_group),
        ((1,), torch.long, 10, group.device_group),
    ]


def test_ming_tts_runner_rejects_sharded_hidden_for_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner

    runner = object.__new__(MingTTSModelRunner)
    runner.model = _fake_ar_model(torch)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(1, 1, 3))
    )

    with pytest.raises(RuntimeError, match="full hidden states"):
        runner._run_tts_step(
            result,
            SimpleNamespace(),
            SimpleNamespace(),
            [_fake_ar_request()],
        )


def test_ming_tts_runner_rejects_feedback_buffer_reallocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.ar_runtime import get_ming_ar_feedback_state_pool
    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner

    runner = object.__new__(MingTTSModelRunner)
    runner.model = _fake_ar_model(torch)
    pool = get_ming_ar_feedback_state_pool(runner.model)
    runner._feedback_buffer_contract = runner._capture_feedback_buffer_contract(pool)

    runner.model._decode_input_embedding = SimpleNamespace(
        weight=torch.empty(1, 4, dtype=torch.float32)
    )
    delattr(runner.model, "_ming_ar_feedback_state_pool")
    changed_pool = get_ming_ar_feedback_state_pool(runner.model)

    with pytest.raises(RuntimeError, match="feedback buffer changed"):
        runner._validate_feedback_buffer_contract(changed_pool)


def test_ming_tts_runner_rejects_projected_prefill_hidden_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner

    runner = object.__new__(MingTTSModelRunner)
    runner.model = _fake_ar_model(torch)

    with pytest.raises(RuntimeError, match="projected prefill hidden size"):
        runner._validate_projected_prefill_embeds(torch.ones(1, 3))


def test_ming_ar_result_adapter_serializes_then_releases_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_fake_sglang(monkeypatch)
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )

    request_builder, result_adapter = make_ming_tts_scheduler_adapters(
        model=_fake_ming_tts_model(),
        tokenizer=_fake_tokenizer_bundle(),
        projected_prefill_requires_radix_disabled=True,
        radix_cache_disabled=True,
    )
    data = request_builder(_reference_payload())
    data.ar_state.generated_latents.append(torch.ones(2, 3))
    data.ar_state.generated_last_chunk.append(False)
    data.finish_reason = "length"

    result = result_adapter(data)

    assert result.data["generated_latents_shape"] == [1, 2, 3]
    assert data.ar_state.generated_latents == []
    assert data.ar_state.latent_history is None
    assert len(data.ar_state.pending_feedback_queue) == 0


def _run_fake_ming_ar_graph_executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gpu_id: int | None = None,
    enable_ming_ar_cuda_graph: bool = False,
    enable_ming_ar_projected_prefill_radix_cache: bool = False,
    llm_config: SimpleNamespace | None = None,
    server_args_overrides: dict[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
) -> SimpleNamespace:
    from sglang_omni.models.ming_tts import stages

    build_kwargs: dict[str, object] = {}
    infrastructure_saw_graph_disabled: list[bool] = []
    graph_init_saw_graph_enabled: list[bool] = []
    infrastructure_calls: list[dict[str, object]] = []

    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda path: path)
    if llm_config is None:
        llm_config = SimpleNamespace(
            max_position_embeddings=4096,
            hidden_size=8,
            head_dim=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    monkeypatch.setattr(
        stages,
        "_load_ming_tts_config",
        lambda path: SimpleNamespace(llm_config=llm_config),
    )
    monkeypatch.setattr(
        stages,
        "load_ming_tts_tokenizer",
        lambda *args, **kwargs: _fake_tokenizer_bundle(),
    )

    backend_module = types.ModuleType("sglang_omni.scheduling.sglang_backend")

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        build_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    backend_module.build_sglang_server_args = fake_build_sglang_server_args
    backend_module.SGLangOutputProcessor = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        backend_module,
    )

    bootstrap_module = types.ModuleType("sglang_omni.scheduling.bootstrap")

    class FakeRunner:
        def __init__(self, server_args) -> None:
            self.server_args = server_args
            self.model = SimpleNamespace(
                model=SimpleNamespace(layers=[object(), object()]),
                eval=lambda: None,
            )

        def init_device_graphs(self) -> None:
            graph_init_saw_graph_enabled.append(
                not bool(self.server_args.disable_cuda_graph)
            )

    class FakeWorker:
        def __init__(self, server_args) -> None:
            self.model_runner = FakeRunner(server_args)

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        infrastructure_calls.append({"gpu_id": gpu_id, **kwargs})
        infrastructure_saw_graph_disabled.append(bool(server_args.disable_cuda_graph))
        return (
            FakeWorker(server_args),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        )

    bootstrap_module.create_sglang_infrastructure = fake_create_sglang_infrastructure
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.bootstrap",
        bootstrap_module,
    )

    scheduler_module = types.ModuleType("sglang_omni.scheduling.omni_scheduler")
    scheduler_module.OmniScheduler = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.omni_scheduler",
        scheduler_module,
    )

    model_runner_module = types.ModuleType("sglang_omni.models.ming_tts.model_runner")
    model_runner_module.MingTTSModelRunner = lambda *args, **kwargs: SimpleNamespace(
        args=args, kwargs=kwargs
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_tts.model_runner",
        model_runner_module,
    )

    request_builders_module = types.ModuleType(
        "sglang_omni.models.ming_tts.sglang_request_builders"
    )
    adapter_kwargs: dict[str, object] = {}

    def fake_make_adapters(**kwargs):
        adapter_kwargs.update(kwargs)
        return (lambda payload: payload, lambda data: data)

    request_builders_module.make_ming_tts_scheduler_adapters = fake_make_adapters
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_tts.sglang_request_builders",
        request_builders_module,
    )

    scheduler = stages.create_sglang_tts_engine_executor(
        "model",
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
        enable_ming_ar_cuda_graph=enable_ming_ar_cuda_graph,
        enable_ming_ar_projected_prefill_radix_cache=(
            enable_ming_ar_projected_prefill_radix_cache
        ),
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    )

    return SimpleNamespace(
        adapter_kwargs=adapter_kwargs,
        build_kwargs=build_kwargs,
        graph_init_saw_graph_enabled=graph_init_saw_graph_enabled,
        infrastructure_calls=infrastructure_calls,
        infrastructure_saw_graph_disabled=infrastructure_saw_graph_disabled,
        scheduler=scheduler,
    )


def test_ming_ar_cuda_graph_disables_radix_and_chunked_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        enable_ming_ar_cuda_graph=True,
    )
    build_kwargs = result.build_kwargs

    assert build_kwargs["tp_size"] == 1
    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["enable_torch_compile"] is False
    assert build_kwargs["torch_compile_max_bs"] == 0
    assert build_kwargs["disable_radix_cache"] is True
    assert build_kwargs["chunked_prefill_size"] == 0
    assert result.infrastructure_saw_graph_disabled == [True]
    assert result.graph_init_saw_graph_enabled == [True]
    assert result.adapter_kwargs["projected_prefill_requires_radix_disabled"] is True
    assert result.adapter_kwargs["radix_cache_disabled"] is True
    assert result.scheduler.server_args.disable_radix_cache is True


def test_ming_tts_tp2_bootstrap_passes_tp_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        gpu_id=1,
        enable_ming_ar_cuda_graph=True,
        total_gpu_memory_fraction=0.72,
        tp_rank=1,
        tp_size=2,
        nccl_port=29501,
    )

    assert result.build_kwargs["tp_size"] == 2
    assert result.infrastructure_calls == [
        {
            "gpu_id": 1,
            "model_arch_override": "MingTTSSGLangModel",
            "tp_rank": 1,
            "nccl_port": 29501,
            "total_gpu_memory_fraction": 0.72,
        }
    ]


def test_ming_tts_tp2_requires_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires enable_ming_ar_cuda_graph"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp2_rejects_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="torch.compile is not currently supported"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            server_args_overrides={"torch_compile_max_bs": 1},
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp2_rejects_projected_prefill_radix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="radix cache disabled"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            enable_ming_ar_projected_prefill_radix_cache=True,
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp2_requires_captured_graph_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="graph-only requires max_running_requests"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            server_args_overrides={
                "cuda_graph_bs": [1, 2],
                "cuda_graph_max_bs": 2,
                "max_running_requests": 3,
            },
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp2_rejects_non_divisible_attention_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="attention heads divisible"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            llm_config=SimpleNamespace(
                max_position_embeddings=4096,
                hidden_size=6,
                head_dim=2,
                num_attention_heads=3,
                num_key_value_heads=2,
            ),
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp2_requires_nccl_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="TP requires nccl_port"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            tp_size=2,
        )


def test_ming_tts_tp_rank_must_be_in_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="tp_rank=2 is out of range"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            tp_rank=2,
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_tts_tp_size_override_must_match_stage_tp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="server_args_overrides.tp_size"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            server_args_overrides={"tp_size": 1},
            tp_size=2,
            nccl_port=29501,
        )


def test_ming_ar_cuda_graph_profile_defaults_to_graph_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        enable_ming_ar_cuda_graph=True,
    )
    build_kwargs = result.build_kwargs

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["enable_torch_compile"] is False
    assert build_kwargs["torch_compile_max_bs"] == 0
    assert build_kwargs["disable_radix_cache"] is True
    assert build_kwargs["chunked_prefill_size"] == 0
    assert result.infrastructure_saw_graph_disabled == [True]
    assert result.graph_init_saw_graph_enabled == [True]
    assert result.adapter_kwargs["projected_prefill_requires_radix_disabled"] is True
    assert result.adapter_kwargs["radix_cache_disabled"] is True


def test_ming_ar_cuda_graph_defaults_capacity_to_graph_max_bs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        enable_ming_ar_cuda_graph=True,
        server_args_overrides={"cuda_graph_bs": [1, 2, 4]},
    )
    build_kwargs = result.build_kwargs

    assert build_kwargs["cuda_graph_max_bs"] == 4
    assert build_kwargs["max_running_requests"] == 4
    assert build_kwargs["torch_compile_max_bs"] == 0


def test_ming_ar_cuda_graph_row_prefill_radix_keeps_cache_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        enable_ming_ar_cuda_graph=True,
        server_args_overrides={
            "enable_ming_ar_projected_prefill_radix_cache": True,
            "cuda_graph_bs": [1],
            "cuda_graph_max_bs": 1,
            "torch_compile_max_bs": 0,
        },
    )
    build_kwargs = result.build_kwargs

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["enable_torch_compile"] is False
    assert build_kwargs["disable_radix_cache"] is False
    assert build_kwargs["chunked_prefill_size"] == 0
    assert build_kwargs["disable_overlap_schedule"] is True
    assert result.adapter_kwargs["projected_prefill_requires_radix_disabled"] is True
    assert result.adapter_kwargs["radix_cache_disabled"] is False
    assert result.adapter_kwargs["projected_prefill_radix_cache_enabled"] is True
    assert result.scheduler.server_args.disable_radix_cache is False


def test_ming_ar_cuda_graph_rejects_capacity_smaller_than_graph_max_bs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="max_running_requests >= cuda_graph_max_bs"):
        _run_fake_ming_ar_graph_executor(
            monkeypatch,
            enable_ming_ar_cuda_graph=True,
            server_args_overrides={
                "cuda_graph_bs": [1, 2],
                "max_running_requests": 1,
            },
        )


def test_ming_ar_cuda_graph_allows_without_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_fake_ming_ar_graph_executor(
        monkeypatch,
        enable_ming_ar_cuda_graph=True,
        server_args_overrides={
            "cuda_graph_bs": [1],
            "cuda_graph_max_bs": 1,
            "torch_compile_max_bs": 0,
        },
    )
    build_kwargs = result.build_kwargs

    assert build_kwargs["disable_cuda_graph"] is False
    assert build_kwargs["enable_torch_compile"] is False
    assert build_kwargs["torch_compile_max_bs"] == 0
    assert build_kwargs["disable_radix_cache"] is True
    assert build_kwargs["chunked_prefill_size"] == 0
    assert result.infrastructure_saw_graph_disabled == [True]
    assert result.graph_init_saw_graph_enabled == [True]
    assert result.adapter_kwargs["projected_prefill_requires_radix_disabled"] is True
    assert result.adapter_kwargs["radix_cache_disabled"] is True
