# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.models.ming_tts.model_runner import (
    MingTTSModelRunner,
    MingTTSTPStepUpdate,
    _MingTTSRequestState,
)


def test_ming_tts_entry_tail_failure_is_published_before_reraise() -> None:
    runner = object.__new__(MingTTSModelRunner)
    runner._tp_rank = 0
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(weight=torch.empty(1, 4))
    )
    published = []

    def fail_tail(*_):
        raise RuntimeError("tail failed")

    runner._run_entry_tail_step = fail_tail
    runner._broadcast_tp_step_update = published.append
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(2, 1, 4))
    )

    with pytest.raises(RuntimeError, match="tail failed"):
        runner._collect_ming_tts_step(
            result,
            forward_batch=None,
            schedule_batch=SimpleNamespace(),
            requests=[SimpleNamespace(), SimpleNamespace()],
        )

    assert len(published) == 1
    assert published[0].tail_failed.tolist() == [1, 1]
    assert torch.count_nonzero(published[0].feedback_embeddings).item() == 0


def test_ming_tts_follower_rejects_tail_failure() -> None:
    runner = object.__new__(MingTTSModelRunner)
    update = MingTTSTPStepUpdate.empty_for_broadcast(
        batch_size=1,
        hidden_size=4,
        device=torch.device("cpu"),
        feedback_dtype=torch.float32,
    )
    update.tail_failed.fill_(1)

    with pytest.raises(RuntimeError, match="acoustic tail failed"):
        runner._apply_follower_step_update(update, [SimpleNamespace()])


def test_prefill_forward_publishes_sglang_forward_context() -> None:
    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda _batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        return "logits"

    model._decode_input_embedding = SimpleNamespace(
        weight=torch.zeros(1, dtype=torch.float32)
    )
    runner.model = model
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner._forward_with_input_embeds(forward_batch, torch.ones(1, 2))

    assert seen == [attn_backend]
    assert result.logits_output == "logits"


def test_ming_tts_prefill_replays_prompt_and_generated_feedback() -> None:
    def fail_token_embedding(_input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("continuous rows must not use token embeddings")

    runner = MingTTSModelRunner.__new__(MingTTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.empty((1, 2), dtype=torch.float32)
        ),
        get_input_embeddings=lambda: fail_token_embedding,
    )
    runner._request_states = {
        "req-ming-tts": _MingTTSRequestState(
            prefill_input_embeds=torch.tensor(
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
            ),
            feedback_embeddings=[
                torch.tensor([40.0, 41.0]),
                torch.tensor([50.0, 51.0]),
            ],
        )
    }
    request = SimpleNamespace(
        request_id="req-ming-tts",
        data=SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3]),
            req=SimpleNamespace(
                prefix_indices=torch.empty(0, dtype=torch.long),
                extend_range=SimpleNamespace(length=5),
            ),
        ),
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    actual = runner._build_prefill_input_embeds(forward_batch, [request])

    assert torch.equal(
        actual,
        torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ]
        ),
    )
