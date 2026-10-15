# SPDX-License-Identifier: Apache-2.0
"""Finished speculative rows release admission capacity before memory checks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.managers import scheduler as native_scheduler
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, ScheduleBatch

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.scheduling.spec_v2_helpers import make_owner


@pytest.fixture
def admission_owner() -> tuple[OmniScheduler, ScheduleBatch]:
    scheduler, batch, _ = make_owner()
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.orig_seq_lens = batch.seq_lens.clone()
    batch.spec_info = None
    batch.check_decode_mem = Mock(return_value=True)
    batch.prepare_for_decode = Mock()
    scheduler.new_token_ratio_tracker = Mock()
    scheduler.drain_spec_v2 = Mock()
    return scheduler, batch


@pytest.mark.parametrize("overlap_enabled", [False, True])
@pytest.mark.parametrize("initially_full", [False, True])
@pytest.mark.parametrize("finished_count", [0, 1, 3])
def test_real_native_updater_reopens_only_capacity_released_by_finished_rows(
    admission_owner: tuple[OmniScheduler, ScheduleBatch],
    overlap_enabled: bool,
    initially_full: bool,
    finished_count: int,
) -> None:
    scheduler, batch = admission_owner
    scheduler.enable_spec_v2_overlap = overlap_enabled
    batch.batch_is_full = initially_full
    expected_requests = batch.reqs[finished_count:]
    for request in batch.reqs[:finished_count]:
        request.finished_reason = FINISH_LENGTH(length=1)
    result = scheduler.update_running_batch(batch)
    assert result is batch
    assert batch.reqs == expected_requests
    assert batch.batch_is_full is (initially_full and finished_count == 0)
    scheduler.drain_spec_v2.assert_not_called()
    if expected_requests:
        batch.prepare_for_decode.assert_called_once_with()
    else:
        batch.prepare_for_decode.assert_not_called()


@pytest.mark.parametrize("memory_available", [False, True])
@pytest.mark.parametrize("force_retraction", [False, True])
def test_finished_rows_are_removed_before_memory_check_and_retraction_delegate(
    admission_owner: tuple[OmniScheduler, ScheduleBatch],
    monkeypatch: pytest.MonkeyPatch,
    memory_available: bool,
    force_retraction: bool,
) -> None:
    scheduler, batch = admission_owner
    batch.batch_is_full = True
    batch.reqs[1].finished_reason = FINISH_LENGTH(length=1)
    expected_requests = [batch.reqs[0], batch.reqs[2]]
    original_generations = scheduler.req_to_token_pool.req_generation.clone()
    execution_order: list[str] = []
    monkeypatch.setattr(native_scheduler, "TEST_RETRACT", force_retraction)
    scheduler.forward_ct = native_scheduler.TEST_RETRACT_INTERVAL

    def check_memory() -> bool:
        assert batch.reqs == expected_requests
        assert batch.req_pool_indices_cpu.tolist() == [7, 12]
        assert batch.seq_lens_cpu.tolist() == [2, 2]
        execution_order.append("memory")
        return memory_available

    def drain_pending() -> None:
        execution_order.append("drain")

    def update_native(
        native_owner: OmniScheduler, native_batch: ScheduleBatch
    ) -> ScheduleBatch:
        assert native_owner is scheduler
        assert native_batch is batch
        assert batch.reqs == expected_requests
        assert not batch.batch_is_full
        assert scheduler.req_to_token_pool.req_generation.equal(original_generations)
        execution_order.append("native")
        return native_batch

    batch.check_decode_mem = check_memory
    scheduler.drain_spec_v2 = drain_pending
    monkeypatch.setattr(
        native_scheduler.Scheduler, "update_running_batch", update_native
    )
    assert scheduler.update_running_batch(batch) is batch
    expected_order = (
        ["memory", "native"]
        if memory_available and not force_retraction
        else ["memory", "drain", "native"]
    )
    assert execution_order == expected_order


def test_off_path_leaves_finished_rows_for_the_native_updater(
    admission_owner: tuple[OmniScheduler, ScheduleBatch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, batch = admission_owner
    scheduler.enable_spec_v2_overlap = False
    batch.batch_is_full = True
    batch.reqs[0].finished_reason = FINISH_LENGTH(length=1)

    def update_native(
        native_owner: OmniScheduler, native_batch: ScheduleBatch
    ) -> ScheduleBatch:
        assert native_owner is scheduler
        assert native_batch is batch
        assert len(batch.reqs) == 3
        assert batch.reqs[0].finished()
        assert batch.batch_is_full
        return batch

    monkeypatch.setattr(
        native_scheduler.Scheduler, "update_running_batch", update_native
    )
    assert scheduler.update_running_batch(batch) is batch
    batch.check_decode_mem.assert_not_called()
    scheduler.drain_spec_v2.assert_not_called()


def test_unknown_writer_drain_does_not_reach_native_retraction(
    admission_owner: tuple[OmniScheduler, ScheduleBatch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, batch = admission_owner
    batch.batch_is_full = True
    finished_request = batch.reqs[1]
    finished_request.finished_reason = FINISH_LENGTH(length=1)
    original_generations = scheduler.req_to_token_pool.req_generation.clone()
    batch.check_decode_mem = Mock(return_value=False)
    scheduler.drain_spec_v2 = Mock(
        side_effect=RuntimeError("Unknown writer completion")
    )
    native_update = Mock()
    monkeypatch.setattr(
        native_scheduler.Scheduler, "update_running_batch", native_update
    )
    with pytest.raises(RuntimeError, match="Unknown writer completion"):
        scheduler.update_running_batch(batch)
    native_update.assert_not_called()
    assert finished_request not in batch.reqs
    assert finished_request.kv.req_pool_idx == 2
    assert scheduler.req_to_token_pool.req_generation.equal(original_generations)
