# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 DSpark worker whose verification keeps the stage's sampling exact.

The DSpark verify samples from temperature and top-k/top-p only. The stage also
applies a repetition penalty over the output history, suppresses stop ids until the
minimum length, and redraws a speech token that repeats inside the recent window, so
this worker feeds every verify row's history to dspark_sampling.exact_accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import FinalizeAcceptLens
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import DynamicGradMode

from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.models.fun_cosyvoice3.dspark_draft import FunCosyVoice3DSparkDraftModel
from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    EMPTY_HISTORY_TOKEN,
    DraftSamplingState,
    ExactAccept,
    exact_accept,
    sample_draft_block,
)
from sglang_omni.models.fun_cosyvoice3.model_runner import COSYVOICE3_RAS_WINDOW_SIZE
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE


@dataclass(kw_only=True)
class StepSamplingState:
    """Sampling inputs of the running requests for one speculative step."""

    draft: DraftSamplingState
    recent_tokens: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor


class SequenceTokenCountPublisher(Protocol):
    def __call__(
        self,
        sequence_token_counts: torch.Tensor,
        /,
        *,
        confidence: torch.Tensor | None = None,
    ) -> None: ...


class GrammarStateAdvancer(Protocol):
    def __call__(self) -> None: ...


class RequestHistories:
    """Committed output history, advanced on the device after each verified block."""

    def __init__(self, *, slot_count: int, device: torch.device) -> None:
        self.presence_by_slot: torch.Tensor = torch.zeros(
            (slot_count, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=device
        )
        self.recent_token_ids_by_slot: torch.Tensor = torch.full(
            (slot_count, COSYVOICE3_RAS_WINDOW_SIZE),
            EMPTY_HISTORY_TOKEN,
            dtype=torch.long,
            device=device,
        )
        self.output_token_counts_by_slot: torch.Tensor = torch.zeros(
            slot_count, dtype=torch.long, device=device
        )
        self.slot_owners: list[int | None] = [None] * slot_count

    def ensure_initialized(self, batch: ScheduleBatch) -> None:
        for request in batch.reqs:
            request_pool_index = request.kv.req_pool_idx
            if self.slot_owners[request_pool_index] == id(request):
                continue
            else:
                self.slot_owners[request_pool_index] = id(request)
            output_token_ids = list(request.output_ids)
            device = self.presence_by_slot.device
            self.presence_by_slot[request_pool_index] = False
            self.presence_by_slot[
                request_pool_index,
                torch.tensor(output_token_ids, dtype=torch.long, device=device),
            ] = True
            recent_token_ids = output_token_ids[-COSYVOICE3_RAS_WINDOW_SIZE:]
            self.recent_token_ids_by_slot[request_pool_index] = torch.tensor(
                [EMPTY_HISTORY_TOKEN]
                * (COSYVOICE3_RAS_WINDOW_SIZE - len(recent_token_ids))
                + recent_token_ids,
                dtype=torch.long,
                device=device,
            )
            self.output_token_counts_by_slot[request_pool_index] = len(output_token_ids)

    def presence_mask(self, batch: ScheduleBatch) -> torch.Tensor:
        self.ensure_initialized(batch)
        return self.presence_by_slot[batch.req_pool_indices]

    def commit(
        self,
        request_pool_indices: torch.Tensor,
        accepted_token_ids: torch.Tensor,
        accepted_token_counts: torch.Tensor,
    ) -> None:
        """Append valid outputs, including the bonus; padding never enters history."""
        accepted_token_counts = accepted_token_counts.to(torch.long)
        valid_token_mask = (
            torch.arange(accepted_token_ids.shape[1], device=accepted_token_ids.device)[
                None, :
            ]
            < accepted_token_counts[:, None]
        )
        safe_token_ids = accepted_token_ids.long().masked_fill(~valid_token_mask, 0)
        token_presence_mask = self.presence_by_slot[request_pool_indices]
        # note (Codex): CUDA amax scatter requires a numeric dtype, even for 0/1 data.
        token_presence_mask.view(torch.uint8).scatter_reduce_(
            1,
            safe_token_ids,
            valid_token_mask.view(torch.uint8),
            reduce="amax",
            include_self=True,
        )
        self.presence_by_slot[request_pool_indices] = token_presence_mask
        recent_and_accepted_token_ids = torch.cat(
            (self.recent_token_ids_by_slot[request_pool_indices], accepted_token_ids),
            dim=1,
        )
        recent_token_positions = (
            accepted_token_counts[:, None]
            + torch.arange(
                COSYVOICE3_RAS_WINDOW_SIZE, device=accepted_token_ids.device
            )[None, :]
        )
        self.recent_token_ids_by_slot[request_pool_indices] = (
            recent_and_accepted_token_ids.gather(1, recent_token_positions)
        )
        self.output_token_counts_by_slot[request_pool_indices] += accepted_token_counts


def build_step_sampling_state(
    batch: ScheduleBatch,
    sampling_info: SamplingBatchInfo,
    histories: RequestHistories,
) -> StepSamplingState:
    """Collect each running request's committed history and sampling parameters."""
    device = sampling_info.temperatures.device
    token_presence_mask = histories.presence_mask(batch)
    request_values = torch.tensor(
        [
            [
                request.sampling_params.repetition_penalty,
                request.sampling_params.min_new_tokens,
            ]
            for request in batch.reqs
        ],
        dtype=torch.float64,
        device=device,
    )
    top_ks = sampling_info.top_ks.view(-1)
    top_ps = sampling_info.top_ps.view(-1).float()
    if not (sampling_info.need_top_p_sampling or sampling_info.need_top_k_sampling):
        # The stage samples the full softmax when no request filters tokens.
        top_ps = torch.full_like(top_ps, float("inf"))
    else:
        pass
    return StepSamplingState(
        draft=DraftSamplingState(
            presence_mask=token_presence_mask,
            repetition_penalties=request_values[:, 0].float(),
            output_lengths=histories.output_token_counts_by_slot[
                batch.req_pool_indices
            ],
            min_new_tokens=request_values[:, 1].long(),
            temperatures=sampling_info.temperatures.view(-1).float(),
            greedy=top_ks == 1,
        ),
        recent_tokens=histories.recent_token_ids_by_slot[batch.req_pool_indices],
        top_ks=top_ks,
        top_ps=top_ps,
    )


class ExactVerifyExecutor(TargetVerifyExecutor):
    """Accept drafts against the stage's exact per-position distribution."""

    step_state: StepSamplingState | None = None
    accept_function: ExactAccept = staticmethod(exact_accept)

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor | None,
        draft_block: DraftBlockResult,
        sampling_info: SamplingBatchInfo,
        draft_input: DFlashDraftInputV2,
        layout: None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> AcceptOuts:
        state = self.step_state
        assert not folded_accept and layout is None and target_logits is not None
        assert state is not None, "a verify step needs its sampling state"
        draft_state = state.draft
        draft_tokens = draft_tokens.long()
        accepted_count, next_tokens = self.accept_function(
            target_logits,
            draft_block.corrected_logits,
            draft_tokens,
            draft_state.presence_mask,
            draft_state.repetition_penalties,
            draft_state.output_lengths,
            draft_state.min_new_tokens,
            draft_state.temperatures,
            draft_state.greedy,
            state.top_ks,
            state.top_ps,
            state.recent_tokens,
            torch.rand(draft_tokens.shape, device=draft_tokens.device),
            torch.empty(
                (bs, TOTAL_VOCAB_SIZE), device=draft_tokens.device
            ).exponential_(),
        )
        finalized = FinalizeAcceptLens.execute(
            correct_len=accepted_count,
            cap_trim_lens=torch.zeros_like(accepted_count),
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=accepted_count,
            bonus=next_tokens,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=accepted_count,
            bonus=next_tokens,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )


FA3_DRAFT_WIDTH_LOCK = Lock()
fa3_draft_width_installed = False


def install_fa3_draft_width() -> None:
    """Make SGLang's FA3 backend lay out a DSpark draft batch at the draft's real width."""
    global fa3_draft_width_installed
    with FA3_DRAFT_WIDTH_LOCK:
        if fa3_draft_width_installed:
            return
        else:
            pass
        sglang_init = FlashAttentionBackend.__init__

        def init_with_draft_width(
            self: FlashAttentionBackend,
            model_runner: ModelRunner,
            skip_prefill: bool = False,
            speculative_step_id: int = 0,
            topk: int = 0,
            speculative_num_steps: int = 0,
            fa_impl_ver: int = 3,
        ) -> None:
            sglang_init(
                self,
                model_runner,
                skip_prefill,
                speculative_step_id,
                topk,
                speculative_num_steps,
                fa_impl_ver,
            )
            # note (wilsonzheng0327): SGLang 0.5.21 sizes a DSpark draft batch as gamma
            # rows per request, but a sample_from_anchor=False draft (this checkpoint)
            # also feeds the anchor row, so every request after the first would attend
            # over shifted rows. Its model runner already sizes logits this way; remove
            # once the FA3 backend does too.
            if (
                model_runner.is_draft_worker
                and self.speculative_num_draft_tokens is not None
                and SpeculativeAlgorithm.from_string(
                    get_spec().speculative_algorithm
                ).is_dspark()
                and not get_dspark_sample_from_anchor(
                    model_runner.model_config.hf_config
                )
            ):
                self.speculative_num_draft_tokens = int(
                    get_spec().speculative_num_draft_tokens
                )
            else:
                pass

        FlashAttentionBackend.__init__ = init_with_draft_width
        fa3_draft_width_installed = True


class FunCosyVoice3DSparkWorker(DSparkWorkerV2):
    """DSpark worker that mirrors the stage's sampling in drafting and verification."""

    request_histories: RequestHistories

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        install_fa3_draft_width()
        # note (wilsonzheng0327): the draft draw and the acceptance noise are local,
        # so tensor-parallel ranks would disagree on the committed tokens.
        if get_parallel().tp_size != 1:
            raise ValueError("Fun-CosyVoice3 speculative decoding runs on one GPU")
        elif (
            read_ragged_verify_mode() != RaggedVerifyMode.STATIC
            or envs.SGLANG_SIMULATE_ACC_LEN.get() > 0
        ):
            raise ValueError(
                "Fun-CosyVoice3 speculative decoding verifies whole blocks; unset "
                "SGLANG_RAGGED_VERIFY_MODE and SGLANG_SIMULATE_ACC_LEN"
            )
        else:
            pass
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # The engine builder always materializes the draft under this class.
        assert isinstance(self.draft_model, FunCosyVoice3DSparkDraftModel)
        self.exact_verify_executor = ExactVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            tp_sync=self._tp_sync,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self._verify_executor = (
            self.exact_verify_executor
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        # note (wilsonzheng0327): eager, the draft block and the exact verify are about a
        # hundred small kernels per step; compiled they fuse into a few.
        self.exact_verify_executor.accept_function = torch.compile(
            exact_accept, dynamic=True
        )
        self.draft_model.markov_head.block_sampler = torch.compile(
            sample_draft_block, dynamic=True
        )
        self.warm_up_compiled_sampling()

    def alloc_memory_pool(
        self,
        memory_pool_config: MemoryPoolConfig | None = None,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
    ) -> None:
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        # note (wilsonzheng0327): the draft loads before the target sizes its pool, so
        # the per-slot histories are built once the slots exist.
        self.request_histories = RequestHistories(
            slot_count=self.model_runner.req_to_token_pool.req_to_token.shape[0],
            device=self.device,
        )

    # note (wilsonzheng0327): compiled graphs guard on grad mode, so the warm-up runs
    # under the scheduler loop's mode.
    @DynamicGradMode()
    def warm_up_compiled_sampling(self) -> None:
        """Compile the draft block and the exact verify before the first request.

        Dynamic shapes still specialize a single running request, so both one and two
        requests are traced here instead of on the serving thread.
        """
        markov_head = self.draft_model.markov_head
        draft_count = self.gamma
        for batch_size in (1, 2):
            zero_per_request = torch.zeros(
                batch_size, dtype=torch.long, device=self.device
            )
            one_per_request = torch.ones(
                batch_size, dtype=torch.float32, device=self.device
            )
            presence_mask = torch.zeros(
                (batch_size, TOTAL_VOCAB_SIZE), dtype=torch.bool, device=self.device
            )
            greedy = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            base_logits = torch.zeros(
                (batch_size, draft_count, TOTAL_VOCAB_SIZE), device=self.device
            )
            draft_tokens, shaped_logits = markov_head.block_sampler(
                base_logits,
                zero_per_request,
                torch.ones_like(base_logits),
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                presence_mask,
                one_per_request,
                zero_per_request,
                zero_per_request,
                one_per_request,
                greedy,
            )
            self.exact_verify_executor.accept_function(
                torch.zeros(
                    (batch_size * (draft_count + 1), TOTAL_VOCAB_SIZE),
                    device=self.device,
                ),
                shaped_logits,
                draft_tokens,
                presence_mask,
                one_per_request,
                zero_per_request,
                zero_per_request,
                one_per_request,
                greedy,
                torch.full(
                    (batch_size,),
                    TOTAL_VOCAB_SIZE,
                    dtype=torch.int32,
                    device=self.device,
                ),
                one_per_request,
                torch.full(
                    (batch_size, COSYVOICE3_RAS_WINDOW_SIZE),
                    EMPTY_HISTORY_TOKEN,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.zeros((batch_size, draft_count), device=self.device),
                torch.ones((batch_size, TOTAL_VOCAB_SIZE), device=self.device),
            )

    def _maybe_build_draft_sampler(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self, *, available_memory_gb: float
    ) -> None:
        """The folded in-graph sampler would skip the mirrored Markov head."""
        return None

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: SequenceTokenCountPublisher | None = None,
        grammar_barrier: GrammarStateAdvancer | None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        prefill_token_counts: torch.Tensor | None = None
        is_decode = not (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        )
        if is_decode:
            sampling_info = batch.sampling_info
            # Forward isolation hands over a copy without the penalizer, whose
            # step-start penalties would be wrong for every verify row after the first.
            assert sampling_info.penalizer_orchestrator is None
            # The stage's request builder sets none of these, and the exact verify has no
            # term for them.
            assert not (
                sampling_info.need_min_p_sampling
                or sampling_info.logit_bias is not None
                or sampling_info.has_custom_logit_processor
                or batch.return_logprob
            )
            state = build_step_sampling_state(
                batch, sampling_info, self.request_histories
            )
            self.draft_model.markov_head.sampling_state = state.draft
            self.exact_verify_executor.step_state = state
        elif not batch.forward_mode.is_idle():
            for request in batch.reqs:
                self.request_histories.slot_owners[request.kv.req_pool_idx] = None
            self.request_histories.ensure_initialized(batch)
            prefill_token_counts = torch.tensor(
                [
                    int(request is not batch.chunked_req and not batch.is_prefill_only)
                    for request in batch.reqs
                ],
                dtype=torch.long,
                device=batch.req_pool_indices.device,
            )
        else:
            pass
        result = super().forward_batch_generation(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if is_decode:
            self.request_histories.commit(
                batch.req_pool_indices,
                result.next_token_ids.view(len(batch.reqs), -1),
                result.accept_lens,
            )
        elif prefill_token_counts is not None:
            # note (Codex): The first decode can launch before CPU prefill commits.
            self.request_histories.commit(
                batch.req_pool_indices,
                result.next_token_ids.view(len(batch.reqs), -1),
                prefill_token_counts,
            )
        else:
            pass
        return result
