# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)
from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    MingAudioDecoderState,
    _AudioVAEFixedShapeStreamingDecoder,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.config import (
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.models.ming_tts.streaming_vocoder import _AudioVAEStreamingStateManager
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeDecoder:
    sample_rate = 44100
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls = 0

    def decode_nonstreaming(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        assert latents.shape == (0, 2, 3)
        self.calls += 1
        return torch.empty((0,), dtype=torch.float32)


class _RecordingDecoder:
    sample_rate = 44100

    def __init__(self, waveform: torch.Tensor) -> None:
        self.waveform = waveform
        self.calls: list[torch.Tensor] = []

    def decode_nonstreaming(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls.append(latents.clone())
        return self.waveform.clone()


class _FailingAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()))

    @property
    def decoder(self):
        raise AssertionError("empty latents should not call the AudioVAE decoder")


def test_ming_audio_decoder_skips_audio_vae_for_empty_latents() -> None:
    decoder = MingAudioDecoder(_FailingAudioVAE(), sample_rate=44100)

    waveform = decoder.decode_nonstreaming(torch.empty((0, 2, 3), dtype=torch.float32))

    assert waveform.shape == (0,)
    assert waveform.dtype == torch.float32


def _make_tiny_audio_decoder() -> MingAudioDecoder:
    backbone = {
        "_attn_implementation": "sdpa",
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "max_position_embeddings": 256,
        "max_window_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "sliding_window": 64,
        "use_cache": False,
        "use_sliding_window": True,
        "vocab_size": 1,
    }
    config = AudioVAEconfig(
        sample_rate=44100,
        enc_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 4},
            "input_dim": 4,
            "hop_size": 4,
            "latent_dim": 4,
        },
        dec_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 1},
            "output_dim": 4,
            "latent_dim": 4,
        },
        patch_size=4,
    )
    return MingAudioDecoder(AudioVAE(config).eval(), sample_rate=44100)


def _assert_incremental_matches_full_sequence(
    chunk_patches: tuple[int, ...],
) -> list[torch.Tensor]:
    assert chunk_patches and all(count > 0 for count in chunk_patches)
    num_patches = sum(chunk_patches)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        decoder = _make_tiny_audio_decoder()
        latents = torch.randn(num_patches, 4, 4)

        full = decoder.decode_nonstreaming(latents)
        state = MingAudioDecoderState()
        incremental_parts = []
        patch_start = 0
        for chunk_index, patch_count in enumerate(chunk_patches):
            patch_end = patch_start + patch_count
            incremental_parts.append(
                decoder.decode_streaming_step(
                    latents[patch_start:patch_end].flatten(0, 1),
                    state=state,
                    is_last=chunk_index == len(chunk_patches) - 1,
                )
            )
            patch_start = patch_end

    incremental = torch.cat(incremental_parts)
    assert full.numel() > 0
    assert incremental.shape == full.shape
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-6)
    return incremental_parts


def test_ming_audio_decoder_incremental_matches_full_sequence_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((1, 4, 2))

    assert incremental_parts[0].numel() == 0
    assert incremental_parts[-1].numel() > 0


def test_ming_audio_decoder_single_terminal_patch_matches_full_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((1,))

    assert incremental_parts[0].numel() > 0


def test_ming_audio_decoder_matches_full_after_window_saturation_on_cpu() -> None:
    # Eleven patches upsample to 176 decoder frames. With sliding_window=64,
    # the fifth non-terminal call runs after the cache has crossed the window.
    incremental_parts = _assert_incremental_matches_full_sequence((1, 2, 2, 2, 2, 2))

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_matches_full_at_shipped_cadence_on_cpu() -> None:
    initial = MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    steady = MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    incremental_parts = _assert_incremental_matches_full_sequence(
        (initial, steady, steady, steady)
    )

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_matches_full_when_initial_exceeds_steady() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((4, 2, 2))

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_short_terminal_group_matches_full_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence(
        (
            MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
            MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
            2,
        )
    )

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def _decode_fixed_c1_step(
    decoder: _AudioVAEFixedShapeStreamingDecoder,
    latents: torch.Tensor,
    *,
    terminal: bool,
    poison_invalid_tail: bool = False,
) -> tuple[torch.Tensor, int]:
    envelope = (
        torch.linspace(
            0.01,
            0.25,
            steps=decoder.max_step_latents * decoder.latent_dim,
            dtype=decoder.input_dtype,
            device=decoder.device,
        ).reshape(1, decoder.max_step_latents, decoder.latent_dim)
        if poison_invalid_tail
        else torch.zeros(
            (1, decoder.max_step_latents, decoder.latent_dim),
            dtype=decoder.input_dtype,
            device=decoder.device,
        )
    )
    envelope[0, : latents.shape[0]].copy_(latents)
    output = decoder.decode(
        envelope,
        torch.tensor([latents.shape[0]], device=decoder.device),
        torch.ones(1, dtype=torch.bool, device=decoder.device),
        torch.tensor([terminal], device=decoder.device),
    )
    return output.waveform[0].clone(), int(output.sample_lengths[0].item())


def _assert_fixed_row_matches(
    reference: _AudioVAEFixedShapeStreamingDecoder,
    candidate: _AudioVAEFixedShapeStreamingDecoder,
    slot: int,
) -> None:
    for (name, reference_tensor, reference_dim), (
        _,
        candidate_tensor,
        candidate_dim,
    ) in zip(
        reference._state.slot_tensors(),
        candidate._state.slot_tensors(),
        strict=True,
    ):
        reference_row = reference_tensor.select(reference_dim, 0)
        candidate_row = candidate_tensor.select(candidate_dim, slot)
        if reference_row.dtype.is_floating_point:
            torch.testing.assert_close(
                candidate_row,
                reference_row,
                rtol=1e-4,
                atol=1e-6,
                msg=name,
            )
        else:
            assert torch.equal(candidate_row, reference_row), name


def test_fixed_streaming_decoder_matches_dynamic_c1_k4_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        decoder = _make_tiny_audio_decoder()
        latents = torch.randn(11, 4, 4)

    dynamic_state = MingAudioDecoderState()
    fixed_zero = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=1,
        max_step_latents=16,
    )
    fixed_poison = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=1,
        max_step_latents=16,
    )

    patch_start = 0
    for step_index, (patch_count, terminal) in enumerate(
        ((1, False), (4, False), (4, False), (2, True))
    ):
        patch_end = patch_start + patch_count
        step_latents = latents[patch_start:patch_end].flatten(0, 1)
        patch_start = patch_end
        dynamic_waveform = decoder.decode_streaming_step(
            step_latents,
            state=dynamic_state,
            is_last=terminal,
        )
        fixed_waveform, fixed_length = _decode_fixed_c1_step(
            fixed_zero,
            step_latents,
            terminal=terminal,
        )
        poison_waveform, poison_length = _decode_fixed_c1_step(
            fixed_poison,
            step_latents,
            terminal=terminal,
            poison_invalid_tail=True,
        )

        assert fixed_length == poison_length == dynamic_waveform.numel()
        torch.testing.assert_close(
            fixed_waveform[:fixed_length],
            dynamic_waveform,
            rtol=1e-4,
            atol=1e-6,
        )
        assert torch.count_nonzero(fixed_waveform[fixed_length:]).item() == 0
        assert torch.equal(fixed_waveform, poison_waveform)
        for (name, zero_tensor, _), (_, poison_tensor, _) in zip(
            fixed_zero._state.slot_tensors(),
            fixed_poison._state.slot_tensors(),
            strict=True,
        ):
            assert torch.equal(zero_tensor, poison_tensor), name

        if terminal:
            fixed_zero.assert_rows_clean()
            fixed_poison.assert_rows_clean()
            continue

        dynamic_upsample = dynamic_state.upsample_state
        assert dynamic_upsample is not None
        assert dynamic_upsample["is_first"] is False
        fixed_state = fixed_zero._state
        dynamic_pending = dynamic_upsample["prev_chunk"]
        pending_length = dynamic_pending.shape[1]
        assert fixed_state.upsample_pending_lengths[0].item() == pending_length
        torch.testing.assert_close(
            fixed_state.upsample_pending[0, :pending_length],
            dynamic_pending[0],
            rtol=1e-4,
            atol=1e-6,
        )
        assert (
            torch.count_nonzero(fixed_state.upsample_pending[0, pending_length:]).item()
            == 0
        )

        dynamic_history = dynamic_upsample["history_last"]
        assert fixed_state.upsample_has_history[0].item() is (
            dynamic_history is not None
        )
        if dynamic_history is None:
            assert torch.count_nonzero(fixed_state.upsample_history).item() == 0
        else:
            torch.testing.assert_close(
                fixed_state.upsample_history,
                dynamic_history.transpose(1, 2),
                rtol=1e-4,
                atol=1e-6,
            )

        dynamic_cache = dynamic_state.dynamic_cache
        if dynamic_cache is None:
            assert fixed_state.qwen_lengths[0].item() == 0
            assert fixed_state.qwen_positions[0].item() == 0
            assert dynamic_state.audio_buffer is None
            assert dynamic_state.window_buffer is None
            assert fixed_state.istft_started[0].item() is False
        else:
            position = dynamic_cache.get_seq_length()
            dynamic_keys = torch.stack(
                [layer.keys[0] for layer in dynamic_cache.layers]
            )
            dynamic_values = torch.stack(
                [layer.values[0] for layer in dynamic_cache.layers]
            )
            stored_length = dynamic_keys.shape[-2]
            assert fixed_state.qwen_positions[0].item() == position
            assert fixed_state.qwen_lengths[0].item() == stored_length
            torch.testing.assert_close(
                fixed_state.qwen_keys[:, 0, :, -stored_length:],
                dynamic_keys,
                rtol=1e-4,
                atol=1e-6,
            )
            torch.testing.assert_close(
                fixed_state.qwen_values[:, 0, :, -stored_length:],
                dynamic_values,
                rtol=1e-4,
                atol=1e-6,
            )

            assert dynamic_state.audio_buffer is not None
            assert dynamic_state.window_buffer is not None
            assert fixed_state.istft_started[0].item() is True
            torch.testing.assert_close(
                fixed_state.istft_audio_overlap,
                dynamic_state.audio_buffer,
                rtol=1e-4,
                atol=1e-6,
            )
            torch.testing.assert_close(
                fixed_state.istft_window_overlap,
                dynamic_state.window_buffer,
                rtol=1e-4,
                atol=1e-6,
            )

        if step_index == 2:
            assert fixed_state.qwen_positions[0].item() == 80
            assert fixed_state.qwen_lengths[0].item() == 63

    assert patch_start == latents.shape[0]


def test_fixed_streaming_decoder_matches_direct_terminal_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        decoder = _make_tiny_audio_decoder()
        latents = torch.randn(4, 4)

    dynamic_waveform = decoder.decode_streaming_step(
        latents,
        state=MingAudioDecoderState(),
        is_last=True,
    )
    fixed_zero = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=1,
        max_step_latents=16,
    )
    fixed_poison = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=1,
        max_step_latents=16,
    )

    fixed_waveform, fixed_length = _decode_fixed_c1_step(
        fixed_zero,
        latents,
        terminal=True,
    )
    poison_waveform, poison_length = _decode_fixed_c1_step(
        fixed_poison,
        latents,
        terminal=True,
        poison_invalid_tail=True,
    )

    assert fixed_length == poison_length == dynamic_waveform.numel()
    torch.testing.assert_close(
        fixed_waveform[:fixed_length],
        dynamic_waveform,
        rtol=1e-4,
        atol=1e-6,
    )
    assert torch.count_nonzero(fixed_waveform[fixed_length:]).item() == 0
    assert torch.equal(fixed_waveform, poison_waveform)
    fixed_zero.assert_rows_clean()
    fixed_poison.assert_rows_clean()


def test_fixed_c8_matches_c1_and_isolates_inactive_rows_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        decoder = _make_tiny_audio_decoder()
        a_latents = torch.randn(11, 4, 4)
        b_latents = torch.randn(3, 4, 4)

    reference_by_slot = {
        0: _AudioVAEFixedShapeStreamingDecoder(
            decoder.audio_vae.decoder,
            capacity=1,
            max_step_latents=16,
        ),
        7: _AudioVAEFixedShapeStreamingDecoder(
            decoder.audio_vae.decoder,
            capacity=1,
            max_step_latents=16,
        ),
    }
    fixed_zero = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=8,
        max_step_latents=16,
    )
    fixed_poison = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=8,
        max_step_latents=16,
    )
    waves = (
        (
            (0, a_latents[:1].flatten(0, 1), False),
            (7, b_latents[:1].flatten(0, 1), False),
        ),
        ((0, a_latents[1:5].flatten(0, 1), False),),
        (
            (0, a_latents[5:9].flatten(0, 1), False),
            (7, b_latents[1:].flatten(0, 1), True),
        ),
        ((0, a_latents[9:].flatten(0, 1), True),),
    )

    for wave_index, wave in enumerate(waves):
        active_slots = {slot for slot, _, _ in wave}
        inactive_slots = torch.tensor(
            [slot for slot in range(8) if slot not in active_slots],
            dtype=torch.long,
        )
        inactive_before = tuple(
            (
                name,
                tensor.index_select(row_dim, inactive_slots).clone(),
            )
            for name, tensor, row_dim in fixed_zero._state.slot_tensors()
        )

        zero_latents = torch.zeros(
            (8, fixed_zero.max_step_latents, fixed_zero.latent_dim),
            dtype=fixed_zero.input_dtype,
        )
        poison_latents = torch.linspace(
            0.01,
            0.25,
            steps=8 * fixed_poison.max_step_latents * fixed_poison.latent_dim,
            dtype=fixed_poison.input_dtype,
        ).reshape(8, fixed_poison.max_step_latents, fixed_poison.latent_dim)
        zero_lengths = torch.zeros(8, dtype=torch.long)
        poison_lengths = torch.full(
            (8,), fixed_poison.max_step_latents, dtype=torch.long
        )
        zero_exec = torch.zeros(8, dtype=torch.bool)
        poison_exec = torch.zeros(8, dtype=torch.bool)
        zero_terminal = torch.zeros(8, dtype=torch.bool)
        poison_terminal = torch.ones(8, dtype=torch.bool)
        references = {}

        for slot, step_latents, terminal in wave:
            step_length = step_latents.shape[0]
            zero_latents[slot, :step_length].copy_(step_latents)
            poison_latents[slot].zero_()
            poison_latents[slot, :step_length].copy_(step_latents)
            zero_lengths[slot] = step_length
            poison_lengths[slot] = step_length
            zero_exec[slot] = True
            poison_exec[slot] = True
            zero_terminal[slot] = terminal
            poison_terminal[slot] = terminal
            references[slot] = _decode_fixed_c1_step(
                reference_by_slot[slot],
                step_latents,
                terminal=terminal,
            )

        zero_output = fixed_zero.decode(
            zero_latents,
            zero_lengths,
            zero_exec,
            zero_terminal,
        )
        zero_waveform = zero_output.waveform.clone()
        zero_sample_lengths = zero_output.sample_lengths.clone()
        poison_output = fixed_poison.decode(
            poison_latents,
            poison_lengths,
            poison_exec,
            poison_terminal,
        )
        poison_waveform = poison_output.waveform.clone()
        poison_sample_lengths = poison_output.sample_lengths.clone()

        assert torch.equal(zero_waveform, poison_waveform)
        assert torch.equal(zero_sample_lengths, poison_sample_lengths)
        for (name, zero_tensor, _), (_, poison_tensor, _) in zip(
            fixed_zero._state.slot_tensors(),
            fixed_poison._state.slot_tensors(),
            strict=True,
        ):
            assert torch.equal(zero_tensor, poison_tensor), name

        for slot, _, terminal in wave:
            reference_waveform, reference_length = references[slot]
            assert zero_sample_lengths[slot].item() == reference_length
            torch.testing.assert_close(
                zero_waveform[slot],
                reference_waveform,
                rtol=1e-4,
                atol=1e-6,
            )
            _assert_fixed_row_matches(
                reference_by_slot[slot],
                fixed_zero,
                slot,
            )
            if terminal:
                fixed_zero.assert_rows_clean((slot,))
                fixed_poison.assert_rows_clean((slot,))

        assert (
            torch.count_nonzero(
                zero_sample_lengths.index_select(0, inactive_slots)
            ).item()
            == 0
        )
        assert (
            torch.count_nonzero(zero_waveform.index_select(0, inactive_slots)).item()
            == 0
        )
        for (name, before), (_, tensor, row_dim) in zip(
            inactive_before,
            fixed_zero._state.slot_tensors(),
            strict=True,
        ):
            after = tensor.index_select(row_dim, inactive_slots)
            assert torch.equal(after, before), name

        if wave_index == 2:
            assert fixed_zero._state.qwen_positions[0].item() == 80
            assert fixed_zero._state.qwen_lengths[0].item() == 63

    fixed_zero.assert_rows_clean()
    fixed_poison.assert_rows_clean()


def test_audio_vae_state_manager_resets_before_reusing_slot_on_cpu() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        decoder = _make_tiny_audio_decoder()
        open_latents = torch.randn(2, 4, 4)
        terminal_latents = torch.randn(4, 4)

    fixed = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=8,
        max_step_latents=16,
    )
    manager = _AudioVAEStreamingStateManager(fixed)
    request_ids = ("a", "b", "d", "e", "f", "g", "h", "i")
    bindings = {}
    for request_id in request_ids:
        slot = manager.try_bind(request_id)
        assert slot is not None
        bindings[request_id] = slot

    assert set(bindings.values()) == set(range(8))
    assert manager.try_bind("overflow") is None
    a_slot = bindings["a"]
    b_slot = bindings["b"]

    envelope = torch.zeros(
        (8, fixed.max_step_latents, fixed.latent_dim),
        dtype=fixed.input_dtype,
    )
    lengths = torch.zeros(8, dtype=torch.long)
    exec_mask = torch.zeros(8, dtype=torch.bool)
    terminal_mask = torch.zeros(8, dtype=torch.bool)
    for slot, latents in ((a_slot, open_latents[0]), (b_slot, open_latents[1])):
        envelope[slot, : latents.shape[0]].copy_(latents)
        lengths[slot] = latents.shape[0]
        exec_mask[slot] = True
    fixed.decode(envelope, lengths, exec_mask, terminal_mask)

    b_state = tuple(
        (name, tensor.select(row_dim, b_slot).clone())
        for name, tensor, row_dim in fixed._state.slot_tensors()
    )
    manager.reset_and_release(("a",))

    assert manager.slot_for("a") is None
    assert manager.slot_for("b") == b_slot
    fixed.assert_rows_clean((a_slot,))
    for (name, before), (_, tensor, row_dim) in zip(
        b_state,
        fixed._state.slot_tensors(),
        strict=True,
    ):
        assert torch.equal(tensor.select(row_dim, b_slot), before), name

    reused_slot = manager.try_bind("c")
    assert reused_slot == a_slot
    fresh = _AudioVAEFixedShapeStreamingDecoder(
        decoder.audio_vae.decoder,
        capacity=1,
        max_step_latents=16,
    )
    reference_waveform, reference_length = _decode_fixed_c1_step(
        fresh,
        terminal_latents,
        terminal=True,
    )

    envelope.zero_()
    lengths.zero_()
    exec_mask.zero_()
    terminal_mask.zero_()
    envelope[reused_slot, : terminal_latents.shape[0]].copy_(terminal_latents)
    lengths[reused_slot] = terminal_latents.shape[0]
    exec_mask[reused_slot] = True
    terminal_mask[reused_slot] = True
    output = fixed.decode(envelope, lengths, exec_mask, terminal_mask)

    assert output.sample_lengths[reused_slot].item() == reference_length
    torch.testing.assert_close(
        output.waveform[reused_slot],
        reference_waveform,
        rtol=1e-4,
        atol=1e-6,
    )
    _assert_fixed_row_matches(fresh, fixed, reused_slot)
    fixed.assert_rows_clean((reused_slot,))
    for (name, before), (_, tensor, row_dim) in zip(
        b_state,
        fixed._state.slot_tensors(),
        strict=True,
    ):
        assert torch.equal(tensor.select(row_dim, b_slot), before), name

    manager.reset_all_and_assert_clean()
    assert all(
        manager.slot_for(request_id) is None for request_id in (*request_ids, "c")
    )


@pytest.mark.parametrize("keep_latents", [False, True])
def test_ming_tts_nonstreaming_payload_decodes_full_sequence_once(
    keep_latents: bool,
) -> None:
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=2,
        generated_latents=latents,
    )
    payload = StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    waveform = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32)
    decoder = _RecordingDecoder(waveform)

    result = decode_ming_tts_audio_payload(
        payload,
        decoder,
        keep_latents=keep_latents,
    )

    assert len(decoder.calls) == 1
    torch.testing.assert_close(decoder.calls[0], latents)
    restored = MingTTSState.from_dict(result.data)
    if keep_latents:
        torch.testing.assert_close(restored.generated_latents, latents)
    else:
        assert restored.generated_latents is None
    assert restored.sample_rate == 44100
    assert restored.duration_s == pytest.approx(waveform.numel() / 44100)
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(audio, waveform.numpy())


def test_ming_tts_audio_decode_accepts_empty_generated_latents() -> None:
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=0,
        generated_latents=torch.empty((0, 2, 3), dtype=torch.float32),
    )
    payload = StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    decoder = _FakeDecoder()

    result = decode_ming_tts_audio_payload(payload, decoder)

    assert decoder.calls == 1
    assert result.data["sample_rate"] == 44100
    assert result.data["duration_s"] == 0.0
    assert result.data["audio_waveform_shape"] == [0]
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == []
