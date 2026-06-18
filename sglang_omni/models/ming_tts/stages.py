# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Ming-Omni-TTS 16.8B pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

from sglang_omni.models.ming_tts.hf_config import (
    MING_TTS_MODEL_ARCH_OVERRIDE,
    register_ming_tts_hf_config,
)
from sglang_omni.models.ming_tts.request_builders import preprocess_ming_tts_payload
from sglang_omni.models.ming_tts.tokenizer import load_ming_tts_tokenizer
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)


def create_preprocessing_executor(
    model_path: str,
    *,
    context_length: int | None = None,
    max_decode_steps_cap: int | None = None,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    context_length = int(context_length or _resolve_context_length(config))
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )

    def _preprocess(payload):
        return preprocess_ming_tts_payload(
            payload,
            tokenizer=tokenizer,
            context_length=context_length,
            max_decode_steps_cap=max_decode_steps_cap,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    context_length: int | None = None,
    server_args_overrides: dict[str, Any] | None = None,
    total_gpu_memory_fraction: float | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
) -> Any:
    if int(tp_size) != 1 or int(tp_rank) != 0:
        raise ValueError("Ming-Omni-TTS currently supports only tp_size=1")

    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner
    from sglang_omni.models.ming_tts.sglang_request_builders import (
        make_ming_tts_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    user_overrides = dict(server_args_overrides or {})
    overrides: dict[str, Any] = {
        "dtype": dtype,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_running_requests": 8,
        "sampling_backend": "pytorch",
        "trust_remote_code": False,
    }
    overrides.update(user_overrides)

    context_length = int(overrides.pop("context_length", context_length or 0) or 0)
    if context_length <= 0:
        context_length = _resolve_context_length(config)
    if "max_prefill_tokens" not in user_overrides:
        overrides["max_prefill_tokens"] = min(int(context_length), 8192)
    if not bool(overrides.get("disable_cuda_graph", True)):
        raise ValueError("Ming-Omni-TTS currently requires disable_cuda_graph=True")
    if not bool(overrides.get("disable_overlap_schedule", True)):
        raise ValueError(
            "Ming-Omni-TTS currently requires disable_overlap_schedule=True"
        )
    if bool(overrides.get("enable_torch_compile", False)):
        raise ValueError("Ming-Omni-TTS currently requires enable_torch_compile=False")

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=int(context_length),
        **overrides,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override=MING_TTS_MODEL_ARCH_OVERRIDE,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    model = model_worker.model_runner.model
    model.eval()
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )
    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_ming_tts_scheduler_adapters(
        model=model,
        tokenizer=tokenizer,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MingTTSModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    context_length: int | None = None,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    from sglang_omni.models.ming_tts.reference_encode import MingTTSReferenceEncoder
    from sglang_omni.models.ming_tts.weight_loading import (
        load_ming_tts_audio_vae_weights,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    context_length = int(context_length or _resolve_context_length(config))
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    encoder = MingTTSReferenceEncoder.from_config(
        config.audio_tokenizer_config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        dtype=dtype,
        patch_size=int(config.ditar_config["patch_size"]),
    )
    report = load_ming_tts_audio_vae_weights(checkpoint_dir, encoder.audio_vae)
    logger.info("%s", report.summary())

    def _encode(payload):
        return encoder.encode_payload(
            payload,
            tokenizer=tokenizer,
            context_length=context_length,
        )

    return SimpleScheduler(_encode, max_concurrency=max_concurrency)


def create_audio_decode_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    decode_mode: str = "chunked",
    keep_latents: bool = False,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 0,
) -> SimpleScheduler:
    if decode_mode != "chunked":
        raise ValueError("Ming-Omni-TTS currently supports only decode_mode='chunked'")

    from sglang_omni.models.ming_tts.audio_decode import (
        MingAudioDecoder,
        decode_ming_tts_audio_payload,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    decoder = MingAudioDecoder.from_config(
        config.audio_tokenizer_config,
        device=device,
        dtype=dtype,
    )
    try:
        from sglang_omni.models.ming_tts.weight_loading import (
            load_ming_tts_audio_vae_weights,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ming-Omni-TTS audio_decode stage requires weight_loading.py to "
            "consume the checkpoint's audio.* tensors."
        ) from exc
    report = load_ming_tts_audio_vae_weights(checkpoint_dir, decoder.audio_vae)
    logger.info("%s", report.summary())

    def _decode(payload):
        return decode_ming_tts_audio_payload(
            payload,
            decoder,
            decode_mode=decode_mode,
            keep_latents=keep_latents,
        )

    return SimpleScheduler(
        _decode,
        batch_compute_fn=lambda payloads: [_decode(payload) for payload in payloads],
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_ming_tts_config(model_path: str) -> Any:
    register_ming_tts_hf_config()
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=False)


def _resolve_context_length(config: Any) -> int:
    llm_config = config.llm_config
    value = getattr(llm_config, "max_position_embeddings", None)
    if value is None:
        raise ValueError("Ming-Omni-TTS llm_config is missing max_position_embeddings")
    return int(value)


__all__ = [
    "create_audio_decode_executor",
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_sglang_tts_engine_executor",
    "create_tts_engine_executor",
]
