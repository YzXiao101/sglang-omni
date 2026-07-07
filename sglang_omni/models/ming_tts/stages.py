# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Ming-Omni-TTS 16.8B pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.ming_tts.audio_config import resolve_ming_tts_audio_vae_config
from sglang_omni.models.ming_tts.hf_config import (
    MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    MING_TTS_MODEL_ARCH_OVERRIDE,
    register_ming_tts_hf_config,
)
from sglang_omni.models.ming_tts.request_builders import preprocess_ming_tts_payload
from sglang_omni.models.ming_tts.tokenizer import load_ming_tts_tokenizer
from sglang_omni.models.ming_tts.weight_loading import load_ming_tts_audio_vae_weights
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MingTTSEngineStartup:
    checkpoint_dir: str
    config: Any
    gpu_id: int
    dtype: str
    context_length: int | None
    user_overrides: dict[str, Any]
    total_gpu_memory_fraction: float | None
    tp_rank: int
    tp_size: int
    nccl_port: int | None


def _check_ming_tts_tp_backbone_config(config: Any, tp_size: int) -> None:
    tp_size = int(tp_size)
    if tp_size <= 1:
        return

    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise ValueError("Ming-Omni-TTS TP requires llm_config")

    def require_int_config_field(field: str) -> int:
        value = getattr(llm_config, field, None)
        if value is None:
            raise ValueError(f"Ming-Omni-TTS llm_config is missing {field}")
        return int(value)

    hidden_size = require_int_config_field("hidden_size")
    head_dim = require_int_config_field("head_dim")
    num_heads = require_int_config_field("num_attention_heads")
    num_kv_heads = require_int_config_field("num_key_value_heads")
    if min(hidden_size, head_dim, num_heads, num_kv_heads) <= 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires positive hidden/head dimensions: "
            f"hidden_size={hidden_size}, head_dim={head_dim}, "
            f"num_attention_heads={num_heads}, "
            f"num_key_value_heads={num_kv_heads}"
        )
    if head_dim * num_heads != hidden_size:
        raise ValueError(
            "Ming-Omni-TTS TP requires head_dim * num_attention_heads "
            f"to equal hidden_size ({head_dim} * {num_heads} != {hidden_size})"
        )
    if hidden_size % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires hidden_size divisible by tp_size: "
            f"hidden_size={hidden_size}, tp_size={tp_size}"
        )
    if num_heads % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires attention heads divisible by tp_size: "
            f"num_attention_heads={num_heads}, tp_size={tp_size}"
        )
    if num_kv_heads >= tp_size and num_kv_heads % tp_size != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires KV heads divisible by tp_size: "
            f"num_key_value_heads={num_kv_heads}, tp_size={tp_size}"
        )
    if num_kv_heads < tp_size and tp_size % num_kv_heads != 0:
        raise ValueError(
            "Ming-Omni-TTS TP requires KV heads to divide or be divisible "
            f"by tp_size: num_key_value_heads={num_kv_heads}, "
            f"tp_size={tp_size}"
        )


def _resolve_ming_tts_engine_startup(
    model_path: str,
    *,
    device: str,
    gpu_id: int | None,
    dtype: str,
    context_length: int | None,
    server_args_overrides: dict[str, Any] | None,
    total_gpu_memory_fraction: float | None,
    tp_rank: int,
    tp_size: int,
    nccl_port: int | None,
) -> MingTTSEngineStartup:
    tp_rank = int(tp_rank)
    tp_size = int(tp_size)
    if tp_size not in (1, 2):
        raise ValueError(
            "Ming-Omni-TTS tts_engine currently supports only tp_size=1 or "
            f"tp_size=2; got tp_size={tp_size}"
        )
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(
            f"Ming-Omni-TTS tts_engine tp_rank={tp_rank} is out of range "
            f"for tp_size={tp_size}"
        )
    if tp_size > 1 and nccl_port is None:
        raise ValueError("Ming-Omni-TTS tts_engine TP requires nccl_port")

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    _check_ming_tts_tp_backbone_config(config, tp_size)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    user_overrides = dict(server_args_overrides or {})
    if "tp_size" in user_overrides and int(user_overrides["tp_size"]) != tp_size:
        raise ValueError(
            "Ming-Omni-TTS tts_engine tp_size conflicts with "
            f"server_args_overrides.tp_size={user_overrides['tp_size']!r}"
        )

    return MingTTSEngineStartup(
        checkpoint_dir=checkpoint_dir,
        config=config,
        gpu_id=gpu_id,
        dtype=dtype,
        context_length=context_length,
        user_overrides=user_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    )


def _build_ming_tts_server_args(startup: MingTTSEngineStartup) -> Any:
    from sglang_omni.scheduling.generation_batch_policy import (
        build_generation_batch_overrides,
    )
    from sglang_omni.scheduling.sglang_backend import build_sglang_server_args

    def coerce_bool(value: Any, *, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(f"{name} must be a boolean value, got {value!r}")

    user_overrides = startup.user_overrides
    context_length = int(
        user_overrides.get("context_length", startup.context_length or 0) or 0
    )
    if context_length <= 0:
        context_length = _resolve_context_length(startup.config)

    overrides = build_generation_batch_overrides(
        max_running_requests=8,
        server_args_overrides=user_overrides,
        dtype=startup.dtype,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        enable_torch_compile=False,
        max_prefill_tokens=min(int(context_length), 8192),
        sampling_backend="pytorch",
        trust_remote_code=False,
    )
    overrides.pop("context_length", None)
    overrides["tp_size"] = startup.tp_size

    radix_cache_enabled = not coerce_bool(
        overrides.get("disable_radix_cache", True),
        name="server_args_overrides.disable_radix_cache",
    )
    graph_enabled = not coerce_bool(
        overrides.get("disable_cuda_graph", True),
        name="server_args_overrides.disable_cuda_graph",
    )
    if graph_enabled or radix_cache_enabled:
        if (
            "chunked_prefill_size" in user_overrides
            and int(overrides.get("chunked_prefill_size") or 0) != 0
        ):
            raise ValueError(
                "Ming-Omni-TTS graph/cache requires chunked_prefill_size=0 "
                "because generated continuous state does not have chunk "
                "rollback semantics"
            )
        if "chunked_prefill_size" not in user_overrides:
            overrides["chunked_prefill_size"] = 0
    if bool(overrides.get("enable_torch_compile", False)):
        raise ValueError("Ming-Omni-TTS torch.compile is not currently supported")

    server_args = build_sglang_server_args(
        startup.checkpoint_dir,
        context_length=int(context_length),
        **overrides,
    )
    return server_args


def _create_ming_tts_sglang_infra(
    startup: MingTTSEngineStartup,
    server_args: Any,
):
    from sglang_omni.scheduling.bootstrap import (
        create_sglang_infrastructure_defer_cuda_graph,
    )

    requested_disable_cuda_graph = bool(
        getattr(server_args, "disable_cuda_graph", False)
    )
    graph_capture_deferred = not requested_disable_cuda_graph
    tail_cuda_graph_enabled = bool(graph_capture_deferred and startup.tp_rank == 0)

    logger.info(
        "Ming AR SGLang startup: gpu_id=%s tp_rank=%s/%s pid=%s "
        "total_gpu_memory_fraction=%s ar_cuda_graph_requested=%s "
        "disable_cuda_graph=%s graph_capture_deferred=%s "
        "cuda_graph_bs=%s cuda_graph_max_bs=%s enable_torch_compile=%s "
        "torch_compile_max_bs=%s radix_cache=%s tail_cuda_graph=%s nccl_port=%s",
        startup.gpu_id,
        startup.tp_rank,
        startup.tp_size,
        os.getpid(),
        startup.total_gpu_memory_fraction,
        graph_capture_deferred,
        requested_disable_cuda_graph,
        graph_capture_deferred,
        getattr(server_args, "cuda_graph_bs", None),
        getattr(server_args, "cuda_graph_max_bs", None),
        getattr(server_args, "enable_torch_compile", None),
        getattr(server_args, "torch_compile_max_bs", None),
        not bool(getattr(server_args, "disable_radix_cache", True)),
        tail_cuda_graph_enabled,
        startup.nccl_port,
    )
    return create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        startup.gpu_id,
        model_arch_override=MING_TTS_MODEL_ARCH_OVERRIDE,
        tp_rank=startup.tp_rank,
        nccl_port=startup.nccl_port,
        total_gpu_memory_fraction=startup.total_gpu_memory_fraction,
    )


def _finish_ming_tts_graph_startup(
    startup: MingTTSEngineStartup,
    server_args: Any,
    *,
    want_cuda_graph: bool,
    model_worker: Any,
) -> None:
    from sglang_omni.scheduling.generation_batch_policy import (
        validate_generation_batch_policy,
    )

    model = model_worker.model_runner.model
    model.eval()
    validate_generation_batch_policy(
        model_name="Ming-Omni-TTS",
        server_args=server_args,
        model_buffer_bs=int(model._decode_input_embedding.num_embeddings),
    )
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
        if startup.tp_rank == 0:
            model.init_tail_graphs(
                list(model_worker.model_runner.graph_runner.capture_bs)
            )

    ming_ar_text_model = getattr(model, "model", None)
    ming_ar_layers = getattr(ming_ar_text_model, "layers", None)
    layer_count = len(ming_ar_layers) if ming_ar_layers is not None else None
    tail_cuda_graph_enabled = bool(want_cuda_graph and startup.tp_rank == 0)
    tail_attn_backend = getattr(model, "tail_attn_backend", None)
    logger.info(
        "Ming AR CUDA graph startup: enabled=%s target=%s layer_count=%s "
        "gpu_id=%s tp_rank=%s/%s pid=%s total_gpu_memory_fraction=%s "
        "max_running_requests=%s cuda_graph_bs=%s "
        "enable_torch_compile=%s torch_compile_max_bs=%s "
        "disable_radix_cache=%s chunked_prefill_size=%s "
        "radix_cache=%s tail_cuda_graph=%s "
        "tail_attn_backend=%s",
        bool(want_cuda_graph),
        "MingTTSSGLangModel.forward via SGLang cuda graph",
        layer_count,
        startup.gpu_id,
        startup.tp_rank,
        startup.tp_size,
        os.getpid(),
        startup.total_gpu_memory_fraction,
        getattr(server_args, "max_running_requests", None),
        getattr(server_args, "cuda_graph_bs", None),
        getattr(server_args, "enable_torch_compile", None),
        getattr(server_args, "torch_compile_max_bs", None),
        getattr(server_args, "disable_radix_cache", None),
        getattr(server_args, "chunked_prefill_size", None),
        not bool(getattr(server_args, "disable_radix_cache", True)),
        tail_cuda_graph_enabled,
        tail_attn_backend,
    )


def _build_ming_tts_omni_scheduler(
    startup: MingTTSEngineStartup,
    server_args: Any,
    infra: tuple[Any, Any, Any, Any, Any, Any, Any],
) -> Any:
    from sglang_omni.models.ming_tts.engine_io import make_ming_tts_scheduler_adapters
    from sglang_omni.models.ming_tts.model_runner import MingTTSModelRunner
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = infra
    model = model_worker.model_runner.model
    tokenizer = load_ming_tts_tokenizer(
        startup.checkpoint_dir,
        llm_config=startup.config.llm_config,
    )
    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_ming_tts_scheduler_adapters(
        model=model,
        tokenizer=tokenizer,
        radix_cache_enabled=not bool(getattr(server_args, "disable_radix_cache", True)),
        model_cache_identity=str(startup.checkpoint_dir),
        owns_acoustic_result=startup.tp_rank == 0,
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
    startup = _resolve_ming_tts_engine_startup(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        context_length=context_length,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    )
    server_args = _build_ming_tts_server_args(startup)
    want_cuda_graph, infra = _create_ming_tts_sglang_infra(startup, server_args)
    _finish_ming_tts_graph_startup(
        startup,
        server_args,
        want_cuda_graph=want_cuda_graph,
        model_worker=infra[0],
    )
    return _build_ming_tts_omni_scheduler(
        startup,
        server_args,
        infra,
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

    checkpoint_dir = _resolve_checkpoint(model_path)
    config = _load_ming_tts_config(checkpoint_dir)
    context_length = int(context_length or _resolve_context_length(config))
    tokenizer = load_ming_tts_tokenizer(
        checkpoint_dir,
        llm_config=config.llm_config,
    )
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    audio_config = resolve_ming_tts_audio_vae_config(
        config.audio_tokenizer_config,
        attn_implementation=MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    )
    encoder = MingTTSReferenceEncoder.from_config(
        audio_config,
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

    audio_config = resolve_ming_tts_audio_vae_config(
        config.audio_tokenizer_config,
        attn_implementation=MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION,
    )
    decoder = MingAudioDecoder.from_config(
        audio_config,
        device=device,
        dtype=dtype,
    )
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
