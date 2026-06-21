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

MING_AR_CUDA_GRAPH_TARGET = "MingTTSSGLangModel.forward via SGLang cuda graph"
MING_AR_COMPILE_TARGET = MING_AR_CUDA_GRAPH_TARGET


def _require_int_config_field(config: Any, field: str) -> int:
    value = getattr(config, field, None)
    if value is None:
        raise ValueError(f"Ming-Omni-TTS llm_config is missing {field}")
    return int(value)


def _validate_ming_tts_tp_backbone_invariants(config: Any, tp_size: int) -> None:
    tp_size = int(tp_size)
    if tp_size <= 1:
        return

    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise ValueError("Ming-Omni-TTS TP requires llm_config")

    hidden_size = _require_int_config_field(llm_config, "hidden_size")
    head_dim = _require_int_config_field(llm_config, "head_dim")
    num_heads = _require_int_config_field(llm_config, "num_attention_heads")
    num_kv_heads = _require_int_config_field(llm_config, "num_key_value_heads")
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


def _coerce_bool(value: Any, *, name: str) -> bool:
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
    enable_ming_ar_cuda_graph: bool = False,
    enable_ming_ar_backbone_compile: bool = False,
    enable_ming_ar_sglang_compile: bool | None = None,
    enable_ming_ar_projected_prefill_radix_cache: bool = False,
    ming_ar_compile_mode: str | None = "max-autotune-no-cudagraphs",
    total_gpu_memory_fraction: float | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
) -> Any:
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
    _validate_ming_tts_tp_backbone_invariants(config, tp_size)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    user_overrides = dict(server_args_overrides or {})
    cuda_graph_settings: list[tuple[str, bool]] = []
    legacy_compile_alias_enabled = False
    explicit_cuda_graph_arg = _coerce_bool(
        enable_ming_ar_cuda_graph,
        name="enable_ming_ar_cuda_graph",
    )
    if explicit_cuda_graph_arg:
        cuda_graph_settings.append(("enable_ming_ar_cuda_graph", True))
    legacy_compile_arg = _coerce_bool(
        enable_ming_ar_backbone_compile,
        name="enable_ming_ar_backbone_compile",
    )
    if legacy_compile_arg:
        legacy_compile_alias_enabled = True
        cuda_graph_settings.append(("enable_ming_ar_backbone_compile", True))
    if enable_ming_ar_sglang_compile is not None:
        sglang_compile_arg = _coerce_bool(
            enable_ming_ar_sglang_compile,
            name="enable_ming_ar_sglang_compile",
        )
        if sglang_compile_arg:
            legacy_compile_alias_enabled = True
            cuda_graph_settings.append(("enable_ming_ar_sglang_compile", True))
    for key in (
        "enable_ming_ar_cuda_graph",
        "enable_ming_ar_backbone_compile",
        "enable_ming_ar_sglang_compile",
    ):
        if key in user_overrides:
            value = _coerce_bool(user_overrides.pop(key), name=key)
            if key == "enable_ming_ar_cuda_graph":
                cuda_graph_settings.append((f"server_args_overrides.{key}", value))
            elif value:
                legacy_compile_alias_enabled = True
                cuda_graph_settings.append((f"server_args_overrides.{key}", True))
    projected_prefill_radix_cache_enabled = _coerce_bool(
        enable_ming_ar_projected_prefill_radix_cache,
        name="enable_ming_ar_projected_prefill_radix_cache",
    )
    if "enable_ming_ar_projected_prefill_radix_cache" in user_overrides:
        projected_prefill_radix_cache_enabled = _coerce_bool(
            user_overrides.pop("enable_ming_ar_projected_prefill_radix_cache"),
            name="server_args_overrides.enable_ming_ar_projected_prefill_radix_cache",
        )
    cuda_graph_values = {value for _, value in cuda_graph_settings}
    if len(cuda_graph_values) > 1:
        detail = ", ".join(f"{name}={value}" for name, value in cuda_graph_settings)
        raise ValueError(f"Conflicting Ming AR CUDA graph settings: {detail}")
    enable_ming_ar_cuda_graph = any(value for _, value in cuda_graph_settings)
    if tp_size > 1:
        if not enable_ming_ar_cuda_graph:
            raise ValueError(
                "Ming-Omni-TTS TP2 currently requires " "enable_ming_ar_cuda_graph=True"
            )
        if legacy_compile_alias_enabled:
            raise ValueError(
                "Ming-Omni-TTS TP2 currently supports graph-only execution; "
                "use enable_ming_ar_cuda_graph=True instead of legacy "
                "compile aliases"
            )
        if projected_prefill_radix_cache_enabled:
            raise ValueError(
                "Ming-Omni-TTS TP2 currently requires projected-prefill "
                "radix cache disabled"
            )
    if "ming_ar_compile_mode" in user_overrides:
        ming_ar_compile_mode = user_overrides.pop("ming_ar_compile_mode")
    if ming_ar_compile_mode is not None:
        ming_ar_compile_mode = str(ming_ar_compile_mode).strip()
    if "tp_size" in user_overrides and int(user_overrides["tp_size"]) != tp_size:
        raise ValueError(
            "Ming-Omni-TTS tts_engine tp_size conflicts with "
            f"server_args_overrides.tp_size={user_overrides['tp_size']!r}"
        )

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "enable_torch_compile": False,
        "max_running_requests": 8,
        "sampling_backend": "pytorch",
        "trust_remote_code": False,
    }
    overrides.update(user_overrides)
    overrides["tp_size"] = tp_size

    context_length = int(overrides.pop("context_length", context_length or 0) or 0)
    if context_length <= 0:
        context_length = _resolve_context_length(config)
    if "max_prefill_tokens" not in user_overrides:
        overrides["max_prefill_tokens"] = min(int(context_length), 8192)
    if not bool(overrides.get("disable_overlap_schedule", True)):
        raise ValueError(
            "Ming-Omni-TTS currently requires disable_overlap_schedule=True"
        )
    if (
        not projected_prefill_radix_cache_enabled
        and "disable_radix_cache" in user_overrides
    ):
        radix_disabled = _coerce_bool(
            overrides.get("disable_radix_cache"),
            name="server_args_overrides.disable_radix_cache",
        )
        if not radix_disabled:
            raise ValueError(
                "Ming-Omni-TTS requires disable_radix_cache=True unless "
                "enable_ming_ar_projected_prefill_radix_cache=True"
            )
    if projected_prefill_radix_cache_enabled:
        overrides["disable_overlap_schedule"] = True
        if (
            "chunked_prefill_size" in user_overrides
            and int(overrides.get("chunked_prefill_size") or 0) != 0
        ):
            raise ValueError(
                "Ming projected prefill radix cache requires "
                "chunked_prefill_size=0 because generated continuous state does "
                "not have chunk rollback semantics"
            )
        overrides["chunked_prefill_size"] = 0
        if "disable_radix_cache" in user_overrides:
            radix_disabled = _coerce_bool(
                overrides.get("disable_radix_cache"),
                name="server_args_overrides.disable_radix_cache",
            )
            if radix_disabled:
                raise ValueError(
                    "enable_ming_ar_projected_prefill_radix_cache=True requires "
                    "disable_radix_cache=False"
                )
        overrides["disable_radix_cache"] = False
    if enable_ming_ar_cuda_graph:
        overrides["disable_cuda_graph"] = False
        overrides["disable_overlap_schedule"] = True
        if "disable_radix_cache" in user_overrides:
            radix_disabled = _coerce_bool(
                overrides.get("disable_radix_cache"),
                name="server_args_overrides.disable_radix_cache",
            )
            if not radix_disabled and not projected_prefill_radix_cache_enabled:
                raise ValueError(
                    "Ming AR CUDA graph requires disable_radix_cache=True unless "
                    "enable_ming_ar_projected_prefill_radix_cache=True"
                )
        overrides["disable_radix_cache"] = not projected_prefill_radix_cache_enabled
        chunked_prefill_size = overrides.get("chunked_prefill_size", 0)
        if (
            "chunked_prefill_size" in user_overrides
            and int(chunked_prefill_size or 0) != 0
        ):
            raise ValueError(
                "Ming AR CUDA graph requires chunked_prefill_size=0 because "
                "projected prefill and feedback state do not yet have chunk "
                "rollback semantics"
            )
        overrides["chunked_prefill_size"] = 0
        if "cuda_graph_bs" not in user_overrides:
            overrides["cuda_graph_bs"] = [1]
        if "cuda_graph_max_bs" not in user_overrides:
            cuda_graph_bs = overrides.get("cuda_graph_bs")
            if isinstance(cuda_graph_bs, (list, tuple)) and cuda_graph_bs:
                overrides["cuda_graph_max_bs"] = int(max(cuda_graph_bs))
            else:
                overrides["cuda_graph_max_bs"] = 1
        cuda_graph_max_bs = int(overrides.get("cuda_graph_max_bs") or 1)
        if "max_running_requests" not in user_overrides:
            overrides["max_running_requests"] = cuda_graph_max_bs
        max_running_requests = int(overrides.get("max_running_requests") or 0)
        if max_running_requests < cuda_graph_max_bs:
            raise ValueError(
                "Ming AR CUDA graph requires max_running_requests >= "
                f"cuda_graph_max_bs ({max_running_requests} < {cuda_graph_max_bs}) "
                "so the fixed decode feedback buffer has one row per captured "
                "request slot"
            )
        if "torch_compile_max_bs" not in user_overrides:
            overrides["torch_compile_max_bs"] = 1 if legacy_compile_alias_enabled else 0
        torch_compile_max_bs = int(overrides.get("torch_compile_max_bs") or 0)
        if torch_compile_max_bs < 0:
            raise ValueError(
                "Ming AR CUDA graph requires torch_compile_max_bs >= 0; "
                "use 0 only for CUDA graph-only isolation"
            )
        torch_compile_enabled = torch_compile_max_bs > 0
        if tp_size > 1 and torch_compile_enabled:
            raise ValueError(
                "Ming-Omni-TTS TP2 currently supports graph-only execution; "
                "set torch_compile_max_bs=0"
            )
        overrides["enable_torch_compile"] = torch_compile_enabled
        if torch_compile_enabled:
            if not ming_ar_compile_mode:
                raise ValueError(
                    "ming_ar_compile_mode must be non-empty when "
                    "Ming AR torch.compile is enabled"
                )
            os.environ["SGLANG_TORCH_COMPILE_MODE"] = ming_ar_compile_mode
        if bool(overrides.get("disable_cuda_graph_padding", False)):
            raise ValueError(
                "Ming AR CUDA graph is incompatible with "
                "disable_cuda_graph_padding=True"
            )
        if tp_size > 1 and max_running_requests != cuda_graph_max_bs:
            raise ValueError(
                "Ming-Omni-TTS TP2 graph-only requires max_running_requests "
                "to equal cuda_graph_max_bs so every runnable batch is covered "
                f"by a captured graph bucket ({max_running_requests} != "
                f"{cuda_graph_max_bs})"
            )
    else:
        if not bool(overrides.get("disable_cuda_graph", True)):
            raise ValueError(
                "Ming-Omni-TTS currently requires disable_cuda_graph=True unless "
                "Ming AR CUDA graph is enabled"
            )
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "Use enable_ming_ar_cuda_graph=True with torch_compile_max_bs > 0 "
                "instead of setting enable_torch_compile directly for "
                "Ming-Omni-TTS"
            )

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=int(context_length),
        **overrides,
    )
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    logger.info(
        "Ming AR SGLang startup: gpu_id=%s tp_rank=%s/%s pid=%s "
        "total_gpu_memory_fraction=%s disable_cuda_graph=%s "
        "cuda_graph_bs=%s cuda_graph_max_bs=%s enable_torch_compile=%s "
        "torch_compile_max_bs=%s projected_prefill_radix_cache=%s "
        "nccl_port=%s",
        gpu_id,
        tp_rank,
        tp_size,
        os.getpid(),
        total_gpu_memory_fraction,
        getattr(server_args, "disable_cuda_graph", None),
        getattr(server_args, "cuda_graph_bs", None),
        getattr(server_args, "cuda_graph_max_bs", None),
        getattr(server_args, "enable_torch_compile", None),
        getattr(server_args, "torch_compile_max_bs", None),
        projected_prefill_radix_cache_enabled,
        nccl_port,
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
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )

    model = model_worker.model_runner.model
    model.eval()
    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()
    ming_ar_text_model = getattr(model, "model", None)
    ming_ar_layers = getattr(ming_ar_text_model, "layers", None)
    ming_ar_cuda_graph_info: dict[str, Any] = {
        "enabled": bool(enable_ming_ar_cuda_graph),
        "target": MING_AR_CUDA_GRAPH_TARGET,
        "layer_count": len(ming_ar_layers) if ming_ar_layers is not None else None,
        "compile_mode": ming_ar_compile_mode,
        "compile_setup_time_s": 0.0,
        "cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "legacy_compile_alias": bool(legacy_compile_alias_enabled),
        "projected_prefill_radix_cache_enabled": bool(
            projected_prefill_radix_cache_enabled
        ),
    }
    logger.info(
        "Ming AR CUDA graph startup: enabled=%s target=%s layer_count=%s "
        "mode=%s compile_setup_time_s=%s cache_dir=%s legacy_compile_alias=%s "
        "gpu_id=%s tp_rank=%s/%s pid=%s total_gpu_memory_fraction=%s "
        "max_running_requests=%s cuda_graph_bs=%s "
        "enable_torch_compile=%s torch_compile_max_bs=%s "
        "disable_radix_cache=%s chunked_prefill_size=%s "
        "projected_prefill_radix_cache=%s",
        ming_ar_cuda_graph_info["enabled"],
        ming_ar_cuda_graph_info["target"],
        ming_ar_cuda_graph_info["layer_count"],
        ming_ar_cuda_graph_info["compile_mode"],
        ming_ar_cuda_graph_info["compile_setup_time_s"],
        ming_ar_cuda_graph_info["cache_dir"],
        ming_ar_cuda_graph_info["legacy_compile_alias"],
        gpu_id,
        tp_rank,
        tp_size,
        os.getpid(),
        total_gpu_memory_fraction,
        getattr(server_args, "max_running_requests", None),
        getattr(server_args, "cuda_graph_bs", None),
        getattr(server_args, "enable_torch_compile", None),
        getattr(server_args, "torch_compile_max_bs", None),
        getattr(server_args, "disable_radix_cache", None),
        getattr(server_args, "chunked_prefill_size", None),
        ming_ar_cuda_graph_info["projected_prefill_radix_cache_enabled"],
    )
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
        projected_prefill_requires_radix_disabled=bool(enable_ming_ar_cuda_graph),
        radix_cache_disabled=bool(getattr(server_args, "disable_radix_cache", False)),
        projected_prefill_radix_cache_enabled=bool(
            projected_prefill_radix_cache_enabled
        ),
        model_cache_identity=str(checkpoint_dir),
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
