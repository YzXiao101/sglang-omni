from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import PortArgs, ServerArgs

from sglang_omni.utils.gpu_memory import (
    calculate_stage_budget_available_bytes,
    calculate_stage_load_delta_bytes,
    format_bytes_gib,
    get_gpu_device_info,
    get_process_gpu_memory_bytes,
)

logger = logging.getLogger(__name__)


_SGLANG_LOADER_POSTPROCESS_PROFILE_PATCHED = False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _param_nbytes(param: Any) -> int:
    try:
        return int(param.numel()) * int(param.element_size())
    except Exception:
        return 0


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1 << 30):.2f}GiB"


def _module_param_bytes_by_device(module: Any) -> dict[str, int]:
    stats: dict[str, int] = {}
    for param in module.parameters():
        device = str(getattr(param, "device", "unknown"))
        stats[device] = stats.get(device, 0) + _param_nbytes(param)
    return stats


def _format_postprocess_module_key(
    name: str,
    module: Any,
    quant_method: Any,
) -> str:
    module_name = name or "<root>"
    cls_name = type(module).__name__
    method_name = type(quant_method).__name__
    layer_id = getattr(module, "layer_id", None)
    if layer_id is None:
        return f"{module_name}|{cls_name}|{method_name}"
    return f"{module_name}|{cls_name}|{method_name}|layer={layer_id}"


def _format_postprocess_top_modules(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> str:
    if not records:
        return "[]"
    top_records = sorted(records, key=lambda item: item["context_s"], reverse=True)[
        :limit
    ]
    parts = []
    for item in top_records:
        overhead_s = max(0.0, item["context_s"] - item["process_s"])
        parts.append(
            f"{item['key']}:context_s={item['context_s']:.2f},"
            f"process_s={item['process_s']:.2f},overhead_s={overhead_s:.2f},"
            f"cpu={_format_gib(item['cpu_bytes'])},"
            f"cuda={_format_gib(item['cuda_bytes'])},"
            f"other={_format_gib(item['other_bytes'])}"
        )
    return "[" + ", ".join(parts) + "]"


def _install_sglang_loader_postprocess_profiler() -> None:
    """Install an opt-in profiler around SGLang's post-load quant loop."""
    global _SGLANG_LOADER_POSTPROCESS_PROFILE_PATCHED
    if not _env_flag("SGLANG_OMNI_SGLANG_POSTPROCESS_PROFILE"):
        return
    if _SGLANG_LOADER_POSTPROCESS_PROFILE_PATCHED:
        return

    from sglang.srt.model_loader import loader as loader_module

    current = loader_module.DefaultModelLoader.load_weights_and_postprocess
    if getattr(current, "_sglang_omni_profile_wrapped", False):
        _SGLANG_LOADER_POSTPROCESS_PROFILE_PATCHED = True
        return

    def profiled_load_weights_and_postprocess(model, weights, target_device):
        load_start_s = time.perf_counter()
        model.load_weights(weights)
        load_weights_s = time.perf_counter() - load_start_s

        postprocess_start_s = time.perf_counter()
        records: list[dict[str, Any]] = []
        total_cpu_bytes = 0
        total_cuda_bytes = 0
        total_other_bytes = 0
        total_context_s = 0.0
        total_process_s = 0.0
        quant_module_count = 0

        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue

            quant_module_count += 1
            device_bytes = _module_param_bytes_by_device(module)
            cpu_bytes = sum(
                num_bytes
                for device, num_bytes in device_bytes.items()
                if device.startswith("cpu")
            )
            cuda_bytes = sum(
                num_bytes
                for device, num_bytes in device_bytes.items()
                if device.startswith("cuda")
            )
            other_bytes = sum(device_bytes.values()) - cpu_bytes - cuda_bytes

            context_start_s = time.perf_counter()
            with loader_module.device_loading_context(module, target_device):
                process_start_s = time.perf_counter()
                quant_method.process_weights_after_loading(module)
                process_s = time.perf_counter() - process_start_s
            context_s = time.perf_counter() - context_start_s

            total_cpu_bytes += cpu_bytes
            total_cuda_bytes += cuda_bytes
            total_other_bytes += other_bytes
            total_context_s += context_s
            total_process_s += process_s
            records.append(
                {
                    "key": _format_postprocess_module_key(
                        name,
                        module,
                        quant_method,
                    ),
                    "context_s": context_s,
                    "process_s": process_s,
                    "cpu_bytes": cpu_bytes,
                    "cuda_bytes": cuda_bytes,
                    "other_bytes": other_bytes,
                }
            )

        if getattr(loader_module, "_is_npu", False):
            import torch

            torch.npu.empty_cache()

        postprocess_s = time.perf_counter() - postprocess_start_s
        logger.info(
            "SGLang loader postprocess profile: load_weights_s=%.2f "
            "postprocess_s=%.2f total_s=%.2f quant_modules=%d "
            "context_s=%.2f process_s=%.2f context_overhead_s=%.2f "
            "cpu_param_bytes=%s cuda_param_bytes=%s other_param_bytes=%s "
            "top_context_modules=%s",
            load_weights_s,
            postprocess_s,
            load_weights_s + postprocess_s,
            quant_module_count,
            total_context_s,
            total_process_s,
            max(0.0, total_context_s - total_process_s),
            _format_gib(total_cpu_bytes),
            _format_gib(total_cuda_bytes),
            _format_gib(total_other_bytes),
            _format_postprocess_top_modules(records, limit=10),
        )

    profiled_load_weights_and_postprocess._sglang_omni_profile_wrapped = True
    loader_module.DefaultModelLoader.load_weights_and_postprocess = staticmethod(
        profiled_load_weights_and_postprocess
    )
    _SGLANG_LOADER_POSTPROCESS_PROFILE_PATCHED = True


def filter_weights_by_prefix(
    weights: Iterator[tuple[str, Any]],
    prefix: str | None,
) -> Iterator[tuple[str, Any]]:
    """Filter weight iterator by prefix, stripping matched prefix from names."""
    if not prefix:
        yield from weights
        return
    for name, tensor in weights:
        if name.startswith(prefix):
            yield name[len(prefix) :], tensor


class SGLModelRunner(ModelRunner):
    """Thin wrapper to bootstrap SGLang ModelRunner from backend args."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        model_arch_override: str | None = None,
        weight_prefix: str | None = None,
        total_gpu_memory_fraction: float | None = None,
    ) -> None:
        self._weight_prefix = weight_prefix
        self._total_gpu_memory_fraction = total_gpu_memory_fraction
        self._register_omni_model()
        _install_sglang_loader_postprocess_profiler()

        port_args = PortArgs.init_new(server_args)
        tp_size = server_args.tp_size
        self.nccl_port = port_args.nccl_port

        # model_config is already fully configured by ModelWorker._init_model_config()
        # (architecture override, text_config swap, etc. are all done there)

        super().__init__(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def _register_omni_model(self):
        # Register sglang_omni model classes directly in SGLang's model registry.
        from sglang.srt.models.registry import ModelRegistry

        from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
            S2ProSGLangTextModel,
        )
        from sglang_omni.models.higgs_tts.model import HiggsTTSModel
        from sglang_omni.models.llada2_uni.components.thinker import LLaDA2MoeModelLM
        from sglang_omni.models.ming_omni.registration import (
            register_ming_hf_config,
            register_ming_model_registry,
        )
        from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
        from sglang_omni.models.qwen3_asr.sglang_model import (
            Qwen3ASRForConditionalGeneration,
        )
        from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
            Qwen3OmniThinkerForCausalLM,
        )
        from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
        from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
        from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel
        from sglang_omni.models.whisper_asr.sglang_model import (
            WhisperForConditionalGeneration,
        )

        register_ming_hf_config()
        register_ming_model_registry()

        ModelRegistry.models["S2ProSGLangTextModel"] = S2ProSGLangTextModel
        ModelRegistry.models["Qwen3OmniTalker"] = Qwen3OmniTalker
        ModelRegistry.models["Qwen3OmniThinkerForCausalLM"] = (
            Qwen3OmniThinkerForCausalLM
        )
        ModelRegistry.models["HiggsMultimodalQwen3ForConditionalGeneration"] = (
            HiggsTTSModel
        )
        ModelRegistry.models["Qwen3TTSTalker"] = Qwen3TTSTalker
        ModelRegistry.models["MossTTSDelaySGLangModel"] = MossTTSDelaySGLangModel
        ModelRegistry.models["VoxtralSGLangTTSModel"] = VoxtralSGLangTTSModel
        ModelRegistry.models["LLaDA2MoeModelLM"] = LLaDA2MoeModelLM
        ModelRegistry.models["WhisperForConditionalGeneration"] = (
            WhisperForConditionalGeneration
        )
        ModelRegistry.models["Qwen3ASRForConditionalGeneration"] = (
            Qwen3ASRForConditionalGeneration
        )

    def _profile_available_bytes(self, pre_model_load_memory: float) -> int:
        """Profile KV-cache headroom for colocated SGLang AR stages.

        Upstream SGLang profiles from global free-memory deltas. That is valid
        for a single AR engine, but colocated Omni stages can load multiple
        SGLang engines in separate processes on the same GPU. In that case
        another process can change global free memory while this process is
        loading weights, making the global delta too small or negative.

        When a stage total-memory budget is provided, compute cache headroom as
        total GPU memory times that budget minus this stage's measured memory.
        NVML process accounting is preferred. If NVML cannot identify the
        current process, use the stage-local load delta measured inside
        SGLang's serialized initialization window. Without a stage budget, keep
        upstream SGLang profiling semantics for ordinary non-colocated AR
        serving.
        """
        if self._total_gpu_memory_fraction is None:
            return self._profile_available_bytes_from_free_memory_delta(
                pre_model_load_memory
            )

        process_memory = get_process_gpu_memory_bytes(self.gpu_id)
        device_info = get_gpu_device_info(self.gpu_id)
        total_memory = device_info.total_memory_bytes

        if total_memory is None:
            raise RuntimeError(
                "Colocated SGLang AR stage requires total GPU memory for "
                f"gpu_id={self.gpu_id}. Check CUDA_VISIBLE_DEVICES and CUDA "
                "device visibility."
            )

        if process_memory is None or process_memory <= 0:
            return self._profile_available_bytes_from_stage_load_delta(
                pre_model_load_memory,
                total_memory,
            )

        return self._profile_available_bytes_from_process_memory(
            total_memory,
            process_memory,
        )

    def _profile_available_bytes_from_stage_load_delta(
        self,
        pre_model_load_memory: float,
        total_memory: int,
    ) -> int:
        """Profile colocated KV headroom from this stage's load-time delta."""
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils.common import get_available_gpu_memory

        world_group = get_world_group()
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        stage_load_bytes = calculate_stage_load_delta_bytes(
            pre_model_load_memory_gib=pre_model_load_memory,
            post_model_load_memory_gib=post_model_load_memory,
        )
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=stage_load_bytes,
            memory_fraction=self._total_gpu_memory_fraction,
            accounted_memory_label="stage_load_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=stage_load_fallback "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self._total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"stage_load_used={format_bytes_gib(stage_load_bytes)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes

    def _profile_available_bytes_from_free_memory_delta(
        self, pre_model_load_memory: float
    ) -> int:
        """Match SGLang free-memory-delta accounting for non-colocated AR stages."""
        from sglang.srt.distributed.parallel_state import get_world_group
        from sglang.srt.utils.common import get_available_gpu_memory

        world_group = get_world_group()
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=world_group.world_size > 1,
            cpu_group=world_group.cpu_group,
        )
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        return int(rest_memory * (1 << 30))

    def profile_max_num_token(self, pre_model_load_memory: float) -> int:
        """Profile token capacity for stage-budgeted colocated AR stages."""
        if self._total_gpu_memory_fraction is None:
            return ModelRunnerKVCacheMixin.profile_max_num_token(
                self,
                pre_model_load_memory,
            )

        num_layers = self._num_kv_cache_layers()
        cell_size = self.get_cell_size_per_token(num_layers)
        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        if self.mambaish_config is not None:
            available_gib = available_bytes / (1 << 30)
            available_bytes = int(
                self.handle_max_mamba_cache(available_gib) * (1 << 30)
            )
        return available_bytes // cell_size

    def _profile_available_bytes_from_process_memory(
        self,
        total_memory: int,
        process_memory: int,
    ) -> int:
        available_bytes = calculate_stage_budget_available_bytes(
            total_memory_bytes=total_memory,
            accounted_memory_bytes=process_memory,
            memory_fraction=self._total_gpu_memory_fraction,
            accounted_memory_label="process_used",
        )
        logger.info(
            f"SGLang AR memory profile: gpu_mem_accounting=nvml_process "
            f"gpu_id={self.gpu_id} "
            f"total_gpu_memory_fraction={self._total_gpu_memory_fraction:.3f} "
            f"mem_fraction_static={self.mem_fraction_static:.3f} "
            f"total={format_bytes_gib(total_memory)} "
            f"process_used={format_bytes_gib(process_memory)} "
            f"available_for_kv={format_bytes_gib(available_bytes)}"
        )
        return available_bytes

    def _num_kv_cache_layers(self) -> int:
        """Return the number of layers used by SGLang KV-cache sizing."""
        if self.is_draft_worker:
            return getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        if mambaish := self.mambaish_config:
            return len(
                [
                    layer_id
                    for layer_id in mambaish.full_attention_layer_ids
                    if self.start_layer <= layer_id < self.end_layer
                ]
            )
        return self.num_effective_layers
