# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MING_AR_MODEL = _REPO_ROOT / "sglang_omni/models/ming_tts/sglang_model.py"
_SGLANG_LAYERS = _REPO_ROOT / "sglang_omni/vendor/sglang/layers.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing method {class_node.name}.{name}")


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def _name_refs(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _forward_defaults(function: ast.FunctionDef) -> dict[str, str]:
    args = function.args.args
    defaults = function.args.defaults
    names = [arg.arg for arg in args[-len(defaults) :]]
    return dict(zip(names, [ast.unparse(default) for default in defaults]))


def _module_exports(module: ast.Module) -> set[str]:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            raise AssertionError("__all__ must stay as a literal list")
        return {
            item.value
            for item in node.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    raise AssertionError("missing __all__")


def test_decoder_layer_uses_sglang_layer_communicator_contract() -> None:
    module = _parse(_MING_AR_MODEL)
    decoder_layer = _class(module, "MingBailingMoeDecoderLayer")
    init = _method(decoder_layer, "__init__")
    forward = _method(decoder_layer, "forward")

    init_names = _name_refs(init)
    assert "LayerScatterModes" in init_names
    assert "LayerCommunicator" in init_names

    forward_calls = _call_names(forward)
    assert "prepare_attn_and_capture_last_layer_outputs" in forward_calls
    assert "prepare_mlp" in forward_calls
    assert "should_fuse_mlp_allreduce_with_next_layer" in forward_calls
    assert "should_use_reduce_scatter" in forward_calls
    assert "postprocess_layer" in forward_calls

    direct_layernorm_calls = {
        call.func.attr
        for call in ast.walk(forward)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"input_layernorm", "post_attention_layernorm"}
    }
    assert direct_layernorm_calls == set()
    assert "_sglang_needs_allreduce_fusion" in ast.unparse(forward)


def test_mlp_and_sparse_moe_keep_tp1_compatible_forward_contract() -> None:
    module = _parse(_MING_AR_MODEL)
    mlp_forward = _method(_class(module, "MingBailingMoeMLP"), "forward")
    moe_forward = _method(_class(module, "MingBailingMoeSparseMoeBlock"), "forward")

    for function in (mlp_forward, moe_forward):
        defaults = _forward_defaults(function)
        assert defaults["forward_batch"] == "None"
        assert defaults["should_allreduce_fusion"] == "False"
        assert defaults["use_reduce_scatter"] == "False"

    mlp_source = ast.unparse(mlp_forward)
    assert "skip_all_reduce=should_allreduce_fusion or use_reduce_scatter" in mlp_source

    moe_calls = _call_names(moe_forward)
    assert "should_skip_post_experts_all_reduce" in moe_calls
    assert "tensor_model_parallel_all_reduce" in moe_calls


def test_decoder_forwards_communicator_flags_to_mlp_with_keywords() -> None:
    module = _parse(_MING_AR_MODEL)
    forward = _method(_class(module, "MingBailingMoeDecoderLayer"), "forward")

    mlp_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mlp"
    ]
    assert len(mlp_calls) == 1
    assert {keyword.arg for keyword in mlp_calls[0].keywords} == {
        "forward_batch",
        "should_allreduce_fusion",
        "use_reduce_scatter",
    }


def test_vendor_layer_wrapper_exports_tp_communication_helpers() -> None:
    module = _parse(_SGLANG_LAYERS)

    imported_names = {
        alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "LayerCommunicator" in imported_names
    assert "LayerScatterModes" in imported_names
    assert "should_skip_post_experts_all_reduce" in imported_names

    exports = _module_exports(module)
    assert "LayerCommunicator" in exports
    assert "LayerScatterModes" in exports
    assert "should_skip_post_experts_all_reduce" in exports
