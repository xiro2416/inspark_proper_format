"""Explicit native imports and a pinned source-only semantic probe.

Source extraction executes unmodified NVIDIA class/function definitions to
check checkpoint mapping without pretending the CUDA executor is installed.
Downloaded source is ignored build material, accompanied by hashes and URL.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import urlopen

import torch
from torch import nn
from torch.nn import functional as F

AUDITED_COMMIT = "0c58480ca680f2946bb94315af3517c9c7000bca"
SOURCE_PATHS = (
    "tensorrt_llm/_torch/models/modeling_speculative.py",
    "tensorrt_llm/_torch/models/modeling_dspark.py",
    "tensorrt_llm/_torch/speculative/dspark.py",
    "tensorrt_llm/_torch/speculative/dflash_attention.py",
    "tensorrt_llm/_torch/modules/linear.py",
    "tensorrt_llm/_torch/modules/embedding.py",
)


def fetch_audited_sources(directory):
    directory = Path(directory)
    manifest = {"commit": AUDITED_COMMIT, "scope": "source_only_not_native_runtime", "files": {}}
    for relative in SOURCE_PATHS:
        url = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/{AUDITED_COMMIT}/{relative}"
        content = urlopen(url, timeout=60).read()
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["files"][relative] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    (directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verified_source(directory, relative):
    directory = Path(directory)
    manifest = json.loads((directory / "source_manifest.json").read_text())
    if manifest["commit"] != AUDITED_COMMIT:
        raise ValueError("Probe source commit differs from audited commit")
    content = (directory / relative).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["files"][relative]["sha256"]:
        raise ValueError(f"Source hash changed: {relative}")
    return content.decode()


def source_rnn_factory(directory):
    """Return actual NVIDIA RNN classes without importing the CUDA package."""
    relative = SOURCE_PATHS[0]
    tree = ast.parse(verified_source(directory, relative), filename=relative)
    wanted = {"markov_prev_embeddings", "VanillaMarkov", "RNNHead"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("Audited NVIDIA RNN source definitions changed")
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Tuple=Tuple,
                     List=List, Dict=Dict, Callable=Callable, Any=Any)
    embedding_tree = ast.parse(verified_source(directory, SOURCE_PATHS[5]))
    embedding = next(node for node in embedding_tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "get_masked_input_and_mask")
    nodes.insert(0, embedding)
    # Only the extracted classes' embedding/step methods are called; their
    # original block sampler is intentionally not used for the project's PCG.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, "exec"), namespace)
    return namespace["RNNHead"]


def native_rnn_factory():
    return importlib.import_module("tensorrt_llm._torch.models.modeling_speculative").RNNHead


def reproduce_standalone_rnn_guard(directory):
    """Run the actual upstream guard body with a minimal config resolver."""
    from types import SimpleNamespace
    source = verified_source(directory, SOURCE_PATHS[1])
    tree = ast.parse(source)
    mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DSparkHeadMixin")
    method = next(n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "_init_dspark_heads")
    # Config lookup is the only external dependency used before the guard.
    # The real guard body, not an inferred duplicate, executes below.
    namespace = {"resolve_dspark_head_config": lambda cfg, key: getattr(cfg, "dspark_" + key, None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), SOURCE_PATHS[1], "exec"), namespace)
    config = SimpleNamespace(dspark_markov_rank=256, dspark_markov_head_type="rnn")
    try:
        namespace["_init_dspark_heads"](SimpleNamespace(), config)
    except ValueError as error:
        return {"status": "blocked", "reason": str(error), "actual_upstream_guard_executed": True}
    return {"status": "unexpectedly_allowed", "actual_upstream_guard_executed": True}


def probe_installed_interfaces():
    """Inspect real installed classes, without constructing a CUDA executor."""
    import inspect
    from types import SimpleNamespace
    model = importlib.import_module("tensorrt_llm._torch.models.modeling_dspark")
    workers = importlib.import_module("tensorrt_llm._torch.speculative.dspark")
    dflash = importlib.import_module("tensorrt_llm._torch.models.modeling_dflash")
    result = {"native_package_imported": True, "executor_constructed": False,
              "draft_model": inspect.getfile(model.GQADSparkForCausalLM),
              "worker": inspect.getfile(workers.DSparkWorker), "interfaces": {}}
    for name in ("precompute_context_kv", "dflash_forward", "project_target_hidden"):
        result["interfaces"][name] = str(inspect.signature(getattr(dflash.DFlashForCausalLM, name)))
    config = SimpleNamespace(dspark_markov_rank=256, dspark_markov_head_type="rnn")
    mixin = getattr(model, "_DSparkHeadMixin", None)
    if mixin is not None:
        try:
            mixin._init_dspark_heads(SimpleNamespace(), config)
        except Exception as error:
            result["standalone_rnn_constructor"] = {"type": type(error).__name__, "reason": str(error), "executed": True}
    else:
        # rc28 keeps the guard inside the full model constructor (after CUDA
        # model allocation). Inspect it on a CPU-only probe rather than
        # monkeypatching its parent constructor and claiming a native run.
        constructor = inspect.getsource(model.GQADSparkForCausalLM.__init__)
        result["standalone_rnn_constructor"] = {
            "executed": False, "location": "GQADSparkForCausalLM.__init__",
            "rnn_rejection_guard_present": "only 'vanilla' is " in constructor,
        }
    llm_args = importlib.import_module("tensorrt_llm.llmapi.llm_args")
    mapping = importlib.import_module("tensorrt_llm.mapping")
    native_config = llm_args.DSparkDecodingConfig(
        max_draft_len=7, block_size=7, markov_rank=256, markov_head_type="rnn",
        mask_token_id=8194, target_layer_ids=[1, 6, 11, 16, 21], attention_backend="VANILLA")
    worker = workers.DSparkWorker(native_config, mapping.Mapping(world_size=1, rank=0, tp_size=1))
    result["worker_constructed"] = type(worker).__name__
    result["worker_target_positions"] = worker._draft_tokens_per_req
    result["worker_backbone_contract_probe"] = {
        "draft_slots": worker._draft_block_width(SimpleNamespace(_dspark_shift_label=True)),
        "scope": "worker_construction_only_no_model_forward",
    }
    return result


def register_native_model_factories(target_factory, draft_factory):
    """Use NVIDIA's registration APIs; do not silently select another model.

    Factories must implement the installed runtime's model/KV interfaces.
    This registers an external architecture and the DSPARK builder. The
    worker's PCG/RNN policy hook still needs integration; registration alone
    is deliberately not reported as successful executor deployment.
    """
    utils = importlib.import_module("tensorrt_llm._torch.models.modeling_utils")
    interface = importlib.import_module("tensorrt_llm._torch.speculative.interface")
    utils.register_auto_model("IndexTTS2ForCausalLM")(target_factory)
    utils.register_draft_model(interface.SpeculativeDecodingMode.DSPARK)(draft_factory)
    return {"target_architecture": "IndexTTS2ForCausalLM", "mode": "DSPARK",
            "registration": True, "executor_validated": False}
