"""Build provenance from actual local files; no model download or historical-hash substitution."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys


def file_record(path, role):
    path = Path(path).resolve()
    if not path.is_relative_to(Path("/workspace")):
        raise ValueError(f"Provenance file must stay within /workspace: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"role": role, "path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def source_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    if not root.is_relative_to(Path("/workspace")):
        raise ValueError("Source provenance root must stay within /workspace")
    scope = ["src", "scripts", "benchmarks", "configs", "pyproject.toml"]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)

    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *scope)
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *scope)
    records = []
    for name in sorted(set(names.decode().split("\0")) - {""}):
        path = root / name
        records.append((name, file_record(path, "source")["sha256"] if path.exists() else None))
    if not records:
        raise ValueError("Source fingerprint cannot be built from an empty source inventory")
    source_hash = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    return {
        "git_sha": head, "dirty": bool(status), "scope": scope, "files": len(records),
        "source_sha256": source_hash,
        "dirty_fingerprint": hashlib.sha256(status + source_hash.encode()).hexdigest() if status else None,
        "snapshot": "local source files at provenance collection time; caller defines build/capture/replay phase",
    }


def capture_onnx_artifact(path):
    """Bind an exported graph and all external tensor payloads by actual hash."""
    import onnx

    path = Path(path).resolve()
    model = onnx.load(str(path), load_external_data=False)

    def tensors(graph):
        yield from graph.initializer
        for sparse in graph.sparse_initializer:
            yield sparse.values
            yield sparse.indices
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    yield from tensors(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        yield from tensors(nested)

    locations = set()
    for tensor in tensors(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            location = dict((entry.key, entry.value) for entry in tensor.external_data).get("location")
            if not location:
                raise ValueError("External ONNX tensor has no location")
            locations.add(location)
    external = []
    for location in sorted(locations):
        payload = (path.parent / location).resolve()
        if not payload.is_relative_to(path.parent):
            raise ValueError("ONNX external tensor escapes the export directory")
        record = file_record(payload, "onnx_external_data")
        record["path"] = str(payload.relative_to(path.parent))
        external.append(record)
    return dict(file_record(path, "onnx"), external_data=external)


def load_onnx_export(path):
    """Only inherit weight/source identity bound to the exact export being built.

    Legacy exports remain buildable, but are explicitly unverified. This function
    never hashes today's checkpoints to manufacture provenance for an older ONNX.
    """
    path = Path(path).resolve()
    actual = file_record(path, "onnx")
    metadata_path = path.with_suffix(".export.json")
    export = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    declared = export.get("onnx_sha256")
    if declared is not None and declared != actual["sha256"]:
        raise ValueError("ONNX hash disagrees with its export metadata")
    original = export.get("provenance")
    binding = {"onnx": actual, "external_data": [], "export_metadata": None}
    if metadata_path.exists():
        binding["export_metadata"] = file_record(metadata_path, "onnx_export_metadata")
    if not isinstance(original, dict) or original.get("status") != "recorded_not_audited":
        return export, {
            "schema": 1, "status": "legacy_unverified",
            "reason": "ONNX export has no recorded, hash-bound checkpoint/source provenance",
            "onnx_binding": binding,
        }
    artifact = export.get("onnx_artifact")
    if (declared is None or not isinstance(artifact, dict)
            or artifact.get("sha256") != actual["sha256"]
            or not isinstance(artifact.get("external_data"), list)):
        raise ValueError("Recorded ONNX provenance requires graph and external-data hash metadata")
    if (original.get("schema") != 1 or not original.get("model_sources")
            or not original.get("source", {}).get("source_sha256")):
        raise ValueError("Recorded ONNX provenance lacks checkpoint or source identity")
    for expected in artifact["external_data"]:
        payload = (path.parent / expected["path"]).resolve()
        if not payload.is_relative_to(path.parent):
            raise ValueError("ONNX external tensor metadata escapes the export directory")
        record = file_record(payload, "onnx_external_data")
        if record["sha256"] != expected.get("sha256"):
            raise ValueError(f"ONNX external tensor hash mismatch: {expected['path']}")
        record["path"] = str(payload.relative_to(path.parent))
        binding["external_data"].append(record)
    inherited = deepcopy(original)
    inherited["onnx_binding"] = binding
    return export, inherited


def capture_provenance(component, config, config_path, deployment_path=None, model=None):
    """Capture one component's real checkpoint files plus build environment.

    ``model`` accepts the current Engine, so Target can use its actual resolved
    ``tts.gpt_path`` rather than assuming the official checkpoint filename.
    Hardware is queried only when this function runs inside an authorized build.
    """
    import numpy
    import torch
    import triton

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Build provenance requires exactly one visible GPU")
    # A PyTorch ONNX export does not need TensorRT installed/imported. Builders
    # have already imported the selected TRT version before capturing provenance.
    trt = sys.modules.get("tensorrt")
    weights = Path(config["weights"]).resolve()
    base = weights / "index_tts2"
    if component == "target":
        if model is None:
            raise ValueError("Target provenance requires the loaded Engine and its actual gpt_path")
        paths = [(model.tts.gpt_path, "target_checkpoint"), (base / "config.yaml", "target_config")]
    elif component == "draft":
        paths = [(weights / "draft_onpolicy100/model.safetensors", "draft_checkpoint"),
                 (weights / "draft_onpolicy100/config.json", "draft_config")]
    elif component == "cfm":
        if model is None:
            raise ValueError("CFM provenance requires the loaded Engine and its actual s2mel config")
        paths = [(Path(model.tts.model_dir) / str(model.tts.cfg.s2mel_checkpoint), "s2mel_checkpoint"),
                 (Path(config["student"]), "student_checkpoint"),
                 (base / "config.yaml", "s2mel_config")]
    elif component == "vocoder":
        directory = base / "hf_cache/bigvgan"
        paths = [(directory / "bigvgan_generator.pt", "vocoder_checkpoint"),
                 (directory / "config.json", "vocoder_config")]
    else:
        raise ValueError(f"Unknown component: {component}")
    configuration = [file_record(config_path, "runtime_config")]
    if deployment_path is not None:
        configuration.append(file_record(deployment_path, "deployment_config"))
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "schema": 1, "status": "recorded_not_audited", "component": component,
        "model_sources": [file_record(path, role) for path, role in paths],
        "configuration": configuration, "source": source_identity(),
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "triton": triton.__version__, "numpy": numpy.__version__,
                     "tensorrt": getattr(trt, "__version__", None)},
        "hardware": {"gpu_name": torch.cuda.get_device_name(0), "sm": major * 10 + minor},
    }
