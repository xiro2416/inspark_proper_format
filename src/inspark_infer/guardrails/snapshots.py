"""Immutable, hash-bound tensor evidence for offline same-input replay.

Snapshots deliberately synchronize/transfer to CPU. They are audit artifacts,
not a benchmark facility, and must never be installed during graph capture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cpu_copy(value):
    """Clone before a producer can reuse static CUDA graph output storage."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to("cpu").contiguous()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    if isinstance(value, dict):
        return {name: cpu_copy(item) for name, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def tensor_inventory(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Inventory requires a CPU snapshot")
        data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return [{"name": prefix, "shape": list(value.shape), "dtype": str(value.dtype),
                 "sha256": hashlib.sha256(data).hexdigest(), "elements": value.numel()}]
    if isinstance(value, (tuple, list)):
        return [row for index, item in enumerate(value)
                for row in tensor_inventory(item, f"{prefix}/{index}")]
    if isinstance(value, dict):
        return [row for key, item in sorted(value.items())
                for row in tensor_inventory(item, f"{prefix}/{key}")]
    return []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_bundle(directory, index, value):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{index:06d}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable evidence: {path}")
    frozen = cpu_copy(value)
    torch.save(frozen, path)
    return {"file": path.name, "sha256": file_sha256(path), "tensors": tensor_inventory(frozen)}


def load_bundle(directory, record):
    import torch
    root = Path(directory).resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != record["sha256"]:
        raise ValueError("Snapshot path/hash does not match its manifest")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_inventory(value) != record["tensors"]:
        raise ValueError("Snapshot tensor inventory does not match its manifest")
    return value


def compare_regions(expected, actual, precision, mask=None):
    """Report CFM masked and generated elements separately; neither dilutes a gate."""
    import torch
    from inspark_infer.guardrails.numerics import compare
    results = {"all": compare(expected, actual, precision)}
    if mask is not None and expected.shape == actual.shape:
        visible = mask.bool().expand_as(expected)
        for name, region in (("masked_prompt", visible), ("generated", ~visible)):
            if bool(region.any()):
                results[name] = compare(expected[region], actual[region], precision)
            else:
                results[name] = {"pass_gate": False, "reason": "empty required CFM region"}
        results["masked_exact_zero"] = {
            "pass_gate": bool(visible.any()) and bool(torch.all(actual[visible] == 0)),
            "elements": int(visible.sum()),
        }
    return {"comparisons": results, "pass_gate": all(row["pass_gate"] for row in results.values())}
