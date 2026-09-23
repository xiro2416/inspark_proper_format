"""TensorRT quantization intent, deliberately separate from builder tactics."""
from __future__ import annotations

import json
from pathlib import Path


def validate_policy(policy: dict) -> dict:
    if set(policy) != {"schema", "scheme", "calibration", "scale_format", "qdq_graph_sha256"} or policy["schema"] != 1:
        raise ValueError("Unsupported TensorRT quantization policy schema")
    if policy["scheme"] == "none":
        if any(policy[key] is not None for key in ("calibration", "scale_format", "qdq_graph_sha256")):
            raise ValueError("Unquantized policy must not claim calibration or Q/DQ artifacts")
        return policy
    if policy["scheme"] not in ("fp8", "nvfp4"):
        raise ValueError("Unsupported TensorRT quantization scheme")
    raise NotImplementedError(
        f"TensorRT {policy['scheme']} is not implemented: calibrate/convert weights, "
        "export an explicit Q/DQ graph with validated scales, then audit quality "
        "and builder tactics on the target GPU; no BF16 substitution is allowed"
    )


def load_policy(path: Path) -> dict:
    return validate_policy(json.loads(Path(path).read_text()))
