#!/usr/bin/env python3
"""Fail closed before model loading or CUDA Graph capture."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from inspark_infer.runtime.config import load
from inspark_infer.ops.planning.plan_cache import PlanCache
from inspark_infer.ops.planning.planner import DeviceCaps

ROOT = Path(__file__).resolve().parents[1]


def digest(plans: dict) -> str:
    return hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()


def main() -> None:
    if torch.__version__ != "2.8.0+cu128" or triton.__version__ != "3.5.0" or torch.version.cuda != "12.8":
        raise RuntimeError((torch.__version__, triton.__version__, torch.version.cuda))
    caps = DeviceCaps.current()
    if caps.sm != 120 or caps.name != "NVIDIA RTX 6000D":
        raise RuntimeError(f"This release contains only RTX 6000D/SM120 plans: {caps}")
    config = load(ROOT / "configs/common/runtime.yaml")
    engine = SimpleNamespace(config=config)
    for path in sorted((ROOT / 'configs/common/kernel_plan_templates').glob("*.json")):
        template = json.loads(path.read_text())
        cache = PlanCache(engine, template["group"])
        actual = json.loads(cache.path.read_text())
        if digest(actual["plans"]) != digest(template["plans"]):
            raise RuntimeError(f"Plan mapping changed: {path.name}")
    print(json.dumps({"status": "ready", "device": caps.name, "sm": caps.sm, "online_tuning": False}))


if __name__ == "__main__":
    main()

