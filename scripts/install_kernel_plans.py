#!/usr/bin/env python3
"""Re-sign measured SM120 plan mappings for this machine; never benchmarks."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from acc_infer_clear.config import load
from acc_infer_clear.kernels.plan_cache import PlanCache
from acc_infer_clear.kernels.planner import DeviceCaps

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/runtime.yaml"))
    args = parser.parse_args()
    config = load(args.config)
    caps = DeviceCaps.current()
    expected = {
        "name": "NVIDIA RTX 6000D", "sm": 120, "sms": 156,
        "shared_per_sm": 102400, "shared_per_cta": 101376,
        "registers_per_sm": 65536, "threads_per_sm": 1536,
        "max_ctas_per_sm": 32,
    }
    if asdict(caps) != expected:
        raise RuntimeError(f"Published plans are SM120/RTX 6000D-specific: {asdict(caps)}")
    engine = SimpleNamespace(config=config)
    for path in sorted((ROOT / "deployment/kernel_plan_templates").glob("*.json")):
        template = json.loads(path.read_text())
        if template["schema"] != 1 or template["source_device"] != expected:
            raise RuntimeError(f"Invalid plan template: {path}")
        if template["student_sha256"] != config["student_sha256"]:
            raise RuntimeError(f"Student mismatch: {path}")
        cache = PlanCache(engine, template["group"])
        payload = {"identity": cache.identity, "plans": template["plans"]}
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        cache.path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"group": template["group"], "plans": len(template["plans"]), "path": str(cache.path)}))


if __name__ == "__main__":
    main()

