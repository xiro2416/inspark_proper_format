#!/usr/bin/env python3
"""Build the fail-closed SM89/BF16 Planner V2 candidate manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from acc_infer_clear.runtime.config import load as load_config
from acc_infer_clear.ops.planning.v2.deploy import model_hash, source_hash, toolchain
from acc_infer_clear.ops.planning.v2.inventory import canonical_inventory, keyed, tunable
from acc_infer_clear.ops.planning.v2.manifest import DeploymentManifest, RolePolicy
from acc_infer_clear.ops.planning.v2.model import HardwareProfile


ROOT = Path(__file__).resolve().parents[1]
BATCHES = tuple(range(1, 9)) + (16,)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance_summary(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    result = {}
    for batch in (1, 4, 8, 16):
        before = baseline["results"][str(batch)]["summary"]
        after = candidate["results"][str(batch)]["summary"]
        old_latency = before["all_first_chunks_ms"]["median"]
        new_latency = after["all_first_chunks_ms"]["median"]
        old_power = before["sustained_power"]
        new_power = after["sustained_power"]
        result[str(batch)] = {
            "all_first_chunks_median_ms": {"baseline": old_latency, "candidate": new_latency,
                                             "change_fraction": new_latency / old_latency - 1},
            "sustained_requests_per_s": {"baseline": old_power["first_chunk_requests_per_s"],
                                           "candidate": new_power["first_chunk_requests_per_s"],
                                           "change_fraction": new_power["first_chunk_requests_per_s"] /
                                           old_power["first_chunk_requests_per_s"] - 1},
            "sustained_joules_per_request": {"baseline": old_power["board_energy_j_per_request"],
                                               "candidate": new_power["board_energy_j_per_request"],
                                               "change_fraction": new_power["board_energy_j_per_request"] /
                                               old_power["board_energy_j_per_request"] - 1},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--formula-candidates", type=Path, required=True)
    parser.add_argument("--device-probe", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--baseline-performance", type=Path, required=True)
    parser.add_argument("--candidate-performance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    hardware = HardwareProfile.current()
    if hardware.sm != 89:
        raise RuntimeError(f"Expected SM89, got SM{hardware.sm}")
    signatures = canonical_inventory(hardware, batches=BATCHES, matrix_dtype="bf16")
    mapped = keyed(signatures)
    policies = {}
    for role in sorted({signature.role_key for signature in signatures if tunable(signature)}):
        kind = next(signature.kind for signature in signatures if signature.role_key == role)
        backend = "cublas_bf16" if kind == "gemm" else "cudnn_bf16"
        policies[role] = RolePolicy(
            backend=backend,
            global_layout="framework_managed",
            schedules={},
            legacy_exception=True,
            exception_reason=(
                "Formula candidates are analytic only: target-GPU shared-memory throughput and "
                "dependency latency counters are unavailable, and no custom backend has passed every supported shape"
            ),
            remove_when=(
                "One fixed backend completes numerical, per-shape, full-graph B1-B8/B16, power, and quality gates"
            ),
        )
    quality = json.loads(args.quality_report.read_text())
    if quality.get("cases") != 256 or not quality.get("all_exact"):
        raise RuntimeError("Alias fusion has not passed the exact 256-case PCM gate")
    probe = json.loads(args.device_probe.read_text())
    formula = json.loads(args.formula_candidates.read_text())
    if formula["hardware"]["sm"] != 89:
        raise RuntimeError("Formula artifact is not SM89")
    if formula.get("matrix_dtype") != "bf16" or tuple(formula.get("batches", ())) != BATCHES:
        raise RuntimeError("Formula artifact must be BF16 and exactly cover B1-B8/B16")
    calibration = {
        "kind": "sm89-bf16-offline-candidate",
        "supported_batches": list(BATCHES),
        "excluded_batches": {"32": "full BigVGAN head graph OOM; eager fallback is not a fair comparison"},
        "formula_candidates": {"path": str(args.formula_candidates.resolve()),
                                 "sha256": digest(args.formula_candidates),
                                 "shape_sets": len(formula["candidates"]),
                                 "analytic_only": formula["analytic_only"]},
        "device_probe": {"path": str(args.device_probe.resolve()), "sha256": digest(args.device_probe),
                         "rates_complete": probe.get("rates_complete", False)},
        "missing_hardware_terms": ["shared_memory_throughput", "global_dependency_latency"],
        "counter_blocker": "NVIDIA profiling counters require administrator permission (ERR_NVGPUCTRPERM)",
        "custom_schedule_performance_claim": False,
        "promoted_non_tile_fusion": "BigVGAN alias-free upsample+Snake",
        "quality_gate": {"path": str(args.quality_report.resolve()), "sha256": digest(args.quality_report),
                         "cases": 256, "exact_pcm16": quality["exact_pcm16"]},
        "performance_gate": performance_summary(args.baseline_performance, args.candidate_performance),
        "online_tuning": False,
    }
    config = load_config(args.config)
    manifest = DeploymentManifest(
        hardware=hardware,
        model_hash=model_hash(config),
        source_hash=source_hash(ROOT),
        toolchain=toolchain(),
        policies=policies,
        signatures=mapped,
        supported_batches=BATCHES,
        calibration=calibration,
        status="candidate",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "manifest_hash": manifest.manifest_hash,
                      "status": manifest.status, "roles": len(policies), "signatures": len(mapped),
                      "supported_batches": list(BATCHES), "custom_schedules_applied": 0}, indent=2))


if __name__ == "__main__":
    main()
