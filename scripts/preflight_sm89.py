#!/usr/bin/env python3
"""Fail-closed preflight for portable SM89 baseline deployments."""
import argparse
import json
from pathlib import Path

import torch
import triton

from acc_infer_clear.kernels.planner import DeviceCaps
from acc_infer_clear.runtime.deployment import load

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()

    versions = (torch.__version__, triton.__version__, torch.version.cuda)
    if versions != ("2.8.0+cu128", "3.5.0", "12.8"):
        raise RuntimeError(f"Unsupported software identity: {versions}")

    caps = DeviceCaps.current()
    if caps.sm != 89 or caps.name != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"SM89 baseline requires RTX 4090: {caps}")

    plan = load(args.deployment.resolve())
    allowed_status = {
        "inspark_marlin_sm89_fp32_baseline": ("fp32", "fp32"),
        "inspark_marlin_sm89_bf16_baseline": ("bf16", "bf16"),
        "inspark_marlin_sm89_bf16_triton_fusions": ("bf16", "bf16"),
        "inspark_marlin_sm89_bf16_triton_device_control_experimental": ("bf16", "bf16"),
        "inspark_marlin_sm89_bf16_triton_device_control_alias_candidate": ("bf16", "bf16"),
    }
    requested = (plan["precision"], plan["rnn_precision"])
    if plan["status"] not in allowed_status or requested != allowed_status[plan["status"]]:
        raise RuntimeError(f"Not an approved SM89 baseline: {plan['status']!r}, {requested!r}")
    if plan["schema"] != 1:
        raise RuntimeError("SM89 baseline must not reference device-specific kernel plans")
    experimental = "_device_control_" in plan["status"]
    device_chain = ("context_graphs", "context_scatter", "device_accept_plan",
                    "device_residual", "device_round_b8")
    if experimental:
        if not plan["fused_acceptance"] or not all(plan.get(key) for key in device_chain):
            raise RuntimeError("Experimental device-control profile requires the complete device chain")
        if plan.get("batched_proposal_rng"):
            raise RuntimeError("Keep batched Proposal RNG out of the isolated device-control experiment")
    elif plan["fused_acceptance"] or any(plan.get(key) for key in device_chain):
        raise RuntimeError("Device-control features must remain disabled in SM89 baselines")
    if plan["tail_graphs"] or plan["overlap_acoustics"]:
        raise RuntimeError("Unrelated experimental features must remain disabled")
    alias_candidate = plan["status"].endswith("_alias_candidate")
    if alias_candidate != (plan.get("acoustic_kernels") == "alias" and plan.get("acoustic_plan") is None):
        raise RuntimeError("Alias candidate status and acoustic kernel selection must match")

    print(json.dumps({
        "status": "ready",
        "profile": plan["status"],
        "device": caps.name,
        "sm": caps.sm,
        "sms": caps.sms,
        "precision": plan["precision"],
        "triton_fusions": plan.get("cfm_triton_fusions", []),
        "custom_triton_fusions": bool(plan.get("cfm_triton_fusions")),
        "custom_kernel_plan": False,
        "alias_free_fusion": alias_candidate,
        "device_control_experimental": experimental,
        "online_tuning": False,
    }))


if __name__ == "__main__":
    main()
