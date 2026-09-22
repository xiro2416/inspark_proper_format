#!/usr/bin/env python3
"""Full-EOS grouped generation for a declared eager/compile/TensorRT deployment.

This is corpus generation, not a scheduler throughput benchmark. Independent
arms preserve their actual codes/chunk metadata; equal seeds do not assert equal
sampling trajectories, especially for legacy shared-RNG deployment profiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from acc_infer_clear.guardrails.snapshots import file_sha256, write_json
from trt113_provenance import capture_provenance, file_record, source_identity


def generate(args):
    import numpy as np
    import soundfile as sf
    import torch
    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.engine import Engine

    corpus = json.loads(args.corpus.read_text())
    cases = corpus["cases"][:args.cases]
    if len(cases) != args.cases or len({row["id"] for row in cases}) != args.cases:
        raise ValueError("Requested corpus must have exactly the requested unique cases")
    config = load(args.config); config["max_batch"] = args.batch
    deployment = load_deployment(args.deployment)
    identity = {"corpus": file_record(args.corpus, "corpus"),
                "config": file_record(args.config, "runtime_config"),
                "deployment_file": file_record(args.deployment, "deployment"),
                "source": source_identity()}
    summary_path = args.output_dir / "generation_summary.json"
    metadata_path = args.output_dir / "generation.jsonl"
    completed = {}
    previous = None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError("Use an empty directory or explicit --resume")
        previous = json.loads(summary_path.read_text())
        for field in ("corpus", "config", "deployment_file"):
            if previous.get(field, {}).get("sha256") != identity[field]["sha256"]:
                raise ValueError(f"Resume {field} identity changed")
        if previous.get("source", {}).get("source_sha256") != identity["source"]["source_sha256"]:
            raise ValueError("Resume source files changed; use a new output directory")
        if previous.get("batch") != args.batch or previous.get("global_seed") != args.global_seed:
            raise ValueError("Resume batch/global seed changed")
        rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
        completed = {row["id"]: row for row in rows}
        if len(completed) != len(rows) or not set(completed) <= {case["id"] for case in cases}:
            raise ValueError("Resume metadata has duplicate/unexpected IDs")
        for case_id, row in completed.items():
            if not row.get("complete") or not row.get("eos") or row["sha256"] != file_sha256(args.output_dir / f"{case_id}.wav"):
                raise ValueError("Resume requires hash-verified full-EOS outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema": 1, "status": "running", **identity, "cases": len(cases),
               "scope": "full_eos_quality_corpus", "batch": args.batch, "performance_claim": False,
               "seed_semantics": "request seeds recorded; legacy shared RNG profiles need not reproduce eager codes",
               "rows_generated_this_invocation": 0, "rows_resumed": len(completed)}
    engine = None
    try:
        # Initialize global RNG deterministically too, recording it separately
        # from request RNG. This does not repair legacy cross-request sharing.
        torch.manual_seed(args.global_seed)
        summary["global_seed"] = args.global_seed
        engine = Engine(config)
        voices = {path: f"voice-{index}" for index, path in enumerate(
            dict.fromkeys(case["reference_audio"] for case in cases))}
        summary["references"] = [file_record(path, "reference_audio") for path in voices]
        for path, name in voices.items():
            engine.prepare_reference(name, path)
        summary["model_provenance"] = {component: capture_provenance(component, config, args.config,
            deployment_path=args.deployment, model=engine) for component in ("target", "draft", "cfm", "vocoder")}
        if previous is not None:
            for component, provenance in summary["model_provenance"].items():
                before = {row["role"]: row["sha256"] for row in previous["model_provenance"][component]["model_sources"]}
                now = {row["role"]: row["sha256"] for row in provenance["model_sources"]}
                if before != now:
                    raise ValueError(f"Resume actual {component} checkpoint hashes changed")
        summary["deployment"] = engine.prepare_deployment(deployment)
        summary["hardware"] = {"name": torch.cuda.get_device_name(), "sm": list(torch.cuda.get_device_capability())}
        write_json(summary_path, summary)
        with metadata_path.open("a" if args.resume else "w") as metadata:
            for group_start in range(0, len(cases), args.batch):
                original_group = cases[group_start:group_start + args.batch]
                group = [case for case in original_group if case["id"] not in completed]
                if group and len(group) != len(original_group):
                    raise ValueError("Resume cannot change a partially completed group; use a fresh directory")
                started = time.perf_counter()
                for case in group:
                    engine.create_session(case["id"], voices[case["reference_audio"]], case["seed"], case["emotion"])
                    engine.push_text(case["id"], case["text"]); engine.finish_input(case["id"])
                while engine.ready():
                    engine.run_ready()
                for offset, case in enumerate(group):
                    session = engine.sessions[case["id"]]
                    if not session["complete"] or session["error"] or not session.get("eos"):
                        raise RuntimeError(f"Incomplete/no-EOS case {case['id']}: {session['error']}")
                    pcm = np.concatenate([chunk["pcm"] for chunk in session["chunks"]])
                    if not pcm.size or pcm.dtype != np.int16:
                        raise RuntimeError("Expected nonempty PCM16 complete waveform")
                    output = args.output_dir / f"{case['id']}.wav"
                    sf.write(output, pcm, 22050, subtype="PCM_16")
                    row = {"id": case["id"], "complete": True, "eos": True, "seed": case["seed"],
                           "text": case["text"], "reference_audio": case["reference_audio"],
                           "sha256": file_sha256(output), "samples": int(pcm.size), "sample_rate": 22050,
                           "codes": list(session["codes"]), "accepted": list(session["accepted"]),
                           "rounds": session["rounds"], "group_elapsed_s": time.perf_counter() - started,
                           "requested_batch": args.batch, "group_ids": [entry["id"] for entry in group],
                           "chunks": [{key: value for key, value in chunk.items() if key != "pcm"}
                                      for chunk in session["chunks"]]}
                    metadata.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); metadata.flush()
                    engine.release(case["id"])
                    summary["rows_generated_this_invocation"] += 1
                    print(json.dumps({"completed": group_start + offset + 1, "total": len(cases), "id": case["id"],
                                      "samples": row["samples"], "group_elapsed_s": row["group_elapsed_s"]}), flush=True)
        summary["head_routes"] = engine.head_graphs.stats() if engine.head_graphs else None
        summary["runtime_counters"] = {
            "target": engine.rt.target.stats(), "draft": engine.rt.backbone.stats(),
            "device_round_attempts": engine.device_round_attempts,
            "device_round_successes": engine.device_round_successes,
            "device_round_fallbacks": engine.device_round_fallbacks,
            "native_target_steps": getattr(engine.rt, "native_target_steps", 0),
            "device_target_steps": getattr(engine.rt, "device_target_steps", 0),
            "native_draft_steps": getattr(engine.rt.backbone, "native_full_steps", 0),
        }
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if engine is not None:
            engine.close()
        write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--config", default="configs/runtime_reference.yaml")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("configs/sm89_quality_256.json"))
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-seed", type=int, default=113)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cases <= 256:
        parser.error("--cases must be in [1,256]")
    if args.resume:
        from acc_infer_clear.runtime.deployment import load
        plan = load(args.deployment)
        if plan.get("batched_proposal_rng") or plan.get("device_round_b8"):
            parser.error("Legacy shared-RNG profiles cannot resume by skipping completed requests")
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        generate(args)


if __name__ == "__main__":
    main()
