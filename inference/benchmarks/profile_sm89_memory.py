#!/usr/bin/env python3
"""VRAM peak and CUDA Graph pool cost for one batch tier (whiteboard section 15).

Separates the three memory questions that matter for admission headroom:
  1. weights + reference bank after model load,
  2. the cost of capturing the configured graph inventory (the part that made B32
     with head graphs fail on a 48 GiB device),
  3. the transient peak while serving a first chunk.

  bash scripts/run.sh scripts/profile_sm89_memory.py --gpu 6 --batch 8 \
      --deployment configs/sm89_bf16_triton.json --ref-audio outputs/profile_sm89/reference.wav
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_triton.json")
    parser.add_argument("--ref-audio", default="outputs/profile_sm89/reference.wav")
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from acc_infer_clear.runtime.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.engine import Engine

    config = load_config(args.config)
    config["max_batch"] = args.batch
    deployment = load_deployment(args.deployment)

    def snapshot(label, torch_module):
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        return {
            "label": label,
            "allocated_mib": torch_module.cuda.memory_allocated() >> 20,
            "reserved_mib": torch_module.cuda.memory_reserved() >> 20,
            "max_allocated_mib": torch_module.cuda.max_memory_allocated() >> 20,
            "device_free_mib": free_bytes >> 20,
            "device_total_mib": total_bytes >> 20,
        }

    report = {"gpu": args.gpu, "batch": args.batch, "deployment": deployment["status"],
              "graphs": {key: deployment[key] for key in
                         ("target_graphs", "draft_graphs", "proposal_graphs",
                          "prefix_graphs", "head_graphs")},
              "stages": [], "online_tuning": False}

    with GPULease(args.gpu):
        torch.cuda.reset_peak_memory_stats()
        engine = Engine(config)
        try:
            report["stages"].append(snapshot("after_model_load", torch))
            engine.prepare_reference("reference", args.ref_audio)
            report["stages"].append(snapshot("after_reference", torch))
            engine.prepare_deployment(deployment)
            torch.cuda.synchronize()
            report["stages"].append(snapshot("after_graph_capture", torch))

            torch.cuda.reset_peak_memory_stats()
            identifiers = [f"mem-{i}" for i in range(args.batch)]
            for ident in identifiers:
                engine.create_session(ident, "reference", 0)
                engine.push_text(ident, args.text)
                engine.finish_input(ident)
            pending = set(identifiers)
            while pending:
                events = engine.run_ready()
                if not events:
                    raise RuntimeError("Scheduler returned no events")
                for event in events:
                    pending.discard(event["request_id"])
            torch.cuda.synchronize()
            report["stages"].append(snapshot("peak_during_first_chunk", torch))

            statistics = torch.cuda.memory_stats()
            report["allocator"] = {
                "num_alloc_retries": statistics.get("num_alloc_retries"),
                "num_ooms": statistics.get("num_ooms"),
                "device_allocations": statistics.get("segment.all.allocated"),
                "active_bytes_mib": statistics.get("active_bytes.all.current", 0) >> 20,
                "segment_bytes_mib": statistics.get("segment_bytes.all.current", 0) >> 20,
            }
            for stage in report["stages"]:
                if stage["label"] in ("after_reference", "after_graph_capture"):
                    previous = report["stages"][report["stages"].index(stage) - 1]
                    stage["delta_reserved_mib"] = stage["reserved_mib"] - previous["reserved_mib"]
            free_after, _ = torch.cuda.mem_get_info()
            report["admission_headroom_mib"] = free_after >> 20
        finally:
            engine.close()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
