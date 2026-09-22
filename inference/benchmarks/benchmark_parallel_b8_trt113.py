#!/usr/bin/env python3
"""Incremental single-process B8-lane concurrency/OOM experiment."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class PowerSampler:
    def __init__(self, gpu):
        self.gpu = gpu; self.rows = []; self.process = None; self.thread = None

    def start(self):
        self.process = subprocess.Popen([
            "nvidia-smi", "-i", str(self.gpu),
            "--query-gpu=timestamp,memory.used,power.draw.instant,utilization.gpu,clocks.sm",
            "--format=csv,noheader,nounits", "-lms", "20",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                fields = [part.strip() for part in line.split(",")]
                if len(fields) != 5: continue
                try: self.rows.append((time.perf_counter(), *(float(v) for v in fields[1:])))
                except ValueError: pass
        self.thread = threading.Thread(target=collect, daemon=True); self.thread.start(); time.sleep(.15)

    def stop(self):
        self.process.terminate()
        try: self.process.wait(timeout=2)
        except subprocess.TimeoutExpired: self.process.kill()
        self.thread.join(timeout=2)

    def summarize(self, start, end):
        rows = [row for row in self.rows if start <= row[0] <= end]
        if not rows: return {"samples": 0}
        return {
            "samples": len(rows), "memory_mib_mean": statistics.fmean(r[1] for r in rows),
            "memory_mib_peak": max(r[1] for r in rows),
            "power_w_mean": statistics.fmean(r[2] for r in rows),
            "power_w_peak": max(r[2] for r in rows),
            "gpu_util_mean": statistics.fmean(r[3] for r in rows),
            "sm_clock_mhz_mean": statistics.fmean(r[4] for r in rows),
        }


def distribution(values):
    ordered = sorted(values)
    def percentile(q):
        p = (len(ordered) - 1) * q / 100; lo = int(p); hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)
    return {"n": len(values), "min": min(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "p95": percentile(95), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--max-lanes", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--power-seconds", type=float, default=5.0)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_trt113_full_b8.json")
    parser.add_argument("--ref-audio", default="/workspace/index-tts/data/audio/old/mingxiang_gao.wav")
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--emotion-mode", choices=("zero", "cycle-half"), default="zero",
                        help="zero matches the historical B8 latency baseline; cycle-half sets one of the first four emotion axes to 0.5")
    parser.add_argument("--output", default="outputs/parallel_b8_trt113_dirty_gpu6.json")
    parser.add_argument("--existing-memory-limit-mib", type=int, default=23000)
    args = parser.parse_args()
    if args.max_lanes < 2: raise ValueError("max-lanes must be at least 2")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["ACC_GPU_EXISTING_MEMORY_LIMIT_MIB"] = str(args.existing_memory_limit_mib)

    import numpy as np
    import torch
    from acc_infer_clear.runtime.config import load
    from benchmarks.parallel_b8 import fork_engine, install_b8_only_graph_policy, install_shared_engine_classes, make_lane_plan, memory_snapshot
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.engine import Engine

    install_b8_only_graph_policy(); install_shared_engine_classes()
    config = load(args.config); config["max_batch"] = 8
    deployment = load_deployment(args.deployment)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    board_before = int(subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    report = {
        "schema": 1, "gpu": args.gpu, "compute_capability": "8.9",
        "pdl": {"enabled": False, "reason": "CUDA PDL requires compute capability >= 9.0"},
        "mechanism": "single process; shared immutable TRT engines/model weights; one context/stream/graph/cache bank per B8 lane",
        "workload": {"text": args.text, "emotion_mode": args.emotion_mode,
                     "reference_audio": args.ref_audio},
        "dirty_gpu_baseline_mib": board_before, "results": [], "oom": None,
    }

    def save(): output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lanes = []; root = None; wave_counter = 0; executors = []; active_executor = None
    with GPULease(args.gpu):
        try:
            root = Engine(config); root.prepare_reference("reference", args.ref_audio)
            root_manifest = root.prepare_deployment(deployment); lanes.append(root)
            report["root_deployment"] = root_manifest["requested"]["status"]
            report["after_root"] = memory_snapshot(torch); save()

            def run_wave(prefix, capture_quality=False):
                nonlocal wave_counter
                wave_counter += 1; prefix = f"{prefix}-{wave_counter}"
                identifiers = []
                for lane_index, lane in enumerate(lanes):
                    ids = []
                    for row in range(8):
                        ident = f"{prefix}-l{lane_index}-r{row}"; ids.append(ident)
                        emotion = [0.0] * 8
                        if args.emotion_mode == "cycle-half":
                            emotion[row % 4] = .5
                        lane.create_session(ident, "reference", row, emotion)
                        lane.push_text(ident, args.text); lane.finish_input(ident)
                    identifiers.append(ids)
                barrier = threading.Barrier(len(lanes))

                def advance(pair):
                    lane, ids = pair; pending = set(ids); events = []
                    barrier.wait()
                    while pending:
                        emitted = lane.run_ready()
                        if not emitted: raise RuntimeError(f"Lane stalled with {len(pending)} requests")
                        for event in emitted:
                            if event["request_id"] in pending:
                                pending.remove(event["request_id"]); events.append(event)
                    return events

                started = time.perf_counter()
                lane_events = list(active_executor.map(advance, zip(lanes, identifiers)))
                completed = time.perf_counter(); quality = []
                for lane_index, (lane, ids, events) in enumerate(zip(lanes, identifiers, lane_events)):
                    event_by_id = {event["request_id"]: event for event in events}
                    lane_quality = []
                    for row, ident in enumerate(ids):
                        result = lane.sessions[ident]
                        pcm = event_by_id[ident]["chunk"]["pcm"]
                        if capture_quality:
                            lane_quality.append((list(result["codes"]), pcm.copy()))
                        lane.cancel(ident)
                    if lane_quality: quality.append(lane_quality)
                return {"wall_ms": (completed - started) * 1000, "requests": 8 * len(lanes),
                        "requests_per_s": 8 * len(lanes) / (completed - started), "quality": quality}

            for target_lanes in range(1, args.max_lanes + 1):
                if target_lanes > 1:
                    partial = None
                    try:
                        before = memory_snapshot(torch); partial = fork_engine(root)
                        partial.prepare_deployment(make_lane_plan(deployment, target_lanes - 1))
                        lanes.append(partial); torch.cuda.synchronize()
                        report.setdefault("lane_allocations", []).append({
                            "lanes": target_lanes, "before": before, "after": memory_snapshot(torch)})
                        save()
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.OutOfMemoryError): raise
                        report["oom"] = {"attempted_lanes": target_lanes,
                                         "stage": "lane construction/deployment/CUDA Graph capture",
                                         "error": str(exc), "memory": memory_snapshot(torch)}
                        save()
                        print(json.dumps({"output": str(output), "results": len(report["results"]),
                                          "oom": report["oom"]}, ensure_ascii=False, indent=2), flush=True)
                        # A failed CUDA Graph capture can leave allocator-owned
                        # graph pools unsafe to destruct.  The experiment is a
                        # subprocess, so exit after persisting the complete OOM
                        # record and let the driver reclaim the CUDA context.
                        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
                # Production lane workers are persistent.  Recreating and
                # joining an executor for every wave adds host-only latency.
                active_executor = ThreadPoolExecutor(max_workers=len(lanes), thread_name_prefix="b8-lane")
                executors.append(active_executor)
                for warmup in range(args.warmups): run_wave(f"warmup-n{target_lanes}-{warmup}")
                quality_wave = run_wave(f"quality-n{target_lanes}", capture_quality=True)
                reference = quality_wave["quality"][0]
                comparisons = []
                for lane_rows in quality_wave["quality"]:
                    for row, (codes, pcm) in enumerate(lane_rows):
                        expected_codes, expected_pcm = reference[row]
                        comparisons.append({
                            "codes_equal": codes == expected_codes,
                            "pcm_equal": bool(np.array_equal(pcm, expected_pcm)),
                            "pcm_max_lsb": int(np.max(np.abs(
                                pcm.astype(np.int32) - expected_pcm.astype(np.int32)))),
                        })
                quality = {"reference": "lane0 in the same concurrent wave",
                           "codes_all_equal": all(row["codes_equal"] for row in comparisons),
                           "pcm_all_equal": all(row["pcm_equal"] for row in comparisons),
                           "pcm_max_lsb": max(row["pcm_max_lsb"] for row in comparisons)}
                sampler = PowerSampler(args.gpu); sampler.start(); start = time.perf_counter(); waves = []
                try:
                    while time.perf_counter() - start < args.power_seconds:
                        waves.append(run_wave(f"power-n{target_lanes}"))
                finally:
                    end = time.perf_counter(); sampler.stop()
                wall = [wave["wall_ms"] for wave in waves]
                req_s = [wave["requests_per_s"] for wave in waves]
                power = sampler.summarize(start, end)
                result = {"lanes": target_lanes, "concurrent_requests": 8 * target_lanes,
                          "waves": len(waves), "wall_ms": distribution(wall),
                          "requests_per_s": distribution(req_s), "quality": quality,
                          "power": power, "memory": memory_snapshot(torch)}
                if power.get("power_w_mean"):
                    result["joules_per_request"] = power["power_w_mean"] / statistics.fmean(req_s)
                report["results"].append(result); save()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if not quality["codes_all_equal"] or not quality["pcm_all_equal"]:
                    report["quality_failure_at_lanes"] = target_lanes; save(); break
        finally:
            for executor in executors:
                executor.shutdown(wait=True)
            for lane in reversed(lanes[1:]):
                try: lane.close()
                except Exception: pass
            if root is not None: root.close()
    save(); print(json.dumps({"output": str(output), "results": len(report["results"]),
                              "oom": report["oom"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
