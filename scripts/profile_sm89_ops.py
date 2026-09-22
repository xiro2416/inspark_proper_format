#!/usr/bin/env python3
"""Layer-2 operator attribution for the SM89 first-chunk path.

Whiteboard PROFILE_SM89_WHITEBOARD.md section 8, "PyTorch Profiler" half.

Runs the engine IN-PROCESS (not through runtime.Pool) so torch.profiler sees the
real operator timeline instead of a multiprocessing boundary. All CUDA Graphs are
disabled through the diagnostic deployment, because a replayed graph hides the
individual operators this report exists to attribute. These numbers are therefore
attribution-only and must never be mixed with the graph-enabled layer-1 latency
table.

Usage:
  bash scripts/run.sh scripts/profile_sm89_ops.py --gpu 6 --batch 1 \
      --deployment outputs/profile_sm89/deployment_nographs_diagnostic.json \
      --ref-audio outputs/profile_sm89/reference.wav \
      --out-dir outputs/profile_sm89

  # Re-analyse an existing trace without touching the GPU:
  bash scripts/run.sh scripts/profile_sm89_ops.py --trace-only \
      outputs/profile_sm89/trace_b1.json --out-dir outputs/profile_sm89
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- trace report

# The runtime's own phase() record_function ranges show up in key_averages with a
# device time that is NOT kernel time (it is the range's window), so they must never
# be mixed into an operator ranking. They are reported separately as by_annotation.
ANNOTATION_NAMES = {
    "text_prepare", "prefill", "prefill_rebuild", "draft_verify_accept", "overlapped_ar",
    "speech_codes_d2h", "latent", "condition", "cfm2", "vocoder", "pcm_d2h",
    "draft", "verify", "accept_commit",
}


def _nesting_parents(events):
    """Assign each event its innermost enclosing event on the same tid."""
    by_tid = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and "ts" in event and "tid" in event:
            by_tid[event["tid"]].append(event)
    parents = {}
    for tid, rows in by_tid.items():
        rows.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack = []
        for event in rows:
            while stack and stack[-1][0] <= event["ts"]:
                stack.pop()
            parents[id(event)] = stack[-1][1] if stack else None
            stack.append((event["ts"] + event["dur"], event))
    return parents


def _ancestors(event, parents):
    seen = 0
    node = parents.get(id(event))
    while node is not None and seen < 256:
        yield node
        node = parents.get(id(node))
        seen += 1


def analyze_trace(path, top_n=30):
    raw = json.loads(Path(path).read_text())
    events = raw["traceEvents"]
    parents = _nesting_parents(events)

    # correlation id -> the cuda_runtime event that launched the kernel
    runtime_by_correlation = {}
    for event in events:
        if event.get("cat") == "cuda_runtime":
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None:
                runtime_by_correlation.setdefault(correlation, event)

    def module_of(kernel_event):
        correlation = (kernel_event.get("args") or {}).get("correlation")
        runtime = runtime_by_correlation.get(correlation)
        if runtime is None:
            return None
        for ancestor in _ancestors(runtime, parents):
            hierarchy = (ancestor.get("args") or {}).get("Module Hierarchy")
            if hierarchy:
                return hierarchy
            if ancestor is runtime and ancestor.get("args", {}).get("Module Hierarchy"):
                return ancestor["args"]["Module Hierarchy"]
        return None

    by_phase = defaultdict(lambda: {"count": 0, "cuda_us": 0.0})
    by_module = defaultdict(lambda: {"count": 0, "cuda_us": 0.0})
    by_kernel = defaultdict(lambda: {"count": 0, "cuda_us": 0.0, "occupancy_pct": [],
                                     "registers": set(), "shared_bytes": set(),
                                     "grid": set(), "block": set()})
    total_kernel_us = 0.0
    kernel_count = 0
    module_resolved = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        duration = float(event.get("dur") or 0.0)
        kernel_count += 1
        total_kernel_us += duration
        entry = by_kernel[event.get("name")]
        entry["count"] += 1
        entry["cuda_us"] += duration
        # The chrome trace carries the launch geometry CUPTI already knows, which
        # covers part of whiteboard section 9 without an ncu replay.
        args = event.get("args") or {}
        if args.get("est. achieved occupancy %") is not None:
            entry["occupancy_pct"].append(float(args["est. achieved occupancy %"]))
        if args.get("registers per thread") is not None:
            entry["registers"].add(int(args["registers per thread"]))
        if args.get("shared memory") is not None:
            entry["shared_bytes"].add(int(args["shared memory"]))
        if args.get("grid"):
            entry["grid"].add(tuple(args["grid"]))
        if args.get("block"):
            entry["block"].add(tuple(args["block"]))
        phase = None
        for ancestor in _ancestors(event, parents):
            if ancestor.get("cat") in ("user_annotation", "gpu_user_annotation"):
                phase = ancestor.get("name")
                break
        phase = phase or "<unattributed>"
        by_phase[phase]["count"] += 1
        by_phase[phase]["cuda_us"] += duration
        module = module_of(event)
        if module:
            module_resolved += 1
            by_module[module]["count"] += 1
            by_module[module]["cuda_us"] += duration

    def ranked(table):
        rows = []
        for name, value in table.items():
            entry = dict(value)
            occupancy = entry.pop("occupancy_pct", [])
            entry["occupancy_pct_mean"] = statistics.fmean(occupancy) if occupancy else None
            for key, label in (("registers", "registers_per_thread"),
                               ("shared_bytes", "shared_memory_bytes"),
                               ("grid", "grid"), ("block", "block")):
                values = entry.pop(key, set())
                entry[label] = sorted(values)[:4] if values else None
            entry["distinct_launch_shapes"] = len(value.get("grid") or []) or None
            rows.append({"name": name, **entry})
        rows.sort(key=lambda row: -row["cuda_us"])
        return rows[:top_n]

    return {
        "trace": str(path),
        "total_kernel_count": kernel_count,
        "total_kernel_us": total_kernel_us,
        "module_attribution_resolved_kernels": module_resolved,
        "module_attribution_coverage": (module_resolved / kernel_count) if kernel_count else None,
        "by_phase": ranked(by_phase),
        "by_module": ranked(by_module),
        "by_kernel": ranked(by_kernel),
    }


# ------------------------------------------------------------------- profiling

def profile_run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.profiler import ProfilerActivity, profile

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")

    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine

    config = load_config(args.config)
    config["max_batch"] = args.batch
    deployment = load_deployment(args.deployment)
    if args.graphs == "diagnostic":
        for key in ("target_graphs", "draft_graphs", "proposal_graphs", "prefix_graphs", "head_graphs"):
            if deployment[key]:
                raise SystemExit(f"Diagnostic deployment must disable {key}; graph replay hides operators")

    def run_first_chunks(engine, prefix):
        identifiers = [f"{prefix}-{i}" for i in range(args.batch)]
        for ident in identifiers:
            engine.create_session(ident, "reference", 0)
            engine.push_text(ident, args.text)
            engine.finish_input(ident)
        pending = set(identifiers)
        while pending:
            events = engine.run_ready()
            if not events:
                raise RuntimeError(f"Scheduler returned no events with {len(pending)} pending")
            for event in events:
                pending.discard(event["request_id"])
        return identifiers

    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_reference("reference", args.ref_audio)
            manifest = engine.prepare_deployment(deployment)
            engine.configure_profiling(enabled=True, trace_ranges=True)
            memory = {"after_deployment": {
                "allocated_mib": torch.cuda.memory_allocated() >> 20,
                "reserved_mib": torch.cuda.memory_reserved() >> 20,
                "max_allocated_mib": torch.cuda.max_memory_allocated() >> 20,
            }}

            for warmup in range(args.warmups):
                for ident in run_first_chunks(engine, f"warmup{warmup}"):
                    engine.cancel(ident)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "graph" if args.graphs == "production" else "nographs"
            trace_path = out_dir / f"trace_{tag}_b{args.batch}.json"
            identifiers = None
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStart()
            try:
                if args.torch_profiler:
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_modules=True,
                        with_stack=False,
                        profile_memory=False,
                    ) as prof:
                        started = time.perf_counter()
                        identifiers = run_first_chunks(engine, "profile")
                        torch.cuda.synchronize()
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                    prof.export_chrome_trace(str(trace_path))
                else:
                    started = time.perf_counter()
                    identifiers = run_first_chunks(engine, "profile")
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    trace_path = None
                    prof = None
            finally:
                if args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStop()
            memory["after_profile_run"] = {
                "max_allocated_mib": torch.cuda.max_memory_allocated() >> 20,
                "reserved_mib": torch.cuda.memory_reserved() >> 20,
            }
            stage_profile = engine.take_profile()

            ops = None
            if prof is not None:
                averages = prof.key_averages(group_by_input_shape=True)

                # Field name in the report -> FunctionEventAvg attribute holding it.
                attribute_of = {
                    "self_cuda_ms": "self_device_time_total",
                    "total_cuda_ms": "device_time_total",
                    "self_cpu_ms": "self_cpu_time_total",
                }

                def rows(table, field, keep_annotations=False, limit=None):
                    attribute = attribute_of[field]
                    out = []
                    for row in table:
                        is_annotation = row.key in ANNOTATION_NAMES
                        if is_annotation != keep_annotations:
                            continue
                        value = getattr(row, attribute, 0.0)
                        if value <= 0:
                            continue
                        out.append({
                            "name": row.key,
                            "count": row.count,
                            "self_cuda_ms": row.self_device_time_total / 1000.0,
                            "total_cuda_ms": row.device_time_total / 1000.0,
                            "self_cpu_ms": row.self_cpu_time_total / 1000.0,
                            "input_shapes": str(getattr(row, "input_shapes", ""))[:200],
                        })
                    out.sort(key=lambda r: -r[field])
                    return out[:limit or args.top_n]
                ops = {
                    "top_self_cuda": rows(averages, "self_cuda_ms"),
                    "top_total_cuda": rows(averages, "total_cuda_ms"),
                    "top_self_cpu": rows(averages, "self_cpu_ms"),
                    "by_annotation": rows(averages, "self_cuda_ms", keep_annotations=True),
                }
                # torch 2.8 does not write "Module Hierarchy" into the chrome trace, so
                # take module attribution from FunctionEvent.modules instead. Use only
                # the innermost module of each chain, otherwise nested modules double count.
                module_totals = defaultdict(lambda: {"count": 0, "cuda_us": 0.0})
                attributed = 0
                for event in prof.events():
                    modules = getattr(event, "modules", None)
                    if not modules:
                        continue
                    duration = getattr(event, "device_time_total", 0.0)
                    if duration <= 0:
                        continue
                    innermost = modules[-1]
                    module_totals[innermost]["count"] += 1
                    module_totals[innermost]["cuda_us"] += duration
                    attributed += 1
                module_rows = [{"name": name, "count": value["count"],
                                "cuda_us": value["cuda_us"], "cuda_ms": value["cuda_us"] / 1000.0}
                               for name, value in module_totals.items()]
                module_rows.sort(key=lambda row: -row["cuda_us"])
                ops["by_module"] = module_rows[:args.top_n]
                ops["module_events_attributed"] = attributed
            report = {
                "gpu": args.gpu,
                "batch": args.batch,
                "deployment_status": manifest["requested"]["status"],
                "graphs": {k: deployment[k] for k in
                           ("target_graphs", "draft_graphs", "proposal_graphs", "prefix_graphs", "head_graphs")},
                "in_process": True,
                "warmups": args.warmups,
                "profiled_wall_ms": elapsed_ms,
                "per_request_first_chunk_note": "profiler serializes; use layer-1 for latency",
                "memory": memory,
                "torch_profiler": args.torch_profiler,
                "cuda_profiler_range": args.cuda_profiler_range,
                "trace": str(trace_path) if trace_path else None,
                "ops": ops,
                "stage_profile": stage_profile,
            }
            for ident in identifiers or []:
                state = engine.sessions.get(ident)
                if state:
                    report.setdefault("requests", []).append({
                        "id": ident,
                        "rounds": state.get("rounds"),
                        "accepted_tokens": list(state.get("accepted") or []),
                        "total_codes": len(state.get("codes") or []),
                        "first_chunk_samples": (state["chunks"][0]["sample_end"] - state["chunks"][0]["sample_start"])
                                               if state.get("chunks") else None,
                    })
            return report, trace_path
        finally:
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="outputs/profile_sm89/deployment_nographs_diagnostic.json")
    parser.add_argument("--ref-audio", default="outputs/profile_sm89/reference.wav")
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--out-dir", default="outputs/profile_sm89")
    parser.add_argument("--graphs", choices=("diagnostic", "production"), default="diagnostic",
                        help="diagnostic requires every graph disabled; production uses the plan as given")
    parser.add_argument("--torch-profiler", dest="torch_profiler", action="store_true", default=True)
    parser.add_argument("--no-torch-profiler", dest="torch_profiler", action="store_false",
                        help="Skip torch.profiler; use when nsys owns the capture (avoids double instrumentation)")
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="Bracket the measured run with cudaProfilerStart/Stop for nsys capture-range")
    parser.add_argument("--json-out")
    parser.add_argument("--trace-only", help="Analyse an existing chrome trace and exit")
    args = parser.parse_args()

    if args.trace_only:
        report = analyze_trace(args.trace_only, args.top_n)
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.json_out:
            Path(args.json_out).write_text(text)
        print(text)
        return

    report, trace_path = profile_run(args)
    if trace_path is not None:
        report["trace_report"] = analyze_trace(trace_path, args.top_n)
    tag = "graph" if args.graphs == "production" else "nographs"
    text = json.dumps(report, indent=2, ensure_ascii=False)
    out = Path(args.json_out) if args.json_out else Path(args.out_dir) / f"ops_{tag}_b{args.batch}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(json.dumps({
        "ops_report": str(out),
        "trace": str(trace_path) if trace_path else None,
        "profiled_wall_ms": report["profiled_wall_ms"],
        "module_attribution_coverage": (report.get("trace_report") or {}).get("module_attribution_coverage"),
    }, indent=2), file=sys.stderr)
    if args.torch_profiler:
        print(text)


if __name__ == "__main__":
    main()
