#!/usr/bin/env python3
"""Measure true concurrent first-chunk latency and board power on one GPU."""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time


def percentile(values, q):
    """Linear-interpolated percentile; q in [0,100]. Small n, so no numpy dependency."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


class PowerSampler:
    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        command = [
            "nvidia-smi", "-i", str(self.gpu),
            "--query-gpu=power.draw.instant,utilization.gpu,clocks.sm",
            "--format=csv,noheader,nounits", "-lms", "20",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3:
                    continue
                try:
                    self.samples.append((time.perf_counter(), *(float(v) for v in fields)))
                except ValueError:
                    continue

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        time.sleep(0.15)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.thread.join(timeout=2)

    def window(self, start, end):
        rows = [row for row in self.samples if start <= row[0] <= end]
        if not rows:
            # The driver's instantaneous sensor may update more slowly than a B1
            # first chunk. Keep the nearest samples and disclose the count.
            rows = sorted(self.samples, key=lambda row: min(abs(row[0] - start), abs(row[0] - end)))[:2]
        power = [row[1] for row in rows]
        utilization = [row[2] for row in rows]
        clocks = [row[3] for row in rows]
        return {
            "samples": len(rows),
            "power_w_mean": statistics.fmean(power),
            "power_w_peak": max(power),
            "gpu_util_mean": statistics.fmean(utilization),
            "sm_clock_mhz_mean": statistics.fmean(clocks),
        }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    parser.add_argument("--batch", type=int, choices=(1, 8, 16, 32), required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_triton.json")
    parser.add_argument("--disable-head-graphs", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--trace-ranges", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--json-out", help="Also write the full report to this path")
    parser.add_argument("--record-requests", action="store_true",
                        help="Record per-request rounds/accepted-token detail before cancelling")
    args = parser.parse_args()

    from acc_infer_clear.runtime.config import load
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.pool import Pool

    config = load(args.config)
    config["max_batch"] = args.batch
    deployment = load_deployment(args.deployment)
    if args.disable_head_graphs:
        deployment["head_graphs"] = False

    def admit(pool, prefix):
        identifiers = [f"{prefix}-{index}" for index in range(args.batch)]
        for index, ident in enumerate(identifiers):
            pool.create_session(ident, "reference", index)
            pool.push_text(ident, args.text)
            pool.finish_input(ident)
        return identifiers

    def cancel(pool, identifiers):
        for ident in identifiers:
            pool.cancel(ident)

    def run_until_first_chunk(pool, identifiers):
        pending = set(identifiers)
        first = {}
        while pending:
            events = pool.run_ready()
            if not events:
                raise RuntimeError(f"Scheduler returned no events with {len(pending)} first chunks pending")
            for event in events:
                ident = event["request_id"]
                if ident in pending:
                    first[ident] = event
                    pending.remove(ident)
        return list(first.values())

    with GPULease(args.gpu), Pool(config, args.gpu, 1) as pool:
        pool.prepare_reference("reference", args.ref_audio)
        manifest = pool.prepare_deployment(deployment)[0]
        if args.profile_stages or args.trace_ranges:
            pool.configure_profiling(args.profile_stages, args.trace_ranges)

        warm_events = []
        for warmup_index in range(args.warmups):
            warm = admit(pool, f"warmup{warmup_index}")
            warm_events = run_until_first_chunk(pool, warm)
            if len(warm_events) != args.batch:
                raise RuntimeError(f"Warmup returned {len(warm_events)} first chunks, expected {args.batch}")
            cancel(pool, warm)

        sampler = PowerSampler(args.gpu)
        sampler.start()
        runs = []
        measurement_started = time.perf_counter()
        measurement_completed = measurement_started
        if args.cuda_profiler_range:
            import torch
            torch.cuda.cudart().cudaProfilerStart()
        try:
            for repeat in range(args.repeats):
                identifiers = admit(pool, f"run{repeat}")
                started = time.perf_counter()
                events = run_until_first_chunk(pool, identifiers)
                if len(events) != args.batch:
                    raise RuntimeError(f"Run returned {len(events)} first chunks, expected {args.batch}")
                completed = max(event["received"] for event in events)
                measurement_completed = completed
                elapsed = completed - started
                latencies = [(event["received"] - started) * 1000.0 for event in events]
                samples = [event["chunk"]["sample_end"] - event["chunk"]["sample_start"] for event in events]
                cfm_batches = [event["chunk"]["cfm_batch"] for event in events]
                vocoder_batches = [event["chunk"]["vocoder_batch"] for event in events]
                power = sampler.window(started, completed)
                stage_profile = pool.take_profile()[0] if args.profile_stages else None
                request_detail = []
                if args.record_requests:
                    for ident in identifiers:
                        state = pool.result(ident)
                        request_detail.append({
                            "id": ident,
                            "rounds": state.get("rounds"),
                            "accepted_tokens": state.get("accepted"),
                            "accepted_token_count": len(state.get("accepted") or []),
                            "total_codes": len(state.get("codes") or []),
                            "kv_head_lengths": state.get("kv_head_lengths"),
                            "eos": state.get("eos"),
                            "chunk_count": len(state.get("chunks") or []),
                            "first_chunk_samples": (state.get("chunks") or [{}])[0].get("sample_end", 0)
                                                   - (state.get("chunks") or [{}])[0].get("sample_start", 0),
                        })
                runs.append({
                    "repeat": repeat,
                    "batch_complete_ms": max(latencies),
                    "request_first_chunk_ms": latencies,
                    "request_first_chunk_ms_distribution": distribution(latencies),
                    "request_detail": request_detail,
                    "first_chunk_requests_per_s": args.batch / elapsed,
                    "first_chunk_audio_samples_per_s": sum(samples) / elapsed,
                    "first_chunk_audio_seconds_per_s": (sum(samples) / 22050.0) / elapsed,
                    "request_first_chunk_ms_min": min(latencies),
                    "request_first_chunk_ms_mean": statistics.fmean(latencies),
                    "request_first_chunk_ms_max": max(latencies),
                    "first_chunk_samples": sorted(set(samples)),
                    "cfm_batches": sorted(set(cfm_batches)),
                    "vocoder_batches": sorted(set(vocoder_batches)),
                    **power,
                    "board_energy_j": power["power_w_mean"] * elapsed,
                    "board_energy_j_per_request": power["power_w_mean"] * elapsed / args.batch,
                    "stage_profile": stage_profile,
                })
                cancel(pool, identifiers)
            aggregate_power = sampler.window(measurement_started, measurement_completed)
        finally:
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStop()
            sampler.stop()

    pooled_request_latencies = [value for run in runs for value in run["request_first_chunk_ms"]]
    report = {
        "gpu": args.gpu,
        "batch": args.batch,
        "profile": manifest["requested"]["status"],
        "precision": manifest["resolved_precision"],
        "head_graphs": deployment["head_graphs"],
        "head_batch_barrier": deployment.get("head_batch_barrier", False),
        "warmups": args.warmups,
        "warmup_first_chunks": len(warm_events),
        "measurement_window": {
            "seconds": measurement_completed - measurement_started,
            **aggregate_power,
        },
        "runs": runs,
        "summary": {
            "first_chunk_ms_mean": statistics.fmean(run["batch_complete_ms"] for run in runs),
            "first_chunk_ms_min": min(run["batch_complete_ms"] for run in runs),
            "first_chunk_ms_max": max(run["batch_complete_ms"] for run in runs),
            "all_requests_ready_ms_distribution": distribution([run["batch_complete_ms"] for run in runs]),
            "per_request_first_chunk_ms_distribution": distribution(pooled_request_latencies),
            "first_chunk_requests_per_s_mean": statistics.fmean(run["first_chunk_requests_per_s"] for run in runs),
            "first_chunk_requests_per_s_median": statistics.median(run["first_chunk_requests_per_s"] for run in runs),
            "first_chunk_audio_seconds_per_s_mean": statistics.fmean(run["first_chunk_audio_seconds_per_s"] for run in runs),
            "power_w_mean": statistics.fmean(run["power_w_mean"] for run in runs),
            "power_w_peak": max(run["power_w_peak"] for run in runs),
            "board_energy_j_per_request_mean": statistics.fmean(run["board_energy_j_per_request"] for run in runs),
        },
        "power_sensor": "NVIDIA instantaneous board power; documented accuracy +/-5 W",
    }
    if args.record_requests:
        detail = [row for run in runs for row in run["request_detail"]]
        report["summary"]["rounds_distribution"] = distribution([row["rounds"] for row in detail if row["rounds"] is not None])
        accepted_counts = [row["accepted_token_count"] for row in detail]
        report["summary"]["accepted_tokens_distribution"] = distribution(accepted_counts) if accepted_counts else None
        report["summary"]["total_codes_distribution"] = distribution([row["total_codes"] for row in detail])
        report["summary"]["first_chunk_samples_values"] = sorted({row["first_chunk_samples"] for row in detail})
    if args.json_out:
        from pathlib import Path
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({"json_out": str(path), "batch": args.batch, "runs": len(runs)}), file=sys.stderr)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
