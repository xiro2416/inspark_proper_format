#!/usr/bin/env python3
"""Benchmark device-control heads in one prepared worker."""
import argparse
import json
import random
import statistics
import subprocess
import sys
import threading
import time


def percentile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "n": len(values), "min": min(values), "median": statistics.median(values),
        "p90": percentile(values, 90), "p95": percentile(values, 95),
        "mean": statistics.fmean(values), "max": max(values),
    }


class BoardSampler:
    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen([
            "nvidia-smi", "-i", str(self.gpu),
            "--query-gpu=timestamp,memory.used,power.draw.instant,utilization.gpu",
            "--format=csv,noheader,nounits", "-lms", "20",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 4:
                    continue
                try:
                    self.samples.append((time.perf_counter(), float(fields[1]),
                                         float(fields[2]), float(fields[3])))
                except ValueError:
                    continue
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        time.sleep(0.15)

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=2)

    def window(self, start, end):
        rows = [row for row in self.samples if start <= row[0] <= end]
        if not rows:
            rows = sorted(self.samples, key=lambda row: min(abs(row[0] - start), abs(row[0] - end)))[:2]
        return {
            "sensor_samples": len(rows),
            "board_memory_mib_mean": statistics.fmean(row[1] for row in rows),
            "board_memory_mib_peak": max(row[1] for row in rows),
            "power_w_mean": statistics.fmean(row[2] for row in rows),
            "power_w_peak": max(row[2] for row in rows),
            "gpu_util_mean": statistics.fmean(row[3] for row in rows),
        }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batches", default="1,8,16")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_triton_device_control.json")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--text", default="他正在整理文件。")
    parser.add_argument("--texts-json", help="JSON array; each text is measured exactly once per batch size")
    parser.add_argument("--repeat-texts", type=int, default=1,
                        help="Repeat the text list this many times; every occurrence gets a unique seed/emotion")
    parser.add_argument("--record-requests", action="store_true",
                        help="Record rounds, accepted counts and generated-code length outside the timed window")
    parser.add_argument("--emotion-seed", type=int, default=20260920)
    parser.add_argument("--power-seconds", type=float, default=0.0,
                        help="Replay the same cases for at least this long per batch for stable board-power sampling")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if not batches or any(value not in (1, 2, 3, 4, 5, 6, 7, 8, 16) for value in batches):
        raise ValueError("batches must be selected from 1..8,16; B32 is excluded because its full graph does not fit")

    from acc_infer_clear.config import load
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.pool import Pool

    config = load(args.config)
    config["max_batch"] = max(batches)
    deployment = load_deployment(args.deployment)

    corpus = None
    if args.texts_json:
        with open(args.texts_json, encoding="utf-8") as handle:
            texts = json.load(handle)
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts-json must contain a non-empty JSON string array")
        if any(len(texts) % batch for batch in batches):
            raise ValueError("text count must be divisible by every selected batch")
        if args.repeat_texts < 1:
            raise ValueError("repeat-texts must be positive")
        texts = texts * args.repeat_texts
        rng = random.Random(args.emotion_seed)
        corpus = []
        for index, text_value in enumerate(texts):
            raw = [rng.random() for _ in range(8)]
            intensity = rng.random()
            total = sum(raw)
            emotion = [intensity * value / total for value in raw]
            corpus.append({"index": index, "text": text_value, "emotion": emotion, "seed": index})

    def admit(pool, cases, prefix):
        identifiers = [f"{prefix}-{case['index']}" for case in cases]
        for case, ident in zip(cases, identifiers):
            pool.create_session(ident, "reference", case["seed"], case["emotion"])
            pool.push_text(ident, case["text"])
            pool.finish_input(ident)
        return identifiers

    def first_chunks(pool, identifiers):
        pending = set(identifiers)
        events = []
        while pending:
            rows = pool.run_ready()
            if not rows:
                raise RuntimeError(f"No event with {len(pending)} requests pending")
            for row in rows:
                if row["request_id"] in pending:
                    pending.remove(row["request_id"])
                    events.append(row)
        return events

    def cancel(pool, identifiers):
        for ident in identifiers:
            pool.cancel(ident)

    baseline_memory = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    report = {"gpu": args.gpu, "batches": batches, "warmups": args.warmups,
              "repeats": args.repeats if corpus is None else None,
              "baseline_board_memory_mib": int(baseline_memory),
              "reference_audio": args.ref_audio, "emotion_seed": args.emotion_seed,
              "emotion_sampling": "uniform direction normalized, uniform total intensity in [0,1]",
              "corpus": corpus}
    with GPULease(args.gpu), Pool(config, args.gpu, 1) as pool:
        pool.prepare_reference("reference", args.ref_audio)
        manifest = pool.prepare_deployment(deployment)[0]
        report["deployment"] = manifest
        results = {}
        for batch in batches:
            fixed_cases = [{"index": index, "text": args.text, "emotion": [0.0] * 8, "seed": index}
                           for index in range(batch)]
            warmup_cases = corpus[:batch] if corpus is not None else fixed_cases
            for warmup in range(args.warmups):
                identifiers = admit(pool, warmup_cases, f"b{batch}-warmup{warmup}")
                first_chunks(pool, identifiers)
                cancel(pool, identifiers)
            groups = ([corpus[offset:offset + batch] for offset in range(0, len(corpus), batch)]
                      if corpus is not None else [fixed_cases for _ in range(args.repeats)])
            before = pool.stats()[0]
            sampler = BoardSampler(args.gpu)
            sampler.start()
            runs = []
            try:
                for repeat, cases in enumerate(groups):
                    identifiers = admit(pool, cases, f"b{batch}-run{repeat}")
                    started = time.perf_counter()
                    events = first_chunks(pool, identifiers)
                    completed = max(event["received"] for event in events)
                    elapsed = completed - started
                    latencies = [(event["received"] - started) * 1000.0 for event in events]
                    runs.append({
                        "repeat": repeat,
                        "all_first_chunks_ms": max(latencies),
                        "per_request_first_chunk_ms": latencies,
                        "first_chunk_requests_per_s": batch / elapsed,
                        "first_chunk_audio_seconds_per_s": sum(
                            event["chunk"]["sample_end"] - event["chunk"]["sample_start"]
                            for event in events) / 22050.0 / elapsed,
                        "board_energy_j_per_request": None,
                        "case_indices": [case["index"] for case in cases],
                        "cfm_batches": sorted({event["chunk"]["cfm_batch"] for event in events}),
                        "vocoder_batches": sorted({event["chunk"]["vocoder_batch"] for event in events}),
                        **sampler.window(started, completed),
                    })
                    if args.record_requests:
                        details=[]
                        for ident,case in zip(identifiers,cases):
                            state=pool.result(ident);accepted=list(state.get("accepted") or [])
                            details.append({"case_index":case["index"],"rounds":state.get("rounds"),
                                            "accepted":accepted,"accepted_sum":sum(accepted),
                                            "generated_codes":len(state.get("codes") or [])})
                        runs[-1]["request_detail"]=details
                    runs[-1]["board_energy_j_per_request"] = runs[-1]["power_w_mean"] * elapsed / batch
                    cancel(pool, identifiers)
            finally:
                sampler.stop()
            after = pool.stats()[0]
            sustained_power = None
            if args.power_seconds > 0:
                sampler = BoardSampler(args.gpu)
                sampler.start()
                power_started = time.perf_counter()
                power_completed = power_started
                power_requests = 0
                power_group = 0
                try:
                    while power_completed - power_started < args.power_seconds:
                        cases = groups[power_group % len(groups)]
                        identifiers = admit(pool, cases, f"b{batch}-power{power_group}")
                        first_chunks(pool, identifiers)
                        power_completed = time.perf_counter()
                        power_requests += len(cases)
                        cancel(pool, identifiers)
                        power_group += 1
                    window = sampler.window(power_started, power_completed)
                finally:
                    sampler.stop()
                elapsed = power_completed - power_started
                sustained_power = {
                    **window,
                    "window_seconds": elapsed,
                    "requests": power_requests,
                    "first_chunk_requests_per_s": power_requests / elapsed,
                    "board_energy_j_per_request": window["power_w_mean"] * elapsed / power_requests,
                }
            pooled = [value for row in runs for value in row["per_request_first_chunk_ms"]]
            results[str(batch)] = {
                "summary": {
                    "all_first_chunks_ms": distribution([row["all_first_chunks_ms"] for row in runs]),
                    "per_request_first_chunk_ms": distribution(pooled),
                    "first_chunk_requests_per_s": distribution([row["first_chunk_requests_per_s"] for row in runs]),
                    "first_chunk_audio_seconds_per_s": distribution([row["first_chunk_audio_seconds_per_s"] for row in runs]),
                    "board_memory_mib_peak": max(row["board_memory_mib_peak"] for row in runs),
                    "board_memory_mib_mean": statistics.fmean(row["board_memory_mib_mean"] for row in runs),
                    "power_w_peak": max(row["power_w_peak"] for row in runs),
                    "power_w_mean": statistics.fmean(row["power_w_mean"] for row in runs),
                    "board_energy_j_per_request": distribution([row["board_energy_j_per_request"] for row in runs]),
                    "requests_measured": sum(len(row["case_indices"]) for row in runs),
                    "device_round_attempts": after["device_round_attempts"] - before["device_round_attempts"],
                    "device_round_successes": after["device_round_successes"] - before["device_round_successes"],
                    "device_round_fallbacks": after["device_round_fallbacks"] - before["device_round_fallbacks"],
                    "native_target_steps": after.get("native_target_steps",0) - before.get("native_target_steps",0),
                    "native_draft_steps": after.get("native_draft_steps",0) - before.get("native_draft_steps",0),
                    "native_draft_compare": after.get("native_draft_compare",[])[len(before.get("native_draft_compare",[])):],
                    # Head CUDA Graph replay bypasses NativeCFMSolver113.__call__, so
                    # calls count graph capture/enqueue preparation rather than replays.
                    # Any non-zero fallback count still proves that an unsupported
                    # shape reached the wrapper while preparing or measuring this run.
                    "native_cfm_backend": after.get("native_cfm_backend"),
                    "native_cfm_capture_calls_total": after.get("native_cfm_calls",0),
                    "native_cfm_calls": after.get("native_cfm_calls",0) - before.get("native_cfm_calls",0),
                    "native_cfm_fallbacks_total": after.get("native_cfm_fallbacks",0),
                    "native_cfm_fallbacks": after.get("native_cfm_fallbacks",0) - before.get("native_cfm_fallbacks",0),
                    "cfm_batches": sorted({value for row in runs for value in row["cfm_batches"]}),
                    "vocoder_batches": sorted({value for row in runs for value in row["vocoder_batches"]}),
                    "sustained_power": sustained_power,
                },
                "runs": runs,
            }
            if args.record_requests:
                detail=[item for row in runs for item in row.get("request_detail",())]
                results[str(batch)]["summary"]["trajectory"]={
                    "requests":len(detail),
                    "rounds":distribution([item["rounds"] for item in detail]),
                    "accepted_sum":distribution([item["accepted_sum"] for item in detail]),
                    "generated_codes":distribution([item["generated_codes"] for item in detail]),
                }
        report["results"] = results
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"json_out": args.json_out, "results": {key: value["summary"] for key, value in results.items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
