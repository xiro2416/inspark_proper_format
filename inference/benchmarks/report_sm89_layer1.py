#!/usr/bin/env python3
"""Build the layer-1 macro-stage report from benchmark_sm89_batch.py outputs.

Whiteboard PROFILE_SM89_WHITEBOARD.md section 7 output table, plus the
attribution gap that section 7 currently cannot see (PCM/code D2H, scheduler and
IPC overhead).

  bash scripts/run.sh scripts/report_sm89_layer1.py outputs/profile_sm89/layer1_b*.json \
      --json-out outputs/profile_sm89/layer1_report.json \
      --csv-out outputs/profile_sm89/layer1_report.csv
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

TOP_LEVEL = ("text_prepare", "prefill", "prefill_rebuild", "draft_verify_accept",
             "overlapped_ar", "speech_codes_d2h", "latent", "condition", "cfm2",
             "vocoder", "pcm_d2h")
SPANS = ("draft", "verify", "accept_commit")


def distribution(values):
    ordered = sorted(values)
    n = len(ordered)

    def pct(q):
        if n == 1:
            return ordered[0]
        position = (n - 1) * q / 100.0
        low = int(position)
        high = min(low + 1, n - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"n": n, "min": ordered[0], "median": statistics.median(ordered),
            "p90": pct(90), "p95": pct(95), "mean": statistics.fmean(ordered), "max": ordered[-1]}


def analyse(path, top_n=12):
    report = json.loads(Path(path).read_text())
    runs = report["runs"]
    per_run = defaultdict(lambda: defaultdict(list))
    span_runs = defaultdict(lambda: defaultdict(list))
    for run in runs:
        profile = run.get("stage_profile") or {}
        by_name = defaultdict(list)
        for stage in profile.get("stages", []):
            by_name[stage["name"]].append(stage)
        for name, group in by_name.items():
            per_run[name]["gpu_ms"].append(sum(s["gpu_ms"] for s in group))
            per_run[name]["host_ms"].append(sum(s["host_ms"] for s in group))
            per_run[name]["calls"].append(len(group))
        for span in profile.get("ar_spans", []):
            span_runs[span["name"]]["gpu_ms"].append(span["gpu_ms"])
            span_runs[span["name"]]["host_ms"].append(span["host_ms"])
            span_runs[span["name"]]["calls"].append(1)

    request_detail = [row for run in runs for row in run.get("request_detail", [])]
    rounds = [row["rounds"] for row in request_detail if row.get("rounds") is not None]
    # With head_batch_barrier the group runs until its SLOWEST member is ready, so the
    # AR loop count is the per-group max, while the median request needs far fewer
    # rounds. The difference is capacity the barrier deliberately idles.
    group_rounds = [max((row["rounds"] or 0) for row in run["request_detail"])
                    for run in runs if run.get("request_detail")]

    def stage_rows(keys, source):
        rows = []
        for name in keys:
            entry = source.get(name)
            if not entry:
                continue
            gpu = entry["gpu_ms"]
            host = entry["host_ms"]
            rows.append({
                "stage": name,
                "calls_per_run": statistics.fmean(entry["calls"]) if entry["calls"] else 0,
                "gpu_ms_per_run": statistics.fmean(gpu),
                "gpu_ms_median": statistics.median(gpu),
                "gpu_ms_p90": distribution(gpu)["p90"],
                "host_ms_per_run": statistics.fmean(host),
                "host_minus_gpu_ms": statistics.fmean(host) - statistics.fmean(gpu),
                "bound": "host/sync" if statistics.fmean(host) > statistics.fmean(gpu) * 1.15
                         else ("gpu" if statistics.fmean(gpu) > statistics.fmean(host) * 1.15 else "balanced"),
            })
        rows.sort(key=lambda row: -row["gpu_ms_per_run"])
        return rows

    stages = stage_rows(TOP_LEVEL, per_run)
    spans = stage_rows(SPANS, span_runs)
    all_ready = distribution([run["batch_complete_ms"] for run in runs])
    # Sum of the top-level stage times as they appear on the single model stream.
    # Anything outside this sum is D2H/PCM handling, scheduler Python and parent IPC,
    # which the phase() instrumentation cannot see with trace_ranges off.
    attributed_gpu = sum(row["gpu_ms_per_run"] for row in stages)
    attributed_host = sum(row["host_ms_per_run"] for row in stages)
    return {
        "source": str(path),
        "gpu": report["gpu"],
        "batch": report["batch"],
        "profile": report["profile"],
        "precision": report["precision"],
        "head_graphs": report["head_graphs"],
        "head_batch_barrier": report["head_batch_barrier"],
        "warmups": report["warmups"],
        "repeats": len(runs),
        "all_requests_ready_ms": all_ready,
        "per_request_first_chunk_ms": distribution(
            [value for run in runs for value in run["request_first_chunk_ms"]]),
        "rounds": distribution(rounds) if rounds else None,
        "group_rounds": distribution(group_rounds) if group_rounds else None,
        "accepted_tokens_per_request": distribution(
            [row["accepted_token_count"] for row in request_detail]) if request_detail else None,
        "first_chunk_samples": report["summary"].get("first_chunk_samples_values"),
        "cfm_batches": sorted({value for run in runs for value in run["cfm_batches"]}),
        "vocoder_batches": sorted({value for run in runs for value in run["vocoder_batches"]}),
        "power_w_mean": report["summary"]["power_w_mean"],
        "power_w_peak": report["summary"]["power_w_peak"],
        "board_energy_j_per_request": report["summary"]["board_energy_j_per_request_mean"],
        "stages": stages[:top_n],
        "ar_spans": spans,
        "attribution": {
            "all_ready_ms": all_ready["mean"],
            "sum_of_stage_gpu_ms": attributed_gpu,
            "sum_of_stage_host_ms": attributed_host,
            "host_view_gap_ms": all_ready["mean"] - attributed_host,
            "host_view_gap_pct": 100.0 * (all_ready["mean"] - attributed_host) / all_ready["mean"],
            "note": (
                "host_ms is the critical-path view: every phase() call opens a host window that is "
                "disjoint from and sequential with the others, so their sum tracks wall time. gpu_ms "
                "is stream occupancy, and it DOUBLE COUNTS whenever a later blocking phase drains an "
                "earlier phase's asynchronously enqueued work (cfm2 + vocoder are absorbed by the "
                "pcm_d2h wait). Never sum gpu_ms as a critical path. A gap in the host view means a "
                "phase is not wrapped at all: with trace_ranges off, speech_codes_d2h and pcm_d2h are "
                "skipped, which is exactly the ~20-30% 'unattributed' bucket seen in pass A. It is the "
                "drain of work already counted in cfm2/vocoder, not unseen work."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--json-out")
    parser.add_argument("--csv-out")
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    rows = [analyse(path, args.top_n) for path in args.sources]
    rows.sort(key=lambda row: row["batch"])

    header = ("stage", "calls/run", "GPU ms/run", "GPU med", "GPU p90", "host ms/run", "host-GPU", "bound")
    for row in rows:
        print("=" * 100)
        print(f"B{row['batch']}  {row['profile']}  head_graphs={row['head_graphs']} "
              f"barrier={row['head_batch_barrier']}  repeats={row['repeats']}")
        print(f"  all-requests-ready ms : mean {row['all_requests_ready_ms']['mean']:.2f} "
              f"median {row['all_requests_ready_ms']['median']:.2f} "
              f"p90 {row['all_requests_ready_ms']['p90']:.2f} "
              f"min {row['all_requests_ready_ms']['min']:.2f} max {row['all_requests_ready_ms']['max']:.2f}")
        print(f"  per-request ms        : median {row['per_request_first_chunk_ms']['median']:.2f} "
              f"p95 {row['per_request_first_chunk_ms']['p95']:.2f} max {row['per_request_first_chunk_ms']['max']:.2f}")
        if row["rounds"]:
            print(f"  AR rounds             : median {row['rounds']['median']:.0f} "
                  f"min {row['rounds']['min']:.0f} max {row['rounds']['max']:.0f}")
        if row["group_rounds"]:
            ar_per_round = next((s["gpu_ms_per_run"] / s["calls_per_run"]
                                 for s in row["stages"] if s["stage"] == "draft_verify_accept"), None)
            extra = row["group_rounds"]["median"] - row["rounds"]["median"]
            cost = extra * ar_per_round if ar_per_round else None
            print(f"  group rounds (barrier): median {row['group_rounds']['median']:.0f} "
                  f"(= slowest member); barriered capacity = {extra:.0f} extra rounds"
                  + (f" ~= {cost:.1f} ms ({100*cost/row['all_requests_ready_ms']['mean']:.1f}% of group latency)"
                     if cost else ""))
            print(f"  accepted tokens/req   : median {row['accepted_tokens_per_request']['median']:.0f}")
        print(f"  first chunk samples   : {row['first_chunk_samples']}   "
              f"cfm_batch {row['cfm_batches']} vocoder_batch {row['vocoder_batches']}")
        print(f"  board power           : mean {row['power_w_mean']:.1f} W peak {row['power_w_peak']:.1f} W   "
              f"J/request {row['board_energy_j_per_request']:.2f}")
        print(f"  {header[0]:<20}{header[1]:>10}{header[2]:>12}{header[3]:>10}{header[4]:>10}{header[5]:>12}{header[6]:>10}{header[7]:>12}")
        for stage in row["stages"]:
            print(f"  {stage['stage']:<20}{stage['calls_per_run']:>10.1f}{stage['gpu_ms_per_run']:>12.3f}"
                  f"{stage['gpu_ms_median']:>10.3f}{stage['gpu_ms_p90']:>10.3f}{stage['host_ms_per_run']:>12.3f}"
                  f"{stage['host_minus_gpu_ms']:>10.3f}{stage['bound']:>12}")
        print("  AR internals:")
        for span in row["ar_spans"]:
            print(f"  {span['stage']:<20}{'':>10}{span['gpu_ms_per_run']:>12.3f}"
                  f"{span['gpu_ms_median']:>10.3f}{span['gpu_ms_p90']:>10.3f}{span['host_ms_per_run']:>12.3f}"
                  f"{span['host_minus_gpu_ms']:>10.3f}{span['bound']:>12}")
        attribution = row["attribution"]
        print(f"  critical path: host windows sum {attribution['sum_of_stage_host_ms']:.2f} ms vs "
              f"{attribution['all_ready_ms']:.2f} ms all-ready -> "
              f"gap {attribution['host_view_gap_ms']:.2f} ms "
              f"({attribution['host_view_gap_pct']:.1f}%)")
        print(f"  stream occupancy sum (double counts drains): {attribution['sum_of_stage_gpu_ms']:.2f} ms")

    if len(rows) > 1:
        stages = []
        for row in rows:
            for stage in row["stages"]:
                stages.append(stage["stage"])
        ordered = [name for name in TOP_LEVEL if name in set(stages)]
        base = rows[0]
        print("=" * 100)
        print("Cross-tier scaling: GPU ms per REQUEST (stage gpu_ms_per_run / batch).")
        print("A stage whose per-request cost barely drops as batch grows is latency/launch "
              "bound at B1; one that scales close to linearly is throughput bound.")
        header = f"  {'stage':<20}" + "".join(f"{'B'+str(r['batch']):>12}" for r in rows)\
                 + f"{'B8/B1':>9}{'B16/B1':>9}"
        print(header)
        for name in ordered:
            per_request = {}
            for row in rows:
                match = next((s for s in row["stages"] if s["stage"] == name), None)
                if match:
                    per_request[row["batch"]] = match["gpu_ms_per_run"] / row["batch"]
            if not per_request:
                continue
            cells = "".join(f"{per_request[b]:>12.3f}" if b in per_request else f"{'-':>12}"
                            for b in (r["batch"] for r in rows))
            first = per_request.get(base["batch"])
            def ratio(batch):
                if first and batch in per_request:
                    return f"{per_request[batch] / first:>9.2f}"
                return f"{'-':>9}"
            print(f"  {name:<20}{cells}{ratio(8) if 8 in per_request else ratio(rows[1]['batch'])}"
                  f"{ratio(16) if 16 in per_request else ratio(rows[-1]['batch'])}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.csv_out:
        with Path(args.csv_out).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["batch", "stage", "calls_per_run", "gpu_ms_per_run", "gpu_ms_median",
                             "gpu_ms_p90", "host_ms_per_run", "host_minus_gpu_ms", "bound"])
            for row in rows:
                for stage in row["stages"] + row["ar_spans"]:
                    writer.writerow([row["batch"], stage["stage"], f"{stage['calls_per_run']:.1f}",
                                     f"{stage['gpu_ms_per_run']:.4f}", f"{stage['gpu_ms_median']:.4f}",
                                     f"{stage['gpu_ms_p90']:.4f}", f"{stage['host_ms_per_run']:.4f}",
                                     f"{stage['host_minus_gpu_ms']:.4f}", stage["bound"]])


if __name__ == "__main__":
    main()
