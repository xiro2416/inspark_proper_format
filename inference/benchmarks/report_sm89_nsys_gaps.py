#!/usr/bin/env python3
"""Per-range GPU idle analysis over an Nsight Systems sqlite export.

Answers the layer-2 question "where does the stream sit idle while the host works",
independently of the CUDA-event numbers in layer 1. nsys instruments the process, so
the absolute values are inflated; the RATIO of kernel time to range wall time inside
each NVTX range is the meaningful signal, and it should agree with layer 1's
host-minus-gpu column.

Ranges that only enqueue work (cfm2, vocoder) legitimately show ~zero kernel time
inside: their kernels execute later, during the pcm_d2h drain. Read this table
together with layer 1, not instead of it.

  nsys export --type sqlite --output X.sqlite X.nsys-rep   # done by extract_nsys.sh
  bash scripts/run.sh scripts/report_sm89_nsys_gaps.py outputs/profile_sm89/nsys_graph_b1.sqlite
"""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def pick_table(connection, candidates):
    names = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def analyse(path, top_n=20):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    nvtx = pick_table(connection, ["NVTX_EVENTS"])
    kernel = pick_table(connection, ["CUPTI_ACTIVITY_KIND_KERNEL"])
    if not nvtx or not kernel:
        return {"error": "missing NVTX_EVENTS or CUPTI_ACTIVITY_KIND_KERNEL",
                "tables": sorted(row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"))}

    ranges = connection.execute(
        f"SELECT text, start, end FROM {nvtx} WHERE end IS NOT NULL AND text IS NOT NULL"
    ).fetchall()
    kernels = connection.execute(
        f"SELECT start, end, demangledName FROM {kernel} WHERE end IS NOT NULL"
    ).fetchall()

    # Kernel events are few enough that a sort + sweep per range is fine.
    kernel_by_start = sorted(kernels, key=lambda row: row[0])

    def kernels_in(start, end):
        total = 0
        count = 0
        for k_start, k_end, _name in kernel_by_start:
            if k_start >= end:
                break
            if k_end <= end and k_start >= start:
                total += k_end - k_start
                count += 1
        return total, count

    grouped = defaultdict(lambda: {"count": 0, "wall_ns": 0, "kernel_ns": 0, "kernels": 0})
    for text, start, end in ranges:
        entry = grouped[text]
        entry["count"] += 1
        entry["wall_ns"] += end - start
        kernel_ns, kernel_count = kernels_in(start, end)
        entry["kernel_ns"] += kernel_ns
        entry["kernels"] += kernel_count

    rows = []
    for text, entry in grouped.items():
        wall = entry["wall_ns"]
        busy = entry["kernel_ns"]
        rows.append({
            "range": text,
            "count": entry["count"],
            "wall_ms_total": wall / 1e6,
            "kernel_ms_total": busy / 1e6,
            "gpu_busy_pct": (100.0 * busy / wall) if wall else None,
            "idle_ms_total": (wall - busy) / 1e6,
            "kernels": entry["kernels"],
        })
    rows.sort(key=lambda row: -row["idle_ms_total"])

    window_start = min(row[1] for row in ranges)
    window_end = max(row[2] for row in ranges)
    window_kernel_ns = sum(k_end - k_start for k_start, k_end, _ in kernel_by_start
                           if k_start >= window_start and k_end <= window_end)

    connection.close()
    return {
        "sqlite": str(path),
        "measured_window_ms": (window_end - window_start) / 1e6,
        "kernel_time_in_window_ms": window_kernel_ns / 1e6,
        "gpu_busy_pct_in_window": 100.0 * window_kernel_ns / (window_end - window_start),
        "ranges_by_idle": rows[:top_n],
        "note": "Enqueue-only ranges (cfm2, vocoder) show ~0% busy by construction; "
                "their kernels land inside the pcm_d2h range. Compare gpu_busy_pct for "
                "blocking ranges (draft_verify_accept, accept_commit) against layer 1.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+")
    parser.add_argument("--json-out")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    reports = [analyse(path, args.top_n) for path in args.sqlite]
    for report in reports:
        print("=" * 96)
        print(Path(report.get("sqlite", "?")).name)
        if "error" in report:
            print("  ", report["error"])
            continue
        print(f"  measured window {report['measured_window_ms']:.2f} ms   "
              f"kernel time {report['kernel_time_in_window_ms']:.2f} ms   "
              f"GPU busy {report['gpu_busy_pct_in_window']:.1f}%")
        print(f"  {'range':<24}{'count':>7}{'wall ms':>12}{'kernel ms':>12}{'busy %':>9}{'idle ms':>11}{'kernels':>9}")
        for row in report["ranges_by_idle"]:
            busy = f"{row['gpu_busy_pct']:.1f}" if row["gpu_busy_pct"] is not None else "-"
            print(f"  {row['range']:<24}{row['count']:>7}{row['wall_ms_total']:>12.2f}"
                  f"{row['kernel_ms_total']:>12.2f}{busy:>9}{row['idle_ms_total']:>11.2f}{row['kernels']:>9}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
