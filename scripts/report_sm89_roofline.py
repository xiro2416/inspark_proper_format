#!/usr/bin/env python3
"""Shape-derived roofline for the SM89 profile, standing in for the blocked ncu layer.

Whiteboard section 9 wants counter-backed bottleneck labels. This environment cannot
read CUPTI counters (§5 of the report), so this script does the next best thing and
labels itself honestly: it computes FLOPs and minimum bytes moved from the real shapes
the PyTorch Profiler recorded, compares against the RTX 4090 peak, and reports how far
each operator sits from its own roofline.

READ THIS AS AN ESTIMATE, NOT A MEASUREMENT. It assumes ideal caches, no launch
overhead and perfect overlap. Its value is directional: it decides whether an operator
has any headroom left at all, which is what "should we write a faster kernel for this"
actually hinges on.

Reference peaks for the RTX 4090 (public specifications, dense, no sparsity):
  bf16 tensor core with fp32 accumulate : 165.2 TFLOP/s
  fp32                                  :  82.6 TFLOP/s
  DRAM bandwidth                        : 1008 GB/s
  L2                                    :  72 MiB
  SMs                                   : 128

  bash scripts/run.sh scripts/report_sm89_roofline.py \
      outputs/profile_sm89/ops_nographs_b1.json \
      outputs/profile_sm89/ops_nographs_b8.json \
      --json-out outputs/profile_sm89/roofline.json
"""
import argparse
import ast
import json
from pathlib import Path

PEAK_BF16 = 165.2e12      # FLOP/s
PEAK_FP32 = 82.6e12
DRAM_BW = 1008e9          # B/s
L2_BYTES = 72 << 20
RIDGE_BF16 = PEAK_BF16 / DRAM_BW   # FLOP/byte at which the two roofs cross


def parse_shapes(row):
    try:
        return ast.literal_eval(row["input_shapes"])
    except (ValueError, SyntaxError, KeyError):
        return None


def gemm_roofline(m, n, k, bytes_per_element=2):
    flops = 2.0 * m * n * k
    weight_bytes = k * n * bytes_per_element
    activation_bytes = (m * k + m * n) * bytes_per_element
    total_bytes = weight_bytes + activation_bytes
    intensity = flops / total_bytes if total_bytes else 0.0
    compute_ms = flops / PEAK_BF16 * 1e3
    memory_ms = total_bytes / DRAM_BW * 1e3
    return {
        "m": m, "n": n, "k": k,
        "flops": flops,
        "weight_mib": weight_bytes / (1 << 20),
        "bytes": total_bytes,
        "arithmetic_intensity": intensity,
        "bound": "memory" if intensity < RIDGE_BF16 else "compute",
        "roofline_ms": max(compute_ms, memory_ms),
        "compute_roof_ms": compute_ms,
        "memory_roof_ms": memory_ms,
        # Does the weight matrix fit in L2 at all? If not, every call streams it from DRAM.
        "weights_fit_l2": weight_bytes <= L2_BYTES,
    }


def analyse(path):
    report = json.loads(Path(path).read_text())
    batch = report["batch"]
    ops = report["ops"]
    gemms = []
    for row in ops["top_self_cuda"]:
        if row["name"] != "aten::addmm":
            continue
        shapes = parse_shapes(row)
        if not shapes or len(shapes) < 3:
            continue
        m_k, k_n = shapes[1], shapes[2]
        if len(m_k) != 2 or len(k_n) != 2:
            continue
        m, k = m_k
        k2, n = k_n
        if k != k2:
            continue
        entry = gemm_roofline(m, n, k)
        entry["count"] = row["count"]
        entry["measured_ms_total"] = row["self_cuda_ms"]
        entry["measured_ms_per_call"] = row["self_cuda_ms"] / row["count"]
        entry["pct_of_roofline"] = (entry["roofline_ms"] / entry["measured_ms_per_call"] * 100.0
                                    if entry["measured_ms_per_call"] else None)
        gemms.append(entry)

    gemms.sort(key=lambda row: -row["measured_ms_total"])
    # Per-round weight traffic: the transformer block projection weights that must be
    # re-streamed for every speculative verification round.
    rounds = None
    weight_bytes_per_round = 0.0
    for entry in gemms:
        weight_bytes_per_round += entry["bytes"] * entry["count"]
    # One round's worth = total divided by the number of rounds the run covered.
    # The profiler run served one request to its first chunk; derive rounds from the
    # addmm count of a single shape (24 layers per round for this GPT).
    # The verification projections fire once per transformer layer per round, so the
    # largest per-shape addmm count divided by the layer count is the round count.
    # (Smaller counts belong to the proposal/CFM paths, which are not per-round.)
    per_shape_counts = {row["count"] for row in ops["top_self_cuda"] if row["name"] == "aten::addmm"}
    if per_shape_counts:
        rounds = max(per_shape_counts) / 24.0

    return {
        "source": str(path),
        "batch": batch,
        "profiled_wall_ms": report["profiled_wall_ms"],
        "rounds_in_run": rounds,
        "gemms": gemms,
        "total_gemm_measured_ms": sum(entry["measured_ms_total"] for entry in gemms),
        "total_weight_streamed_mib": weight_bytes_per_round / (1 << 20),
        "weight_stream_floor_ms": weight_bytes_per_round / DRAM_BW * 1e3,
        "roofline_peak": {"bf16_tflops": PEAK_BF16 / 1e12, "dram_gb_s": DRAM_BW / 1e9,
                          "ridge_flop_per_byte": RIDGE_BF16, "l2_mib": L2_BYTES >> 20},
        "caveat": "Shape-derived estimate with ideal caches and no launch overhead. "
                  "Not a counter measurement. See report section 5.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    reports = [analyse(path) for path in args.sources]
    for report in reports:
        peak = report["roofline_peak"]
        print("=" * 108)
        print(f"{Path(report['source']).name}   batch={report['batch']}   "
              f"profiled_wall={report['profiled_wall_ms']:.0f} ms   "
              f"rounds_in_run={report['rounds_in_run']:.0f}")
        print(f"  reference peaks: bf16 {peak['bf16_tflops']:.1f} TFLOP/s, "
              f"DRAM {peak['dram_gb_s']:.0f} GB/s, ridge {peak['ridge_flop_per_byte']:.0f} FLOP/byte, "
              f"L2 {peak['l2_mib']} MiB")
        print(f"  {'M':>4}{'N':>7}{'K':>7}{'calls':>7}{'meas ms':>11}{'roofline ms':>13}"
              f"{'% of roof':>11}{'AI':>9}{'bound':>9}{'w MiB':>9}{'in L2':>7}")
        for entry in report["gemms"]:
            print(f"  {entry['m']:>4}{entry['n']:>7}{entry['k']:>7}{entry['count']:>7}"
                  f"{entry['measured_ms_total']:>11.3f}{entry['roofline_ms']:>13.4f}"
                  f"{entry['pct_of_roofline']:>10.1f}%{entry['arithmetic_intensity']:>9.1f}"
                  f"{entry['bound']:>9}{entry['weight_mib']:>9.2f}"
                  f"{str(entry['weights_fit_l2']):>7}")
        print(f"  projection GEMMs measured total : {report['total_gemm_measured_ms']:.3f} ms")
        print(f"  weight bytes streamed over run  : {report['total_weight_streamed_mib']:.1f} MiB")
        print(f"  DRAM floor for that traffic     : {report['weight_stream_floor_ms']:.3f} ms")
        print(f"  NOTE: {report['caveat']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
