#!/usr/bin/env python3
"""Isolated production Target KV append/attention benchmark for TRT comparison."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from acc_infer_clear.kernels.kv_attention import append, attention


def measure(call, warmups, iterations):
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations // 20):
            call()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / (iterations // 20))
    return {"mean_ms": statistics.fmean(samples), "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/sm89_target_attention_benchmark.json")
    p.add_argument("--warmups", type=int, default=200)
    p.add_argument("--iterations", type=int, default=2000)
    args = p.parse_args()
    torch.cuda.set_device(0)
    out = []
    for batch in (1, 4, 8, 16):
        heads, query, dim, capacity, limit = 20, 8, 64, 2048, 128
        q = torch.randn(batch, heads, query, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        pk = torch.randn(batch, heads, capacity, dim, device="cuda", dtype=torch.bfloat16)
        pv = torch.randn_like(pk)
        keep = torch.ones(batch, capacity, device="cuda", dtype=torch.int32)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        lengths = torch.full((batch,), 120, device="cuda", dtype=torch.int32)
        cases = {
            "attention": lambda: attention(q, pk, pv, keep, slots, lengths, limit),
            "append": lambda: append(k, v, pk, pv, slots, lengths),
            "append_attention": lambda: (append(k, v, pk, pv, slots, lengths),
                                         attention(q, pk, pv, keep, slots, lengths, limit)),
        }
        row = {"batch": batch}
        for name, call in cases.items():
            row[name] = measure(call, args.warmups, args.iterations)
        out.append(row)
        print(batch, {k: round(v["median_ms"] * 1000, 2) for k, v in row.items() if k != "batch"})
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
