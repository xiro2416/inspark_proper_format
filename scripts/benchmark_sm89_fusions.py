#!/usr/bin/env python3
"""Numerical and latency checks for portable CFM Triton fusions on SM89."""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import triton
    from acc_infer_clear.kernels.stage2_fusions import _rms_mod, _rope_qkv, _silu_mul

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This benchmark requires exactly one visible SM89 GPU")
    torch.manual_seed(0)

    def measure(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / args.iterations

    results = []

    count = 64 * 512
    a = torch.randn(count, device="cuda")
    b = torch.randn_like(a)
    fused_gate = torch.empty_like(a)
    gate_kernel = lambda: _silu_mul[(triton.cdiv(count, 512),)](
        a, b, fused_gate, count, 512, enable_fp_fusion=False)
    eager_gate = lambda: torch.nn.functional.silu(a) * b
    gate_kernel()
    gate_ref = eager_gate()
    results.append({
        "op": "silu_mul",
        "shape": [64, 512],
        "max_abs": (fused_gate - gate_ref).abs().max().item(),
        "eager_us": measure(eager_gate),
        "triton_us": measure(gate_kernel),
    })

    rows, dim = 64, 512
    x = torch.randn(rows, dim, device="cuda")
    weight = torch.randn(dim, device="cuda")
    fused_rms = torch.empty_like(x)
    rms_kernel = lambda: _rms_mod[(rows,)](
        x, weight, weight, fused_rms, 1, dim, 1e-6, False, 512,
        enable_fp_fusion=False)
    eager_rms = lambda: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    rms_kernel()
    rms_ref = eager_rms()
    results.append({
        "op": "rms_norm",
        "shape": [rows, dim],
        "max_abs": (fused_rms - rms_ref).abs().max().item(),
        "eager_us": measure(eager_rms),
        "triton_us": measure(rms_kernel),
    })

    batch, time, heads, head_dim = 1, 52, 8, 64
    raw = torch.randn(batch, time, 3 * heads * head_dim, device="cuda")
    freq = torch.randn(time, head_dim, device="cuda")
    fused_rope = torch.empty((3, batch, heads, time, head_dim), device="cuda")
    total = batch * heads * time * (head_dim // 2)
    rope_kernel = lambda: _rope_qkv[(triton.cdiv(total, 256),)](
        raw, freq, fused_rope, time, total, heads, head_dim, 256,
        enable_fp_fusion=False)

    def eager_rope():
        qkv = raw.view(batch, time, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        pairs = qkv.view(3, batch, heads, time, head_dim // 2, 2)
        trig = freq.view(time, head_dim // 2, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * trig[None, None, None, :, :, 0] - pairs[..., 1] * trig[None, None, None, :, :, 1]
        out[..., 1] = pairs[..., 1] * trig[None, None, None, :, :, 0] + pairs[..., 0] * trig[None, None, None, :, :, 1]
        out[2] = pairs[2]
        return out.view_as(fused_rope)

    rope_kernel()
    rope_ref = eager_rope()
    results.append({
        "op": "qkv_rope",
        "shape": [batch, time, heads, head_dim],
        "max_abs": (fused_rope - rope_ref).abs().max().item(),
        "eager_us": measure(eager_rope),
        "triton_us": measure(rope_kernel),
    })

    for row in results:
        row["speedup"] = row["eager_us"] / row["triton_us"]
        if row["max_abs"] > 1e-5:
            raise RuntimeError(f"Numerical mismatch: {row}")
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "sm": 89,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
