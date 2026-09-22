#!/usr/bin/env python3
"""Validate the Triton Target LayerNorm+FP8 quantization fusion on SM89."""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import triton
    from acc_infer_clear.ops.triton.fp8 import _quantize
    from acc_infer_clear.ops.triton.target_norm_quant.triton_fused import fused

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
    for batch in (1, 8):
        rows, n = batch * 8, 1280
        x = torch.randn(batch, 8, n, device="cuda")
        weight = torch.randn(n, device="cuda")
        bias = torch.randn(n, device="cuda")
        q = torch.empty((rows, n), device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.empty(rows, device="cuda")

        def separate():
            y = torch.nn.functional.layer_norm(x, (n,), weight, bias, 1e-5)
            _quantize[(rows,)](
                y, q, scale, n, triton.next_power_of_2(n),
                enable_fp_fusion=False)
            return y

        fused_fn = lambda: fused(x, weight, bias, 1e-5, True)
        reference = torch.nn.functional.layer_norm(x, (n,), weight, bias, 1e-5)
        fused_q, fused_scale, fused_y = fused_fn()
        reconstructed = fused_q.float() * fused_scale[:, None]
        reference_flat = reference.view(rows, n)
        separate_us = measure(separate)
        fused_us = measure(fused_fn)
        results.append({
            "batch": batch,
            "rows": rows,
            "separate_us": separate_us,
            "fused_us": fused_us,
            "speedup": separate_us / fused_us,
            "layernorm_max_abs": (fused_y - reference).abs().max().item(),
            "layernorm_rmse": (fused_y - reference).square().mean().sqrt().item(),
            "dequant_max_abs": (reconstructed - reference_flat).abs().max().item(),
            "dequant_rmse": (reconstructed - reference_flat).square().mean().sqrt().item(),
        })

    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "sm": 89,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "kernel": "target_layernorm_fp8_quant",
        "online_tuning": False,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
