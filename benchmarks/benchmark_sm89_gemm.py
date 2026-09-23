#!/usr/bin/env python3
"""Compare generic BF16/FP8 GEMM paths without an online tile search."""
import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import triton
    from inspark_infer.ops.triton.fp8 import _gemm, _quantize, linear
    from inspark_infer.quantization.weights import pack_weight
    from inspark_infer.ops.planning.planner import Tile
    from inspark_infer.ops.triton.stage2_gemm import _epilogue

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This benchmark requires exactly one visible SM89 GPU")

    # One conservative functional launch shape, deliberately not an autotune sweep.
    tile = Tile(bm=16, bn=64, bk=64, warps=4, stages=2)
    shapes = [
        (8, 3840, 1280, "target_qkv_b1"),
        (64, 1280, 1280, "target_out_b8"),
        (310, 1536, 512, "cfm_projection"),
    ]
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

    rows = []
    for m, n, k, label in shapes:
        x = torch.randn(m, k, device="cuda")
        weight = torch.randn(n, k, device="cuda") / (k ** 0.5)
        reference = torch.nn.functional.linear(x, weight)
        bf16_x = x.bfloat16()
        bf16_weight = weight.bfloat16()
        packed, weight_scale = pack_weight(weight)
        weight_col = packed[:, :n].t().contiguous()
        q = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
        x_scale = torch.empty(m, device="cuda")
        y_triton = torch.empty((m, n), device="cuda")
        y_scaled = torch.empty_like(y_triton)
        ones = torch.ones(1, device="cuda")
        no_bias = weight_scale

        def quantize():
            _quantize[(m,)](x, q, x_scale, k, triton.next_power_of_2(k))

        quantize()

        def bf16_cublas():
            return torch.nn.functional.linear(bf16_x, bf16_weight).float()

        def scaled_mm_prequant():
            raw = torch._scaled_mm(
                q, weight_col.t(), scale_a=ones, scale_b=ones,
                out_dtype=torch.float32, use_fast_accum=False)
            _epilogue[(triton.cdiv(m * n, 256),)](
                raw, x_scale, weight_scale, no_bias, y_scaled,
                m, n, False, 256)
            return y_scaled

        def triton_prequant():
            _gemm[(triton.cdiv(m, tile.bm), triton.cdiv(n, tile.bn), 1)](
                q, packed, x_scale, weight_scale, no_bias, y_triton,
                m, n, k, packed.shape[1], tile.bm, tile.bn, tile.bk,
                False, 1, num_warps=tile.warps, num_stages=tile.stages)
            return y_triton

        def scaled_mm_dynamic():
            quantize()
            return scaled_mm_prequant()

        def triton_dynamic():
            return linear(x, packed, weight_scale, None, tile)

        backends = {
            "bf16_cublas": bf16_cublas,
            "scaled_mm_prequant": scaled_mm_prequant,
            "scaled_mm_dynamic_quant": scaled_mm_dynamic,
            "triton_w8a8_prequant": triton_prequant,
            # The current repository wrapper is two launches. It is the baseline
            # for a later genuine quantize+GEMM fusion, not mislabeled as fused.
            "triton_quant_plus_gemm_unfused": triton_dynamic,
        }
        measurements = {}
        for name, fn in backends.items():
            output = fn()
            torch.cuda.synchronize()
            diff = output.float() - reference
            measurements[name] = {
                "latency_us": measure(fn),
                "max_abs": diff.abs().max().item(),
                "rmse": diff.square().mean().sqrt().item(),
            }
        measurements["quantize_only"] = {"latency_us": measure(quantize)}
        rows.append({"label": label, "m": m, "n": n, "k": k, "backends": measurements})

    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "sm": 89,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "online_tuning": False,
        "fixed_functional_tile": tile.__dict__,
        "warning": "The fixed Triton tile is a functional comparison, not an optimality claim.",
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
