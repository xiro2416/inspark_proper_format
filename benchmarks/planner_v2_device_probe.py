"""Single-GPU primitive-rate probe; missing shared/dependency terms stay explicit."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def median_cuda(torch, fn, *, inner=50, repeats=7):
    for _ in range(5): fn()
    values = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        for _ in range(inner): fn()
        end.record(); end.synchronize()
        values.append(begin.elapsed_time(end) * 1e-3 / inner)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu', type=int, default=5); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from inspark_infer.runtime.device import GPULease, select_gpu
    select_gpu(args.gpu)
    import torch
    from inspark_infer.ops.planning.v2.model import HardwareProfile
    with GPULease(args.gpu), torch.inference_mode():
        profile = HardwareProfile.current(); device = torch.device('cuda')
        scalar = torch.ones(1, device=device)
        launch_s = median_cuda(torch, lambda: scalar.add_(1), inner=1000)
        large = 128 * 1024 * 1024 // 2
        source = torch.ones(large, device=device, dtype=torch.bfloat16); target = torch.empty_like(source)
        dram_s = median_cuda(torch, lambda: target.copy_(source), inner=20)
        dram_bytes_s = source.numel() * source.element_size() * 2 / dram_s
        small = torch.ones(2 * 1024 * 1024 // 2, device=device, dtype=torch.bfloat16); small_out = torch.empty_like(small)
        l2_s = median_cuda(torch, lambda: small_out.copy_(small), inner=200)
        l2_bytes_s = small.numel() * small.element_size() * 2 / l2_s
        size = 4096
        if profile.native_fp8:
            a = torch.randn(size, size, device=device).to(torch.float8_e4m3fn)
            b = torch.randn(size, size, device=device).to(torch.float8_e4m3fn).t()
            one = torch.ones(1, device=device)
            tensor_fn = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=False)
            dtype = 'fp8'
        else:
            a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
            b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
            tensor_fn = lambda: a @ b
            dtype = 'bf16'
        tensor_s = median_cuda(torch, tensor_fn, inner=10)
        tensor_flops_s = 2 * size ** 3 / tensor_s
        result = dict(
            hardware=profile.as_dict(), preferred_dtype=dtype,
            measured=dict(launch_s=launch_s, dram_bytes_s=dram_bytes_s,
                          l2_bytes_s=l2_bytes_s, tensor_flops_s=tensor_flops_s),
            missing=['shared_bytes_s', 'global_dependency_latency_s'],
            rates_complete=False,
            note='No performance manifest may be validated until NCU/shared/dependency calibration fills missing terms.',
        )
    out = ROOT / 'reports' / args.output; out.mkdir(parents=True, exist_ok=False)
    (out / 'device_rates.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
