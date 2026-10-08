"""Recheck source GEMM tile sizes at the target batch and internal frame lengths."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl
from inspark_infer.ops.tensorrt.zipvoice.a1007.b1.linear_f32_activation_tf32_rna_kernel import linear_f32_activation
from inspark_infer.ops.tensorrt.zipvoice.a1007.b1.i8_residual_runtime_kernel import i8_residual

ROOT = Path(__file__).resolve().parents[1]


@triton.jit
def residual_tile(Q, W, AS, WS, Bias, R, Y, M, BM: tl.constexpr):
    i8_residual(Q, W, AS, WS, Bias, R, Y, M, 48, 512, BM, 64, 64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, choices=(1, 2, 4, 8, 16, 32, 64), required=True)
    p.add_argument('--skip-nonlinear', action='store_true', help='Reuse the separately recorded nonlinear geometry probe')
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    lock = Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(1007)
    torch.set_num_threads(2)
    value_kernel = importlib.import_module(
        f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{a.batch}.int8_nonlinear_value_runtime_kernel'
    ).int8_nonlinear_value
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    records = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for family, lengths in (('f32_ffn', (380, 760)), ('i8_residual', (190, 380, 760)),
                                ('i8_value', (190, 380, 760))):
            for frames in lengths:
                m = a.batch * frames
                bn = 64
                if family == 'f32_ffn':
                    n = 1536
                    x = torch.randn(m, 512, device='cuda') * .1
                    w = torch.randn(512, n, device='cuda') * .1
                    bias = torch.randn(n, device='cuda') * .1
                    out = torch.empty(m, n, device='cuda')
                    base = 128
                    configs = (128, 64, 32)

                    def launch(bm):
                        return linear_f32_activation[(triton.cdiv(m, bm) * triton.cdiv(n, 64),)](
                            x, w, bias, out, m, 512, n, bm, 64, 32,
                            4., .07999999821186066, .03500000014901161, 'tf32',
                            num_warps=4, num_stages=2, enable_fp_fusion=False)
                elif family == 'i8_residual':
                    n = 512
                    x = torch.randint(-127, 128, (m, 48), device='cuda', dtype=torch.int8)
                    w = torch.randint(-127, 128, (n, 48), device='cuda', dtype=torch.int8)
                    activation_scale = torch.tensor([.01], device='cuda')
                    weight_scale = torch.rand(n, device='cuda') * .01
                    bias = torch.randn(n, device='cuda') * .1
                    residual = torch.randn(m, n, device='cuda') * .1
                    out = torch.empty_like(residual)
                    # All source rewrites explicitly select the 128x64 factory.
                    base = 128
                    configs = (128, 64, 32, 16)

                    def launch(bm):
                        return residual_tile[(triton.cdiv(m, bm), 8)](
                            x, w, activation_scale, weight_scale, bias, residual, out, m, bm,
                            num_warps=4, num_stages=2, enable_fp_fusion=False)
                else:
                    n, bn = 384, 32
                    x = torch.randint(-127, 128, (m, 512), device='cuda', dtype=torch.int8)
                    w = torch.randint(-127, 128, (3*n, 512), device='cuda', dtype=torch.int8)
                    activation_scale = torch.tensor([.01], device='cuda')
                    weight_scale = torch.rand(3*n, device='cuda') * .01
                    bias = torch.randn(3*n, device='cuda') * .1
                    out = torch.empty(2*m*n, device='cuda')
                    value = out[:m*n].view(a.batch, frames, n)
                    gate = out[m*n:].view(m, n)
                    base, configs = 128, (128, 64, 32, 16)

                    def launch(bm):
                        return value_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                            x, w, activation_scale, weight_scale, bias, value, gate,
                            frames, bm, bn, 64, num_warps=4, num_stages=2,
                            enable_fp_fusion=False)
                launch(base)
                stream.synchronize()
                reference = out.clone()
                baseline = torch.cuda.CUDAGraph()
                with torch.cuda.graph(baseline, stream=stream):
                    for _ in range(20):
                        launch(base)
                for bm in configs:
                    row = {'family': family, 'batch': a.batch, 'frames_at_stage': frames,
                           'M': m, 'N': n, 'source_BM': base, 'candidate_BM': bm,
                           'candidate_BN': bn,
                           'candidate_ctas': triton.cdiv(m, bm) * triton.cdiv(n, bn)}
                    try:
                        out.fill_(float('nan'))
                        kernel = launch(bm)
                        stream.synchronize()
                        assert torch.isfinite(out).all()
                        if family in ('i8_residual', 'i8_value'):
                            assert torch.equal(out, reference)
                        row['correctness'] = {'finite_all': True, 'exact': torch.equal(out, reference),
                                              'relative_l2': float((out-reference).norm()/reference.norm().clamp_min(1e-20))}
                        row['resources'] = {'registers': kernel.n_regs, 'spills': kernel.n_spills,
                                            'shared': kernel.metadata.shared}
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(20):
                                launch(bm)
                        times = [[], []]
                        for repeat in range(6):
                            for index in ((0, 1, 1, 0) if repeat % 2 == 0 else (1, 0, 0, 1)):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(5):
                                    (baseline if index == 0 else graph).replay()
                                end.record()
                                end.synchronize()
                                times[index].append(start.elapsed_time(end) / 100)
                        medians = [statistics.median(v) for v in times]
                        row.update(status='complete', source_candidate_ms=medians,
                                   gain_percent=100*(medians[0]-medians[1])/medians[0], samples_ms=times)
                        del graph
                    except Exception as exc:
                        row.update(status='failed', error=str(exc))
                    records.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
                del baseline
    result = {'status': 'geometry_probe_complete_review_pending', 'batch': a.batch,
              'scope': 'Existing source kernels with target M; 20 same-work calls per graph to bound host launch gaps, event duration divided by100 calls; builder Triton JIT fragment; alignment specialization may differ from deployed AOT, requiring AOT rebuild and E2E validation',
              'rows': records}
    path = ROOT / f'reports/sm89/zipvoice/a1007/b{a.batch}/history/009-linear-geometry.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    if not a.skip_nonlinear:
        from probe_zipvoice_nonlinear_geometry import probe
        probe(a.batch, stream)


if __name__ == '__main__':
    main()
