"""Existing nonlinear attention geometry under the actual small target batch.

Called inside the linear-geometry job's GPU lease; no separate GPU process.
"""
import importlib
import json
import os
from pathlib import Path
import statistics

import torch
import triton


def probe(batch, stream):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    kernel = importlib.import_module(
        f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}.online_nonlinear_rna_runtime_kernel'
    ).online_nonlinear_rna
    root = Path(__file__).resolve().parents[1]
    records = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        for length in (150, 190, 230, 380, 760):
            q = torch.randn(1, batch, length, 32, device='cuda') / 32**.5
            k = torch.randn(1, batch, 32, length, device='cuda')
            pq = torch.randn(1, batch, length, 4, device='cuda')
            e = torch.randn(1, 1, 4, 2*length-1, device='cuda')
            mask = torch.zeros(batch, length, device='cuda', dtype=torch.bool)
            mask[:, -3:] = True
            if batch > 1:
                mask[0, :] = True
            value = torch.randn(1, batch, length, 384, device='cuda')
            output = torch.empty_like(value)

            def launch(qb, kb, warps):
                return kernel[(triton.cdiv(length, qb), batch, 1)](
                    q, k, pq, e, mask, value, output, length, True, qb, kb,
                    'tf32', True, num_warps=warps, num_stages=1,
                    enable_fp_fusion=False)

            launch(32, 16, 4)
            stream.synchronize()
            reference = output.clone()
            baseline = torch.cuda.CUDAGraph()
            with torch.cuda.graph(baseline, stream=stream):
                for _ in range(20):
                    launch(32, 16, 4)
            for qb, kb, warps in ((32, 16, 4), (16, 16, 4), (32, 32, 4),
                                  (16, 32, 4), (32, 16, 8)):
                row = {'batch': batch, 'T': length, 'QB': qb, 'KB': kb,
                       'warps': warps, 'ctas': triton.cdiv(length, qb)*batch,
                       'source_geometry': [32, 16, 4]}
                try:
                    output.fill_(float('nan'))
                    compiled = launch(qb, kb, warps)
                    stream.synchronize()
                    assert torch.isfinite(output).all()
                    row['correctness'] = {
                        'all_batch_original_head0_full384_written': True,
                        'exact': torch.equal(output, reference),
                        'relative_l2': float((output-reference).norm()/reference.norm().clamp_min(1e-20))}
                    row['resources'] = {
                        'regs': compiled.n_regs, 'spills': compiled.n_spills,
                        'shared': compiled.metadata.shared,
                        'rna_conversion': 'cvt.rna.tf32.f32' in compiled.asm['ptx'],
                        'dense_tf32_mma': 'mma.sync' in compiled.asm['ptx'] and '.tf32.' in compiled.asm['ptx']}
                    row['aot_resource_eligible'] = (
                        compiled.metadata.shared <= 49152 and
                        not compiled.metadata.global_scratch_size and
                        not compiled.metadata.profile_scratch_size)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(20):
                            launch(qb, kb, warps)
                    samples = [[], []]
                    for repeat in range(6):
                        for index in ((0, 1, 1, 0) if repeat % 2 == 0 else (1, 0, 0, 1)):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(5):
                                (baseline if index == 0 else graph).replay()
                            end.record()
                            end.synchronize()
                            samples[index].append(start.elapsed_time(end)/100)
                    medians = [statistics.median(v) for v in samples]
                    row.update(status='complete', source_candidate_ms=medians,
                               gain_percent=100*(medians[0]-medians[1])/medians[0],
                               samples_ms=samples)
                    del graph
                except Exception as exc:
                    row.update(status='failed', error=str(exc))
                records.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
            del baseline, reference, q, k, pq, e, mask, value, output
    report = {'status': 'nonlinear_source_geometry_probe_complete_review_pending',
              'batch': batch, 'rows': records,
              'scope': 'Original head0 and full384 AV, IEEE QK/RNA TF32 AV unchanged. Small batch changes CTA coverage and live accumulator state. Builder JIT diagnostic only; deployed AOT rebuild and matched E2E required.'}
    (root/f'reports/sm89/zipvoice/a1007/b{batch}/history/015-nonlinear-geometry.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    import argparse
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', required=True,
                        choices=(1, 2, 4, 8, 16, 32, 64))
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    lock = Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.manual_seed(9102)
    torch.set_num_threads(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for batch in args.batches:
        probe(batch, stream)
