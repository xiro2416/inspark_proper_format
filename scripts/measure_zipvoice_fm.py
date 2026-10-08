"""Target FM-only timing and optional diagnostic layer profile, separate from E2E."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    lease = Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    import tensorrt as trt
    from inspark_infer.ops.tensorrt.zipvoice.engine import Engine
    batch = args.batch
    for name in ('dw_int8_plugin', 'normal_tf32_plugin' if batch != 64 else 'online_branch_wide_plugin',
                 'online_nonlinear_rna_plugin', 'f32_tf32_rna_plugin',
                 'int8_nonlinear_value_plugin', 'i8_residual_plugin'):
        importlib.import_module(f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}.{name}')
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (8, 9)
    stream = torch.cuda.Stream()

    class Profiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.rows = {}

        def report_layer_time(self, name, ms):
            self.rows.setdefault(name, []).append(ms)

    result = {'batch': batch, 'scope': 'single FM on fresh seeded full-batch inputs; excludes application, H2D and PCM',
              'status': 'running', 'timing': 'CUDA event around repeated captured FM; profile separately instrumented', 'cases': []}
    for frames in (600, 760, 920):
        rng = torch.Generator(device='cuda').manual_seed(9100)
        inputs = {k: torch.randn(batch, frames, 100, device='cuda', generator=rng)
                  for k in ('x', 'text_condition', 'speech_condition')}
        inputs.update(t=torch.full((batch, 1, 1), .5, device='cuda'),
                      guidance_scale=torch.ones(batch, 1, 1, device='cuda'),
                      padding_mask=torch.zeros(batch, frames, dtype=torch.bool, device='cuda'))
        inputs['padding_mask'][:, -3:] = True
        stream.wait_stream(torch.cuda.current_stream())
        for route in ('native', 'inherited'):
            folder = ROOT / f'artifacts/zipvoice/a1007/b{batch}/fm-{route}'
            metadata = json.loads((folder / 'build.json').read_text())
            engine = Engine(folder / 'engine.plan', trt, torch, True, {k: tuple(v.shape) for k, v in inputs.items()})
            arena = torch.empty(engine.output_arena_size(), device='cuda', dtype=torch.uint8)
            engine.context.set_device_memory(arena.data_ptr(), arena.numel())
            engine.bind_output_arena(arena)
            with torch.inference_mode(), torch.cuda.stream(stream):
                for _ in range(3):
                    engine(inputs, stream)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    engine(inputs, stream)
                samples = []
                for _ in range(20):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 5)
                assert torch.isfinite(engine.outputs['velocity']).all()
                row = {'frames': frames, 'route': route, 'engine_sha256': metadata['engine_sha256'],
                       'median_ms': statistics.median(samples), 'samples_ms': samples}
                del graph
                if args.profile:
                    profiler = Profiler()
                    engine.context.profiler = profiler
                    for _ in range(3):
                        engine(inputs, stream)
                        stream.synchronize()
                    assert profiler.rows and all(len(v) == 3 for v in profiler.rows.values())
                    row['instrumented_layers'] = sorted(
                        [{'name': k, 'mean_ms': statistics.mean(v)} for k, v in profiler.rows.items()],
                        key=lambda x: x['mean_ms'], reverse=True)
                    row['instrumented_sum_ms'] = sum(x['mean_ms'] for x in row['instrumented_layers'])
                result['cases'].append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ('samples_ms', 'instrumented_layers')}), flush=True)
            del engine, arena
    result['status'] = 'framework_timing_complete_e2e_separate'
    path = ROOT / f'reports/sm89/zipvoice/a1007/b{batch}/history/008-framework-timing.json'
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
