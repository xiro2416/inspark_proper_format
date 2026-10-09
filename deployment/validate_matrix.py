"""Serial INT8 route measurement, frozen-input audits and lifecycle acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path(os.environ.get('INDEX_HISTORY_DIR', ROOT / 'deployment/history'))
CONFIG = 'local_assets/runtime/runtime_fp32_b1.yaml'
MANIFEST = 'deployment/history/validation-manifest.json'


def run(label, *args):
    print(json.dumps({'stage': label}), flush=True)
    with (ROOT / '.cache' / (label + '.log')).open('w') as log:
        subprocess.run([sys.executable, *map(str, args)], check=True, stdout=log,
                       stderr=subprocess.STDOUT)


def benchmark(batch, path, label, final=False):
    output = HISTORY / (label + '.json')
    run(label, 'benchmarks/benchmark_unified_first_chunk.py', 'run', '--gpu', 1,
        '--batch', batch, '--manifest', MANIFEST, '--deployment', path,
        '--config', CONFIG, '--out', output, '--warmups', 5 if final else 3,
        '--waves', 30 if final else 10, '--power-seconds', 30 if final else 0,
        '--label', label, '--quant-recipe', 'int8_smoothquant_alpha1.0')
    return json.loads(output.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', type=int, nargs='+', default=[1,2,4,8,16])
    a = p.parse_args()
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Only physical GPU1 is authorized')
    directory = ROOT / 'configs/hardware/sm89/indextts'
    record = HISTORY / 'validated-matrix.json'
    summary = json.loads(record.read_text()) if record.is_file() else []
    for batch in a.batches:
        if batch not in (1,2,4,8,16,32,64,128):
            raise ValueError('Unsupported batch')
        selected_path = directory / f'int8_b{batch}_selected.json'
        if selected_path.is_file() and any(row['batch'] == batch for row in summary):
            prior=json.loads(selected_path.read_text())
            if prior['status'] == 'validated_local_sm89_int8' and '/vocoder-gemm/' in prior['vocoder_plan']:
                print(json.dumps(dict(stage='previously_validated',batch=batch)),flush=True)
                continue
        run(f'materialize-b{batch}', 'deployment/materialize_plans.py',
            '--batches', batch, '--cfm-directory', 'cfm-estimator')
        results = {}
        for route in ('framework_graph', 'native_graph'):
            path = directory / f'int8_b{batch}_{route}.json'
            results[route] = benchmark(batch, path, f'probe-b{batch}-{route}')
        from benchmarks.unified_first_chunk import compare_reports
        comparison = compare_reports(results['framework_graph'], results['native_graph'])
        (HISTORY / f'compare-workers-b{batch}.json').write_text(json.dumps(comparison,indent=2)+'\n')
        key = 'wave_admission_to_last_pcm_ms'
        route = min(results, key=lambda r: results[r]['summary'][key]['p50'])
        candidate = directory / f'int8_b{batch}_{route}.json'
        # Grouped conditions and prefix reuse are hardware-independent candidates.
        # Measure their combined effect under the same workload before retaining.
        if batch > 1:
            plain = json.loads(candidate.read_text())
            plain.update(batch_conditions=False, latent_cached_prefix=False)
            plain_path = directory / f'int8_b{batch}_{route}_plain.json'
            plain_path.write_text(json.dumps(plain, indent=2)+'\n')
            plain_result = benchmark(batch, plain_path, f'probe-b{batch}-plain')
            comparison = compare_reports(plain_result, results[route])
            (HISTORY / f'compare-scheduling-b{batch}.json').write_text(json.dumps(comparison,indent=2)+'\n')
            if plain_result['summary'][key]['p50'] < results[route]['summary'][key]['p50']:
                candidate = plain_path
        lifecycle = HISTORY / f'lifecycle-selected-b{batch}.json'
        run(f'lifecycle-selected-b{batch}', 'deployment/validate_lifecycle.py',
            '--batch', batch, '--deployment', candidate, '--output', lifecycle)
        if not json.loads(lifecycle.read_text())['passed']:
            raise RuntimeError('Lifecycle failed')
        # Report numerical differences without an invented tolerance gate.
        for component in ('ar', 'acoustics'):
            script = f'scripts/audit_unified_{component}.py'
            capture = HISTORY / f'capture-{component}-b{batch}.pt'
            audit = HISTORY / f'audit-{component}-b{batch}.json'
            coverage = ['--waves', max(1, min(4, 32//batch))] if component == 'acoustics' else []
            run(f'capture-{component}-b{batch}', script, 'capture', '--gpu', 1,
                '--config', CONFIG, '--manifest', MANIFEST,
                '--deployment', candidate, '--output', capture, *coverage)
            run(f'audit-{component}-b{batch}', script, 'audit', '--gpu', 1,
                '--config', CONFIG, '--capture', capture, '--output', audit)
        final = benchmark(batch, candidate, f'final-int8-b{batch}', final=True)
        counters = final['measured_counters']
        if not final['execution_pass'] or any(counters[key] for key in
                ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):
            raise RuntimeError('Unexpected first-chunk fallback')
        if any(counters['head_graph_hits'][name] != 30 for name in ('cfm','vocoder')):
            raise RuntimeError('Missing first-chunk acoustic graph coverage')
        selected = json.loads(candidate.read_text())
        selected['status'] = 'validated_local_sm89_int8'
        destination = directory / f'int8_b{batch}_selected.json'
        destination.write_text(json.dumps(selected, indent=2)+'\n')
        summary = [row for row in summary if row['batch'] != batch]
        summary.append(dict(batch=batch, selected=str(destination),
                            route=selected['runtime_backend'],
                            batch_conditions=selected['batch_conditions'],
                            latent_cached_prefix=selected.get('latent_cached_prefix',False),
                            p50_ms=final['summary'][key]['p50'],
                            p95_ms=final['summary'][key]['p95'],
                            lifecycle=str(lifecycle), benchmark=f'final-int8-b{batch}.json'))
        (HISTORY / 'validated-matrix.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary[-1]), flush=True)


if __name__ == '__main__':
    main()
