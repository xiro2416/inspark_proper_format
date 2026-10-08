"""Focused migration recheck of inherited CPU PCM chunk scheduling at new batches."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

from run_zipvoice_validation import ROOT, invoke, sha
from zipvoice_selected_route import selected_route, idle_gpu_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--routes-json', type=Path, help='Explicit batch-to-route mapping after formal compute review')
    args = parser.parse_args()
    lease = Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = idle_gpu_preflight()
    manifest = json.loads((ROOT / 'outputs/zipvoice-validation/cases/manifest.json').read_text())
    for batch in args.batches:
        history = ROOT / f'reports/sm89/zipvoice/a1007/b{batch}/history'
        route, compute_evidence = selected_route(batch, args.routes_json, manifest['primary_760'])
        evidence_name = '022-selected-scheduling.json' if args.routes_json else '006-scheduling.json'
        source = 6 if batch == 32 else 16
        chunks = [source] + [x for x in (1, 4) if x != source and x < batch - 1]
        result = {'batch': batch, 'source_policy': {'chunk': source, 'workers': 4},
                  'status': 'running', 'shapes': {}, 'numerical_contract': 'Selected WAV bytes exact across scheduling choices'}
        result['compute_evidence'] = compute_evidence
        result['preflight'] = preflight
        if batch <= 2:
            result.update(status='not_applicable', reason='After original first PCM there are zero or one remaining rows; all positive chunk sizes execute identical work partition.')
        else:
            cases = {'primary760': manifest['primary_760'],
                     'short': min(manifest['quality_cases'], key=lambda x: x['total_frames']),
                     'long': max(manifest['quality_cases'], key=lambda x: x['total_frames'])}
            for label, case in cases.items():
                values = {c: [] for c in chunks}
                hashes = None
                blocks = []
                # Balanced forward/reverse blocks; selection stays review-pending.
                for index, chunk in enumerate(chunks + list(reversed(chunks))):
                    destination = ROOT / f'outputs/zipvoice-benchmarks/b{batch}/scheduling/{route}/{label}/{index}-chunk{chunk}'
                    report, rows = invoke(batch, route, case, destination,
                                          extra=['--pcm-chunk', str(chunk), '--pcm-workers', '4'],
                                          warmup=3, repetitions=10, functional=False)
                    current = {row: sha(destination / f'{row:04d}.wav') for row in rows}
                    if hashes is None:
                        hashes = current
                    assert hashes == current, (batch, label, chunk, 'PCM scheduling changed output')
                    samples = [x['all_pcm_s'] for x in report['results']]
                    values[chunk].extend(samples)
                    blocks.append({'chunk': chunk, 'median_s': statistics.median(samples),
                                   'report': str(destination / 'report.json')})
                stats = {str(c): {'count': len(v), 'median_s': statistics.median(v)} for c, v in values.items()}
                result['shapes'][label] = {'frames': case['total_frames'], 'stats': stats,
                                           'blocks': blocks, 'selected_wav_exact': True}
                (history / evidence_name).write_text(json.dumps(result, indent=2) + '\n')
            result['status'] = 'matched_scheduling_measurements_complete_review_pending'
        (history / evidence_name).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'batch': batch, 'status': result['status'], 'shapes': result['shapes']}), flush=True)


if __name__ == '__main__':
    main()
