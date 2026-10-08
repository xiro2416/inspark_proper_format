"""Migration recheck of source B32 compact H2D and first-PCM/D2H overlap."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

from run_zipvoice_validation import ROOT, invoke, sha
from zipvoice_selected_route import selected_route, idle_gpu_preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--routes-json', type=Path, help='Explicit batch-to-route mapping after formal compute review')
    args = p.parse_args()
    lease = Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = idle_gpu_preflight()
    manifest = json.loads((ROOT / 'outputs/zipvoice-validation/cases/manifest.json').read_text())
    cases = {'short': min(manifest['quality_cases'], key=lambda c: c['total_frames']),
             'primary760': manifest['primary_760'],
             'long': max(manifest['quality_cases'], key=lambda c: c['total_frames'])}
    for batch in args.batches:
        history = ROOT / f'reports/sm89/zipvoice/a1007/b{batch}/history'
        route, compute_evidence = selected_route(batch, args.routes_json, manifest['primary_760'])
        evidence_name = '023-selected-delivery.json' if args.routes_json else '010-delivery-reuse.json'
        summary = {'batch': batch, 'status': 'running', 'shapes': {},
                   'source_mechanism': 'B32 retained compact caller-row H2D, GPU broadcast/pad, first PCM overlapped with remaining D2H; same stream/event dependencies',
                   'scope': 'Migration of existing mechanism; not a new optimization gain'}
        summary['compute_evidence'] = compute_evidence
        summary['preflight'] = preflight
        for label, case in cases.items():
            samples = {'a1007': [], 'a1007_delivery': []}
            hashes = None
            blocks = []
            for index, application in enumerate(('a1007', 'a1007_delivery', 'a1007_delivery', 'a1007')):
                dest = ROOT / f'outputs/zipvoice-benchmarks/b{batch}/delivery-reuse/{route}/{label}/{index}-{application}'
                report, rows = invoke(batch, route, case, dest, repetitions=10, warmup=3,
                                      functional=False, application=application)
                current = {row: sha(dest / f'{row:04d}.wav') for row in rows}
                if hashes is None:
                    hashes = current
                assert hashes == current, (batch, label, 'delivery changed selected PCM')
                values = [x['all_pcm_s'] for x in report['results']]
                samples[application].extend(values)
                blocks.append({'application': application, 'median_s': statistics.median(values),
                               'report': str(dest / 'report.json')})
            medians = {k: statistics.median(v) for k, v in samples.items()}
            summary['shapes'][label] = {'frames': case['total_frames'], 'medians_s': medians,
                                       'gain_percent': 100*(medians['a1007']-medians['a1007_delivery'])/medians['a1007'],
                                       'samples_per_route': 20, 'blocks': blocks, 'selected_wav_exact': True}
            (history / evidence_name).write_text(json.dumps(summary, indent=2) + '\n')
        # Complete all-row transport and PCM oracle, independent of timed runs.
        functional = ROOT / f'outputs/zipvoice-benchmarks/b{batch}/delivery-reuse/{route}/functional760'
        proof, _ = invoke(batch, route, cases['primary760'], functional, application='a1007_delivery')
        assert proof['compact_input_all_batch_tokens_speech_rms_bitwise_exact']
        assert proof['overlap_all_batch_transfers_and_ordered_pcm_exact']
        summary.update(status='delivery_reuse_measured_review_pending', all_rows_delivery_exact=True,
                       functional_report=str(functional / 'report.json'))
        (history / evidence_name).write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'batch': batch, 'shapes': summary['shapes'], 'status': summary['status']}), flush=True)


if __name__ == '__main__':
    main()
