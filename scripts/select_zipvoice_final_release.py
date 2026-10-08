"""Select completed, reviewed candidates; refuse partial transfer evidence."""
import json
from pathlib import Path
from run_zipvoice_validation import ROOT, sha

REPORTS = ROOT / 'reports/sm89/zipvoice/a1007'


def load(name):
    return json.loads((REPORTS / name).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def main():
    baseline = load('migration-baseline.json')
    graphs = load('model-graph-review.json')
    transfers = load('prepack-transfer-execution.json')
    assert transfers['status'] == 'three_prepack_transfers_complete_review_pending'
    assert set(graphs['batches']) == {'1', '2', '4', '8', '16', '32', '64'}
    selection = {'status': 'all_batches_migrated_optimized_validated', 'batches': {}}
    reviews = {}
    common = ['migration-baseline.json', 'migration-final-review.json',
              'optimization-entry-cost-review.json', 'scheduling-review.json',
              'delivery-review.json', 'prepack-transfer-screen-review.json']
    for batch in (1, 2, 4, 8, 16, 32, 64):
        key = str(batch)
        old, graph = baseline['batches'][key], graphs['batches'][key]
        assert old['migration_accepted'] and graph['decision'] == 'retain_graph'
        history = f'b{batch}/history/'
        evidence = {**old['evidence'], **graph['evidence']}
        for name in common:
            evidence[relative(REPORTS / name)] = sha(REPORTS / name)
        if batch in (8, 16, 32, 64):
            decision_name = history + ('040-f32-prepack-retention.json' if batch == 64 else '042-f32-prepack-retention.json')
            decision = load(decision_name)
            assert decision['status'] == 'f32_operand_prepack_retained_remaining_transfer_review_pending'
            route, application, policy = decision['route'], decision['application'], decision['pcm_policy']
            app = load(history + f'018-{route}-application.json')
            validation = [relative(x['report']) for x in app['contract_cases']]
            inventory = Path(app['cases'][0]['report']).parent / 'engines.json'
            names = [decision_name, history + f'018-{route}-application.json',
                     history + f'019-{route}-quality.json', history + f'020-{route}-performance.json',
                     history + ('038-actual-weight-prepack.json' if batch == 64 else '041-actual-weight-prepack.json')]
            if batch == 64:
                names += [history + '033-normal-k16-engine-review.json',
                          history + '035-normal-k16-quality-review.json',
                          history + '036-normal-k16-retention.json', history + '032-operator-screen-review.json']
            summary = {k: decision[k] for k in ('primary_median_ms', 'power_w', 'gains_percent')}
            if 'shape_tradeoff_review' in decision:summary['shape_tradeoff_review'] = decision['shape_tradeoff_review']
        else:
            route, application, policy = graph['route'], graph['application'], graph['pcm_policy']
            app = load(history + '027-model-graph-validation.json')
            validation = [relative(x['report']) for x in app['functional'] if x['kind'] != 'accepted_quality_case']
            inventory = Path(app['functional'][0]['report']).parent / 'engines.json'
            names = [history + '027-model-graph-validation.json']
            summary = {k: graph[k] for k in ('primary_median_ms', 'power_w', 'gains_percent')}
        for name in names:
            evidence[relative(REPORTS / name)] = sha(REPORTS / name)
        evidence[relative(inventory)] = sha(inventory)
        for path, digest in evidence.items():
            assert sha(ROOT / path) == digest, path
        stop = {
            1: 'Native route retained; graph covers launch overhead. Static custom FFN prepack inapplicable. Migration delivery/PCM alternatives reviewed.',
            2: 'geo3 native attention/FFN/value retained; custom DW/residual preserved. Custom FFN prepack inapplicable. Graph and scheduling measured.',
            4: 'geo1 retained; static operand prepack screen regressed aggregate full-stage cost, so retain original operand path. Graph and scheduling measured.',
            8: 'Inherited geometry retained after migration alternative regression; simple identical-math prepack and graph retained after full E2E validation.',
            16: 'Inherited geometry retained after migration alternative regression; simple identical-math prepack and graph retained after full E2E validation.',
            32: 'Original compact conditioning and overlapping delivery restored before optimization; inherited geometry retained; graph and prepack validated.',
            64: 'Normal attention TF32 Q64/K16 retained with disclosed quality differences. FFN geometry/pipeline/warp sweeps rejected; identical-math weight prepack retained. Protected nonlinear fusion source evidence rejects spill-heavy candidates.',
        }[batch]
        reviews[key] = {'route': route, 'application': application, 'stopping_reason': stop,
                        'optimization_review_complete': True, **summary}
        selection['batches'][key] = {'route': route, 'runner': f'src/inspark_infer/runtime/zipvoice/routes/{application}.py',
                                    'pcm_policy': policy, 'inventory': relative(inventory), 'evidence': evidence,
                                    'fresh_validation_reports': validation, 'migration_accepted': True,
                                    'optimization_review_complete': True}
    review = {'status': 'all_seven_execution_cost_reviews_complete', 'batches': reviews,
              'limits': 'Bounded evidence-based stopping, not a global optimum. Gains use matched controls named in each report; do not add gains across separate runs. Board power includes the preserved idle external allocation. Raw sampled overshoot retained.'}
    path = REPORTS / 'final-execution-cost-review.json'
    path.write_text(json.dumps(review, indent=2) + '\n')
    for chosen in selection['batches'].values():
        chosen['evidence'][relative(path)] = sha(path)
    (REPORTS / 'final-accepted-selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({'status': selection['status'], 'batches': list(reviews)}))


if __name__ == '__main__':
    main()
