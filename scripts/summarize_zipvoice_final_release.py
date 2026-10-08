"""Render final selected latency/power evidence without adding unrelated gains."""
import json
from run_zipvoice_validation import ROOT, sha

REPORTS = ROOT / 'reports/sm89/zipvoice/a1007'


def read(path):
    return json.loads(path.read_text())


def main():
    selection_path = REPORTS / 'final-accepted-selection.json'
    selection = read(selection_path)
    assert selection['status'] == 'all_batches_migrated_optimized_validated'
    baseline = read(REPORTS / 'migration-baseline.json')
    rows = []
    for key, chosen in selection['batches'].items():
        assert chosen['migration_accepted'] and chosen['optimization_review_complete']
        history = REPORTS / f'b{key}/history'
        route = chosen['route']
        prepack = route in ('wp', 'normtf32k16wp')
        evidence_path = history / (f'020-{route}-performance.json' if prepack else '027-model-graph-validation.json')
        assert chosen['evidence'][str(evidence_path.relative_to(ROOT))] == sha(evidence_path)
        data = read(evidence_path)
        candidate = route if prepack else 'candidate'
        control = ('normtf32k16' if key == '64' else 'inherited') if prepack else 'control'
        shapes = {name: {'frames': shape['frames'],
                         'control_ms': shape['statistics'][control]['median_s'] * 1000,
                         'selected_ms': shape['statistics'][candidate]['median_s'] * 1000,
                         'matched_gain_percent': shape['gain_percent']}
                  for name, shape in data['shapes'].items()}
        quality = read(history / f'019-{route}-quality.json')['by_route'][route] if prepack else baseline['batches'][key]['quality']
        rows.append({'batch': int(key), 'route': route, 'application': chosen['runner'],
                     'migration_median_ms': baseline['batches'][key]['primary_median_s'] * 1000,
                     'shapes': shapes, 'board_power_w': data['sustained_power']['telemetry']['board_power_w'],
                     'quality': quality, 'evidence': str(evidence_path.relative_to(ROOT))})
    result = {'status': 'accepted_local_results_publication_pending',
              'selection_sha256': sha(selection_path), 'batches': rows,
              'reference_duration_s': 4.0,
              'raw_generated_duration_s': {'600': 2.3893333333, '760': 4.096, '920': 5.8026666667},
              'limits': 'Migration and final medians come from different measurement rounds; their difference is descriptive, not a single matched gain. Each last-step gain uses the explicitly named control in its evidence. Whole-board power includes preserved external idle allocation; configured/enforced cap400W, raw overshoot retained. Finite corpus quality is not perceptual equivalence.'}
    (REPORTS / 'final-results.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# A_1007 本地最终选择', '',
             '参考音频4秒；760总帧的原始生成波形4.096秒，最终WAV时长由原有静音处理和尾部停顿决定。', '',
             '| Batch | 迁移记录(ms) | 当前760(ms) | 末次同轮收益 | 功耗均值/P95/采样最大(W) |',
             '|---:|---:|---:|---:|---:|']
    for row in rows:
        primary = row['shapes']['primary760']; power = row['board_power_w']
        lines.append(f"| {row['batch']} | {row['migration_median_ms']:.3f} | {primary['selected_ms']:.3f} | {primary['matched_gain_percent']:+.3f}% | {power['mean']:.2f}/{power['p95']:.2f}/{power['max']:.2f} |")
    lines += ['', '迁移记录与当前结果属于不同测量轮次；末次收益仅针对各自报告中的同轮对照，不能跨轮相加。B64的末次对照已包含TF32注意力优化。', '',
              '| Batch | 608帧(ms) | 760帧(ms) | 918帧(ms) |', '|---:|---:|---:|---:|']
    for row in rows:
        s = row['shapes']
        lines.append(f"| {row['batch']} | {s['short']['selected_ms']:.3f} | {s['primary760']['selected_ms']:.3f} | {s['long']['selected_ms']:.3f} |")
    lines += ['', '真实音频测速采用608/760/918帧；引擎600/760/920边界另作接口验证。', '',
              '完整证据与长度取舍见 [最终审核](final-execution-cost-review.json) 和 [机器可读结果](final-results.json)。云端发布及清理完成情况须另行核对发布回执。', '']
    (REPORTS / 'FINAL_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'batches': len(rows)}))


if __name__ == '__main__':
    main()
