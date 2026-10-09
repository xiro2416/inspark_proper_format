#!/usr/bin/env python3
"""Publish aggregate ready-C evidence only after all three formal runs validate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCHES = (32, 64, 128)
MODES = ('baseline', 'barrier16', 'A', 'C')
LABELS = {'baseline': '当前屏障', 'barrier16': 'AR屏障＋B16声学',
          'A': '提前交付／固定AR', 'C': '提前交付／AR缩批'}


def gain(before, after):
    return 100 * (before - after) / before


def validate(report, batch):
    if report.get('status') != 'validated' or report.get('execution_pass') is not True:
        raise ValueError(f'B{batch}: formal report has not validated')
    if report.get('batch') != batch or report.get('gpu') != 7:
        raise ValueError(f'B{batch}: unexpected batch/GPU')
    if report.get('warmups') != 5 or report.get('requested_waves') != 30:
        raise ValueError(f'B{batch}: expected five warmups and thirty measured waves')
    if report.get('repeated_text_reuse_gain_included') is not False:
        raise ValueError(f'B{batch}: repeated-text reuse policy missing')
    if report.get('comparison_common_engine_residency') is not True:
        raise ValueError(f'B{batch}: common candidate residency not confirmed')
    for mode in MODES:
        measured = report.get('modes', {}).get(mode, {})
        if len(measured.get('waves', [])) != 30 or not measured.get('summary'):
            raise ValueError(f'B{batch}/{mode}: thirty-wave summary missing')
        power = report.get('power', {}).get(mode, {})
        if power.get('requested_seconds') != 30 or power.get('duration_seconds', 0) < 30:
            raise ValueError(f'B{batch}/{mode}: thirty-second independent power window missing')
    checks = report.get('identity_checks', {})
    for key in ('barrier16_A', 'C_repeat'):
        if checks.get(key, {}).get('identical') is not True:
            raise ValueError(f'B{batch}: {key} identity check not passed')
    if len(checks['C_repeat'].get('waves', [])) != 30:
        raise ValueError(f'B{batch}: thirty C stability replays missing')
    if not report.get('production_barrier_compacts'):
        if checks.get('baseline_A', {}).get('identical') is not True:
            raise ValueError(f'B{batch}: fixed-shape baseline/A identity check not passed')
    required_counts = {1, 15, 17, batch - 1, batch}
    lifecycle = report.get('lifecycle', [])
    if {x.get('requests') for x in lifecycle} != required_counts or not all(
            x.get('clean') and x.get('stable') for x in lifecycle):
        raise ValueError(f'B{batch}: real-request lifecycle checks missing')
    root = report.get('final_state', {}).get('root', {})
    for key in ('device_round_fallbacks', 'native_cfm_fallbacks', 'native_vocoder_fallbacks'):
        if key not in root or root[key] != 0:
            raise ValueError(f'B{batch}: undeclared fallback/missing evidence: {key}')
    if root.get('unified_dspark', {}).get('failures') != 0:
        raise ValueError(f'B{batch}: DSpark failure evidence missing/nonzero')
    acoustic = report.get('final_state', {}).get('acoustic', {})
    if set(acoustic) != {'cfm_fallbacks', 'vocoder_fallbacks'} or any(acoustic.values()):
        raise ValueError(f'B{batch}: private acoustic fallback evidence missing/nonzero')


def aggregate(report, batch, source_sha256):
    validate(report, batch)
    modes = {}
    for mode in MODES:
        measured, power = report['modes'][mode], report['power'][mode]
        modes[mode] = dict(label=LABELS[mode], latency=measured['summary'],
                          execution=measured.get('execution', {}),
                          power=dict(duration_seconds=power['duration_seconds'],
                                     waves=power['waves'], power_w=power['power_w'],
                                     memory_mib=power['memory_mib'],
                                     utilization_percent=power['utilization_percent']))
    base = modes['baseline']['latency']['admission']
    selected = modes['C']['latency']['admission']
    criteria = {
        'request_median_p50_improved': selected['request_median_ms']['p50'] < base['request_median_ms']['p50'],
        'request_median_p95_improved': selected['request_median_ms']['p95'] < base['request_median_ms']['p95'],
        'first16_p50_improved': selected['first16_ms']['p50'] < base['first16_ms']['p50'],
    }
    return dict(batch=batch, gpu=7, formal_source_sha256=source_sha256,
                provenance={key: report[key] for key in (
                    'manifest_sha256', 'deployment_sha256', 'ready_manifest_sha256', 'config_sha256')},
                modes=modes, criterion_pass=all(criteria.values()), criteria=criteria,
                improvement_percent=dict(
                    request_median_p50=gain(base['request_median_ms']['p50'], selected['request_median_ms']['p50']),
                    request_median_p95=gain(base['request_median_ms']['p95'], selected['request_median_ms']['p95']),
                    first16_p50=gain(base['first16_ms']['p50'], selected['first16_ms']['p50']),
                    last_p50=gain(base['last_ms']['p50'], selected['last_ms']['p50'])),
                sampled_shared_peak_memory_gib=report['board_lifecycle']['memory_mib']['max'] / 1024,
                identity_checks={key: {k: v for k, v in value.items() if k != 'waves'}
                                 for key, value in report['identity_checks'].items()},
                lifecycle={str(x['requests']): {'stable': x['stable'], 'clean': x['clean']}
                           for x in report['lifecycle']},
                fallback_counters={key: root_value for key, root_value in report['final_state']['root'].items()
                                   if key in ('device_round_fallbacks', 'native_cfm_fallbacks', 'native_vocoder_fallbacks')},
                private_acoustic_fallbacks=report['final_state']['acoustic'],
                production_barrier_compacts=report['production_barrier_compacts'])


def triple(value):
    return ' / '.join(f'{value[key]:.2f}' for key in ('p50', 'p95', 'p99'))


def markdown(summary):
    batches = summary['batches']
    lines = ['# C 就绪调度：B32 / B64 / B128 首 chunk 验证', '',
        '当前 NVFP4 GEMM＋FP8 Conv、90 个 BF16 保护角色及完整四步 CFM。所有 GPU 实验在物理 GPU7 串行完成；每档5波预热、30波测量，模式顺序交替/反转，输入与 seed 匹配，另测各模式30秒功率。', '',
        '时间从请求 admission 到 owner 线程即时 PCM 回调。下表每格为跨波 P50/P95/P99；逐请求中位数和请求 P95 先在每波内计算，再对30波统计，不能解释为独立请求池的 P99。未删除波动或 outlier。', '',
        '## 逐请求与交付延迟（ms）', '',
        '| Batch | 模式 | 波内请求中位数 | 首16交付 | 整批末请求交付 | 波内请求P95 |',
        '|---:|---|---:|---:|---:|---:|']
    for batch in BATCHES:
        for mode in MODES:
            record = batches[str(batch)]['modes'][mode]
            latency = record['latency']['admission']
            lines.append(f"| {batch} | {record['label']} | {triple(latency['request_median_ms'])} | "
                         f"{triple(latency['first16_ms'])} | {triple(latency['last_ms'])} | "
                         f"{triple(latency['request_p95_ms'])} |")
    lines += ['', '受理完成后的相同统计、首个请求延迟和原始报告哈希保存在 `summary.json`。', '',
              '## 功率与共同驻留显存', '',
              '| Batch | 模式 | 平均 / 峰值 / P95 功率（W） | 共同进程峰值显存（GiB） |',
              '|---:|---|---:|---:|']
    for batch in BATCHES:
        data = batches[str(batch)]
        for mode in MODES:
            record = data['modes'][mode]; watts = record['power']['power_w']
            lines.append(f"| {batch} | {record['label']} | {watts['mean']:.2f} / {watts['max']:.2f} / "
                         f"{watts['p95']:.2f} | {data['sampled_shared_peak_memory_gib']:.2f} |")
    lines += ['', '四种模式在同一进程内驻留相同候选引擎、context 和 Graph。显存为包括加载、捕获、全部模式、功率、生命周期与释放的20ms NVML采样峰值；不是旧独立屏障部署的显存，也不是各模式独立峰值。功率为真实独立窗口，平均、峰值和分位数分别报告；功耗下降不作为性能收益。', '',
              '## C 相对当前屏障的验收', '',
              '| Batch | 请求中位P50改善 | 请求中位P95改善 | 首16 P50改善 | 末请求P50改善 | 三项提前交付标准 |',
              '|---:|---:|---:|---:|---:|---|']
    for batch in BATCHES:
        data = batches[str(batch)]; gain_values = data['improvement_percent']
        values = [f'{gain_values[key]:.2f}%' for key in ('request_median_p50', 'request_median_p95', 'first16_p50', 'last_p50')]
        lines.append(f"| {batch} | {' | '.join(values)} | {'通过' if data['criterion_pass'] else '未通过，保留屏障默认'} |")
    lines += ['', '改善率为 `100 × (屏障 − C) / 屏障`，末请求列为负表示退化。验收要求请求中位P50/P95及首16 P50均改善；不对整批末请求退化设硬门槛。正式匹配结果支持当前条件下的选择，不宣称全局最优。', '',
              '## C 实际执行与通信', '',
              '| Batch | 实际 AR 桶（target rounds） | 状态读取 / bytes | 元数据读取 / 请求行 / bytes | 声学组 / 真实行 / 含填充行 |',
              '|---:|---|---:|---:|---:|']
    for batch in BATCHES:
        execution = batches[str(batch)]['modes']['C']['execution']
        totals = execution['totals']
        bucket_text = ', '.join(f'B{b}: {n}' for b, n in sorted(execution['ar_target_enqueues_by_batch'].items(), key=lambda x: -int(x[0])))
        status = f"{totals['status_reads']} / {totals['status_d2h_bytes']}"
        metadata = f"{totals['metadata_reads']} / {totals['metadata_rows']} / {totals['metadata_d2h_bytes']}"
        groups = f"{execution['acoustic_groups']} / {execution['acoustic_real_rows']} / {execution['acoustic_padded_rows']}"
        lines.append(f'| {batch} | {bucket_text} | {status} | {metadata} | {groups} |')
    lines += ['', '以上计数只覆盖30个正式波，状态读取和新就绪元数据 D2H 仍在真实调度路径内，PCM交付成本也包含在延迟中。KV、logits、接受拒绝及 token 状态保持 GPU 执行；没有声称消除全部同步。其余模式及 row-round、等待时间、缩批次数见聚合 JSON。', '',
              '## 正确性与适用边界', '',
              '- 固定AR的 barrier16/A 比较通过 code、accepted、rounds、EOS 与 PCM 一致性；C 的30波均另作相同输入/seed重放。跨AR shape 的浮点/采样差异按逻辑正确性审计，不使用固定L2门槛。',
              '- B128 的现行屏障本身包含尾部缩批，A 为固定AR；因此 baseline/A 不施加逐位相同门槛，使用同shape的 barrier16/A 作对应控制，例外明确保留。',
              '- 实际1/15/17/N−1/N请求均重复运行并检查稳定性、无漏/重复/padding假交付以及清理；正式与私有声学路径零未声明回退。synthetic EOS、取消、异常后清理另有CPU状态测试，不当作自然语音质量样本。',
              '- 关闭重复文本计算复用；保留prefix→latent suffix语义、GPU PCG、接受位置commit与请求RNG/KV所有权。D并发保持禁用，C为串行就绪调度。',
              '- 本次只验证首chunk与固定预组批工作。未验证持续到达、动态补槽、长期服务稳定性、CER/MOS或尾chunk/完整句子吞吐。', '',
              '原始逐波/功率与请求记录保留于本地 `artifacts/ready_current/formal_b32.json`、`formal_b64.json`、`formal_b128.json`；公开聚合不含文本、case行或原始PCM。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'artifacts/ready_current')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'reports/current/ready_first_chunk')
    args = parser.parse_args()
    # Validate and aggregate every input before creating output: no partial report.
    batches = {}
    for batch in BATCHES:
        source = args.input_root / f'formal_b{batch}.json'
        raw = source.read_bytes()
        batches[str(batch)] = aggregate(json.loads(raw), batch, hashlib.sha256(raw).hexdigest())
    summary = dict(schema=1, kind='ready_first_chunk_aggregate', batches=batches,
                   all_batches_criterion_pass=all(v['criterion_pass'] for v in batches.values()),
                   method=dict(gpu=7, cfm_steps=4, warmups=5, measured_waves=30,
                               power_seconds_per_mode=30, common_candidate_residency=True,
                               first_chunk_only=True, raw_cases_included=False,
                               statistics_unit='per-wave scalars across thirty waves',
                               last_delivery_has_hard_acceptance_threshold=False))
    text = markdown(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.out_dir / 'RESULTS.md').write_text(text)
    print(json.dumps({'out': str(args.out_dir), 'all_batches_criterion_pass': summary['all_batches_criterion_pass']}))


if __name__ == '__main__':
    main()
