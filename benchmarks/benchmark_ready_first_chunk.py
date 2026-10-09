#!/usr/bin/env python3
"""Matched owner-callback readiness measurements, with one independent wave unit.

No CUDA module is imported until physical GPU selection. All modes retain the
same candidate engines, so memory is the common resident comparison footprint.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import time

from benchmarks.unified_first_chunk import (
    counter_delta, distribution, first_segment_diversity, load_manifest,
    run_wave, save_json, sha256_file, wave_cases,
)

MODES = ('baseline', 'barrier16', 'A', 'C')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--batch', type=int, choices=(32, 64, 128), required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--deployment', type=Path, required=True, help='Selected production barrier deployment')
    p.add_argument('--ready-manifest', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True, help='Verified bilingual input cases')
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--waves', type=int, default=30)
    p.add_argument('--power-seconds', type=float, default=30)
    p.add_argument('--lifecycle', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    return p


def wave_summary(wave):
    """Reduce correlated requests to scalar values before across-wave quantiles."""
    rows = wave['rows']
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise RuntimeError('Missing or repeated request identity in wave')
    result = {}
    for label, field in [('admission', 'admission_to_pcm_ms'),
                         ('postadmission', 'postadmission_to_pcm_ms')]:
        ordered = sorted(r[field] for r in rows)
        result[label] = dict(first1_ms=ordered[0],
                             first16_ms=ordered[min(16, len(ordered)) - 1],
                             last_ms=ordered[-1], request_median_ms=statistics.median(ordered),
                             request_p95_ms=distribution(ordered)['p95'])
    return result


def summarize_waves(waves):
    return dict(statistics_unit='one wave; within-wave requests are correlated',
                request_tail_method='within-wave request P95, then distribution across waves',
                **{clock: {key: distribution(w['summary'][clock][key] for w in waves)
                           for key in ('first1_ms', 'first16_ms', 'last_ms',
                                       'request_median_ms', 'request_p95_ms')}
                   for clock in ('admission', 'postadmission')})


def summarize_routes(waves):
    routes = [wave['route'] for wave in waves if wave.get('route') is not None]
    if not routes:
        return dict(ready_route_waves=0, note='Production barrier counters are retained in measured_counters')
    batches = {}
    for route in routes:
        for batch, count in route['ar_batches'].items():
            batches[batch] = batches.get(batch, 0) + count
    fields = ('rounds', 'row_rounds', 'status_reads', 'status_wait_ms',
              'metadata_reads', 'metadata_rows', 'metadata_d2h_bytes', 'status_d2h_bytes')
    groups = [group for route in routes for group in route['acoustic_groups']]
    return dict(ready_route_waves=len(routes), ar_target_enqueues_by_batch=batches,
                totals={field: sum(route[field] for route in routes)
                        if all(field in route for route in routes) else None for field in fields},
                compaction_count=sum(len(route['compactions']) for route in routes),
                acoustic_groups=len(groups), acoustic_real_rows=sum(group['rows'] for group in groups),
                acoustic_padded_rows=len(groups) * 16,
                missing_field_note='null indicates telemetry unavailable; no additional D2H added to measure it')


def identity_map(wave, *, pcm=True):
    result = {}
    for row in wave['rows']:
        key = row['case_id']
        if key in result:
            raise RuntimeError('Duplicate case identity')
        value = [row['code_sha256'], row['accepted'], row['rounds'], row.get('eos')]
        if pcm:
            value.append(row['pcm_sha256'])
        result[key] = value
    return result


class CallbackGuard:
    """Check owner callbacks; returned events may legitimately repeat callbacks."""
    def __init__(self, engine):
        self.engine = engine
        self.expected = set()
        self.delivered = set()

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def admit_batch(self, payload):
        self.expected = {row['request_id'] for row in payload}
        if len(self.expected) != len(payload):
            raise RuntimeError('Duplicate admission')
        self.delivered = set()
        return self.engine.admit_batch(payload)

    def run_ready(self, on_chunk=None):
        def receive(event):
            key = event['request_id']
            if key not in self.expected:
                raise RuntimeError('Unexpected/padded request callback')
            if key in self.delivered:
                raise RuntimeError('Duplicate owner PCM callback')
            self.delivered.add(key)
            if on_chunk:
                on_chunk(event)
        return self.engine.run_ready(on_chunk=receive)


def assert_clean(stats):
    if stats['sessions'] or stats['active_rows'] or stats['error_sessions']:
        raise RuntimeError('Session/active-row lifecycle leak')
    for key in ('target_slots', 'draft_slots'):
        state = stats.get(key)
        if state and (not state['valid'] or state['leased']):
            raise RuntimeError(f'{key} ownership leak')


def assert_no_fallbacks(delta):
    for key in ('device_round_fallbacks', 'native_cfm_fallbacks', 'native_vocoder_fallbacks'):
        if delta.get(key, 0):
            raise RuntimeError(f'Undeclared fallback: {key}')
    if delta.get('unified_dspark', {}).get('failures', 0):
        raise RuntimeError('Native DSpark failure')


def power_samples(samples, start, end):
    selected = [row for row in samples if start <= row[0] <= end]
    if not selected:
        raise RuntimeError('No sensor samples inside power window')
    return dict(power_w=distribution(row[2] for row in selected),
                memory_mib=distribution(row[1] for row in selected),
                utilization_percent=distribution(row[3] for row in selected),
                raw_samples=[dict(elapsed_seconds=r[0] - start, memory_mib=r[1],
                                  power_w=r[2], utilization_percent=r[3]) for r in selected])


def run(args):
    if args.warmups < 0 or args.waves < 1 or args.power_seconds < 0:
        raise ValueError('Invalid measurement counts')
    if len(set(args.modes)) != len(args.modes):
        raise ValueError('Repeated mode')
    # Must precede config/model/scheduler imports, including in validation mode.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from benchmarks.board_power import BoardSampler
    from benchmarks.benchmark_unified_first_chunk import prepare_references
    from inspark_infer.runtime.bundle_paths import read_json
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.pool import _engine_stats
    from inspark_infer.runtime.ready_scheduler import ReadyPipeline

    inputs = load_manifest(args.manifest)
    cases = inputs['splits']['evaluation']
    config = load(args.config)
    config.update(max_batch=args.batch, precision_batches=[args.batch])
    root_plan = dict(load_deployment(args.deployment), batch_text_dedup=False)
    if root_plan['batch'] != args.batch:
        raise ValueError('Barrier deployment batch does not match admission batch')
    ready = read_json(args.ready_manifest)
    report = dict(schema=1, kind='matched_ready_first_chunk', status='incomplete',
                  gpu=args.gpu, batch=args.batch, warmups=args.warmups,
                  requested_waves=args.waves, profiler_enabled=False,
                  manifest_sha256=inputs['manifest_sha256'],
                  deployment_sha256=sha256_file(args.deployment),
                  ready_manifest_sha256=sha256_file(args.ready_manifest),
                  config_sha256=sha256_file(args.config),
                  modes={mode: {'waves': []} for mode in args.modes}, power={},
                  workload_diversity=first_segment_diversity(cases, args.batch),
                  repeated_text_reuse_gain_included=False,
                  benchmark_overrides={'batch_text_dedup': False},
                  scope='request admission and admission acknowledgment to immediate owner PCM callback',
                  observation='No profiler or additional measurement synchronization; production status/metadata D2H and PCM handoff included',
                  pooling_note='Wave scalar distributions; not pooled correlated request P99',
                  source_barrier_unchanged=True, comparison_common_engine_residency=True)
    save = lambda: save_json(args.out, report)
    sampler = BoardSampler(args.gpu)
    engine = pipeline = None
    started = time.perf_counter()
    try:
        with ExitStack() as resources:
            lease = resources.enter_context(GPULease(args.gpu))
            report['initial_memory_mib'] = lease.initial_memory_mib
            sampler.start()
            resources.callback(sampler.stop)
            engine = Engine(config)
            resources.callback(engine.close)
            report['references'] = prepare_references(engine, inputs)
            report['deployment'] = engine.prepare_deployment(root_plan)
            if getattr(engine, 'head_ready_pipeline', None) is not None:
                raise ValueError('Benchmark --deployment must be explicit production barrier, not ready default')
            pipeline = ReadyPipeline(engine, ready, mode='C')
            resources.callback(pipeline.close)
            client = CallbackGuard(engine)
            root_compacts = getattr(engine.unified_first_chunk.runtime, 'compact_tail', None) is not None
            report['production_barrier_compacts'] = root_compacts

            def snapshot():
                return dict(root=_engine_stats(engine),
                            acoustic=dict(cfm_fallbacks=int(getattr(pipeline.acoustic.student, 'fallbacks', 0)),
                                          vocoder_fallbacks=int(getattr(pipeline.acoustic.vocoder, 'fallbacks', 0))))

            def trial(mode, index, *, count=None, detailed=True):
                engine.head_ready_pipeline = None if mode == 'baseline' else pipeline
                pipeline.mode = mode
                selected = wave_cases(cases, args.batch, index)
                if count is not None:
                    selected = selected[:count]
                previous_routes = len(pipeline.waves)
                value = run_wave(client, selected, f'{mode}-{index}-{count}',
                                 details=detailed, admission_mode='batch')
                if client.delivered != client.expected:
                    raise RuntimeError('Missing immediate owner callback')
                value['summary'] = wave_summary(value)
                value['case_offset_wave'] = index
                value['route'] = (copy.deepcopy(pipeline.waves[-1])
                                  if len(pipeline.waves) > previous_routes else None)
                stats = _engine_stats(engine)
                assert_clean(stats)
                return value

            for mode in args.modes:
                for index in range(args.warmups):
                    trial(mode, index, detailed=False)
                print(json.dumps({'warmed': mode}), flush=True)
            before = snapshot()
            assert_no_fallbacks(before['root'])
            if any(before['acoustic'].values()):
                raise RuntimeError('Readiness acoustic preparation/warmup fallback')
            for index in range(args.waves):
                order = args.modes if index % 2 == 0 else list(reversed(args.modes))
                for mode in order:
                    value = trial(mode, index)
                    report['modes'][mode]['waves'].append(value)
                    save()
                    print(json.dumps(dict(mode=mode, wave=index, **value['summary']['admission'])), flush=True)
            after = snapshot()
            report.update(before=before, after=after,
                          measured_counters=counter_delta(before['root'], after['root']),
                          measured_private_acoustic_fallbacks={key: after['acoustic'][key] - before['acoustic'][key]
                                                              for key in before['acoustic']})
            assert_no_fallbacks(report['measured_counters'])
            if before['acoustic'] != after['acoustic']:
                raise RuntimeError('Readiness acoustic fallback')
            for mode, value in report['modes'].items():
                value['summary'] = summarize_waves(value['waves'])
                value['execution'] = summarize_routes(value['waves'])

            checks = {}
            for left, right, pcm in [('baseline', 'A', False), ('barrier16', 'A', True)]:
                if left not in args.modes or right not in args.modes:
                    continue
                key = left + '_' + right
                if left == 'baseline' and root_compacts:
                    checks[key] = dict(checked=False,
                        reason='Production barrier has tail compaction; A uses fixed AR shape. Compare barrier16/A instead.')
                    continue
                equal = all(identity_map(x, pcm=pcm) == identity_map(y, pcm=pcm)
                            for x, y in zip(report['modes'][left]['waves'], report['modes'][right]['waves']))
                checks[key] = dict(checked=True, pcm_checked=pcm, identical=equal,
                                   waves=args.waves, same_ar_shape=True)
                if not equal:
                    raise RuntimeError(f'Fixed-shape identity mismatch: {key}')
            if 'C' in args.modes:
                stability = []
                for index, original in enumerate(report['modes']['C']['waves']):
                    repeated = trial('C', index)
                    same = identity_map(original) == identity_map(repeated)
                    stability.append(dict(wave=index, identical=same))
                    if not same:
                        raise RuntimeError(f'C same-input/seed stability mismatch at wave {index}')
                checks['C_repeat'] = dict(checked=True, identical=True, waves=stability,
                                          timing_included=False)
            report['identity_checks'] = checks

            for mode in args.modes:
                if args.power_seconds == 0:
                    continue
                begin = time.perf_counter()
                count = 0
                while count == 0 or time.perf_counter() - begin < args.power_seconds:
                    trial(mode, count, detailed=False)
                    count += 1
                end = time.perf_counter()
                report['power'][mode] = dict(duration_seconds=end - begin,
                    requested_seconds=args.power_seconds, waves=count, requests=count * args.batch,
                    scope='Independent sustained full waves, including admission and cleanup',
                    **power_samples(sampler.samples, begin, end))
                save()

            if args.lifecycle:
                lifecycle = []
                for count in dict.fromkeys((1, 15, 17, args.batch - 1, args.batch)):
                    first = trial('C', 0, count=count)
                    second = trial('C', 0, count=count)
                    same = identity_map(first) == identity_map(second)
                    if not same:
                        raise RuntimeError(f'Partial same-seed mismatch: N={count}')
                    lifecycle.append(dict(requests=count, stable=True, clean=True,
                                          route=first['route'], natural_model_outputs=True))
                report['lifecycle'] = lifecycle
            final = snapshot()
            assert_no_fallbacks(counter_delta(before['root'], final['root']))
            if final['acoustic'] != before['acoustic']:
                raise RuntimeError('Final readiness acoustic fallback')
            report.update(final_state=final,
                          ready_acoustic_graphs=pipeline.acoustic.head_graphs.stats(),
                          status='validated', execution_pass=True)
    except Exception as error:
        report.update(status='failed', execution_pass=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['board_lifecycle'] = dict(duration_seconds=time.perf_counter() - started,
            sensor_samples=len(sampler.samples), memory_mib=distribution(r[1] for r in sampler.samples),
            scope='Whole process: model/reference, all candidate engines/captures, all modes, power, lifecycle and cleanup',
            note='20ms NVML samples may miss shorter allocations; this is not old standalone production memory')
        save()
    return report


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
