from benchmarks.benchmark_ready_first_chunk import (
    CallbackGuard, assert_clean, identity_map, parser, power_samples,
    summarize_routes, summarize_waves, wave_summary,
)
import pytest


def rows(count):
    return [dict(case_id=str(i), admission_to_pcm_ms=i + 1,
                 postadmission_to_pcm_ms=i, code_sha256=str(i),
                 pcm_sha256=str(i), accepted=[1], rounds=1) for i in range(count)]


def test_wave_statistics_do_not_pool_requests():
    values = []
    for n in (17, 32):
        wave = {'rows': rows(n)}
        wave['summary'] = wave_summary(wave)
        values.append(wave)
    summary = summarize_waves(values)
    assert summary['admission']['first16_ms']['n'] == 2
    assert summary['admission']['first16_ms']['p50'] == 16
    assert summary['admission']['last_ms']['p50'] == 24.5
    assert wave_summary({'rows': rows(1)})['admission']['first16_ms'] == 1


def test_power_samples_are_exact_window_and_retain_p99():
    value = power_samples([(0., 100., 1., 1.), (1., 200., 3., 2.),
                           (2., 300., 5., 3.)], .5, 2.)
    assert value['power_w']['mean'] == 4
    assert value['power_w']['max'] == 5
    assert value['power_w']['p99'] == 4.98
    assert len(value['raw_samples']) == 2


def test_identity_excludes_pcm_only_when_explicit():
    left = {'rows': rows(2)}
    right = {'rows': rows(2)}
    right['rows'][0]['pcm_sha256'] = 'different'
    assert identity_map(left, pcm=False) == identity_map(right, pcm=False)
    assert identity_map(left) != identity_map(right)
    with pytest.raises(RuntimeError):
        identity_map({'rows': rows(1) * 2})


def test_route_summary_keeps_native_buckets_and_missing_evidence_explicit():
    route = dict(ar_batches={'32': 4, '16': 2}, rounds=6, row_rounds=160,
                 status_reads=3, status_wait_ms=1.2, metadata_reads=2,
                 compactions=[{'from_batch': 32, 'to_batch': 16}],
                 acoustic_groups=[{'rows': 16}, {'rows': 1}])
    summary = summarize_routes([{'route': route}, {'route': route}])
    assert summary['ar_target_enqueues_by_batch'] == {'32': 8, '16': 4}
    assert summary['totals']['metadata_reads'] == 4
    assert summary['totals']['metadata_d2h_bytes'] is None
    assert summary['acoustic_real_rows'] == 34 and summary['acoustic_padded_rows'] == 64
    assert summarize_routes([{'route': None}])['ready_route_waves'] == 0


def test_callback_guard_checks_actual_callbacks_not_returned_events():
    class Client:
        def admit_batch(self, payload):
            return {}
        def run_ready(self, on_chunk):
            item = {'request_id': 'real'}
            on_chunk(item)
            return [item]
    guard = CallbackGuard(Client())
    guard.admit_batch([{'request_id': 'real'}])
    assert guard.run_ready()[0]['request_id'] == 'real'
    with pytest.raises(RuntimeError, match='Duplicate owner'):
        guard.run_ready()


def test_invalid_ownership_and_public_parser_defaults():
    with pytest.raises(RuntimeError, match='ownership'):
        assert_clean(dict(sessions=0, active_rows=0, error_sessions=0,
                          target_slots=dict(valid=True, leased=1)))
    args = parser().parse_args(['--gpu', '6', '--batch', '32', '--config', 'config',
        '--deployment', 'deployment', '--ready-manifest', 'ready', '--manifest', 'cases', '--out', 'out'])
    assert args.waves == 30 and args.warmups == 5 and args.power_seconds == 30
