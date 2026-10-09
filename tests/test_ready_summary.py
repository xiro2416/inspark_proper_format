import pytest

from scripts.summarize_ready_first_chunk import aggregate, markdown, validate


def sample(batch):
    dist = lambda x: dict(n=30, p50=x, p95=x+1, p99=x+2, mean=x, min=x, max=x+2)
    modes, power = {}, {}
    for mode, value in [('baseline', 100), ('barrier16', 90), ('A', 85), ('C', 80)]:
        clock = {key: dist(value) for key in ('first1_ms', 'first16_ms', 'last_ms',
                                              'request_median_ms', 'request_p95_ms')}
        modes[mode] = dict(waves=[{'rows': [{'text': 'must never publish'}]}] * 30,
            summary=dict(admission=clock, postadmission=clock),
            execution=dict(totals=dict(status_reads=1, status_d2h_bytes=4,
                metadata_reads=2, metadata_rows=batch, metadata_d2h_bytes=100),
                ar_target_enqueues_by_batch={str(batch): 4}, acoustic_groups=2,
                acoustic_real_rows=batch, acoustic_padded_rows=32))
        power[mode] = dict(requested_seconds=30, duration_seconds=30.1, waves=10,
                           power_w=dist(300), memory_mib=dist(12000), utilization_percent=dist(70))
    return dict(status='validated', execution_pass=True, batch=batch, gpu=7, warmups=5,
        requested_waves=30, repeated_text_reuse_gain_included=False,
        comparison_common_engine_residency=True, modes=modes, power=power,
        identity_checks=dict(baseline_A=dict(identical=True), barrier16_A=dict(identical=True),
                             C_repeat=dict(identical=True, waves=[{'identical': True}] * 30)),
        lifecycle=[dict(requests=n, clean=True, stable=True) for n in [1, 15, 17, batch-1, batch]],
        final_state=dict(root=dict(device_round_fallbacks=0, native_cfm_fallbacks=0,
                                   native_vocoder_fallbacks=0, unified_dspark=dict(failures=0)),
                         acoustic=dict(cfm_fallbacks=0, vocoder_fallbacks=0)),
        production_barrier_compacts=batch == 128, board_lifecycle=dict(memory_mib=dist(12000)),
        manifest_sha256='cases-hash', deployment_sha256='deployment-hash',
        ready_manifest_sha256='ready-hash', config_sha256='config-hash')


def test_aggregate_excludes_raw_request_data_and_last_is_not_gate():
    report = sample(32)
    report['modes']['C']['summary']['admission']['last_ms'] = dict(
        report['modes']['C']['summary']['admission']['last_ms'], p50=150)
    value = aggregate(report, 32, 'source-hash')
    assert value['criterion_pass']
    assert value['improvement_percent']['last_p50'] == -50
    assert 'must never publish' not in str(value)
    assert all('waves' not in mode and 'rows' not in mode for mode in value['modes'].values())
    assert value['sampled_shared_peak_memory_gib'] == 12002 / 1024


@pytest.mark.parametrize('change', ['incomplete', 'waves', 'power', 'fallback', 'stability', 'lifecycle'])
def test_incomplete_or_unvalidated_inputs_rejected(change):
    report = sample(32)
    if change == 'incomplete': report['status'] = 'incomplete'
    if change == 'waves': report['modes']['C']['waves'] = []
    if change == 'power': report['power']['A']['requested_seconds'] = 15
    if change == 'fallback': report['final_state']['acoustic']['cfm_fallbacks'] = 1
    if change == 'stability': report['identity_checks']['C_repeat']['identical'] = False
    if change == 'lifecycle': report['lifecycle'].pop()
    with pytest.raises(ValueError): validate(report, 32)


def test_render_contains_all_three_batches_and_honest_compaction_boundary():
    values = {str(batch): aggregate(sample(batch), batch, 'source') for batch in [32,64,128]}
    text = markdown({'batches': values})
    assert 'B128' in text and 'baseline/A' in text and '完整四步 CFM' in text
    assert text.count('| 提前交付／AR缩批 |') == 6
    assert '持续到达' in text and 'CER/MOS' in text
