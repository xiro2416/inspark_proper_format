"""Bind migration scheduling probes to explicitly reviewed compute routes."""
import json
from pathlib import Path
from run_zipvoice_validation import ROOT, inventory, sha


def selected_route(batch, routes_path, case):
    route = 'inherited'
    if routes_path:
        path = routes_path.resolve()
        path.relative_to(ROOT)
        mapping = json.loads(path.read_text())
        route = mapping[str(batch)]
    if route not in ('native', 'inherited', 'geo1', 'geo3'):
        raise ValueError(f'Unsupported reviewed route: {route}')
    history = ROOT / f'reports/sm89/zipvoice/a1007/b{batch}/history'
    if route in ('native', 'inherited'):
        performance = history / '005-migration-performance.json'
        quality = history / '004-quality.json'
        assert json.loads(performance.read_text())['status'] == 'matched_migration_measurements_complete_review_pending'
    else:
        performance = history / f'020-{route}-performance.json'
        quality = history / f'019-{route}-quality.json'
        record = json.loads(performance.read_text())
        assert record['status'] == 'formal_candidate_e2e_quality_power_complete_review_pending'
        assert record['quality_evidence_sha256'] == sha(quality)
        assert record['application_evidence_sha256'] == sha(history / f'018-{route}-application.json')
    assert json.loads(quality.read_text())['status'] == 'complete'
    identity = inventory(batch, route, case['workload'])
    if route not in ('native', 'inherited'):
        assert record['engine_identities'][route]['engine_sha256'] == identity['engines']['fm']['sha256']
        assert json.loads((history / f'018-{route}-application.json').read_text())['engine_sha256'] == identity['engines']['fm']['sha256']
    return route, {'route': route, 'engine_sha256': identity['engines']['fm']['sha256'],
                   'plugin_package': identity['plugin_package'],
                   'performance_evidence_sha256': sha(performance),
                   'quality_evidence_sha256': sha(quality)}


def idle_gpu_preflight():
    """Recheck identified idle shared state without touching external work."""
    import subprocess
    import time
    processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
    assert not any('evaluate_zipvoice_audio.py --device' in line for line in processes), 'Wait for CPU quality evaluation before decisive delivery measurements'
    for attempt in range(10):
        row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True).strip()
        memory,util,cap,enforced=map(float,row.split(','))
        active=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        known=active=='1820496, 4082' and not Path('/proc/1820496').exists() and memory==4108
        assert cap==enforced==400
        if util==0 and memory<=4352 and (not active or known):
            return {'memory_mib':memory,'gpu_util_percent':util,'processes':active,'power_cap_w':cap,'scope':'Known idle allocation retained; not exclusive device ownership'}
        time.sleep(1)
    raise RuntimeError(f'GPU1 idle state changed; preserve external work: {row}, {active}')
