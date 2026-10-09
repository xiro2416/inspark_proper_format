"""Validate a B32 candidate, optionally retain it, while preserving migration records."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import run,benchmark
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--select',action='store_true');a=p.parse_args()
    h=ROOT/'deployment/b32/history';path=a.candidate.resolve();plan=json.loads(path.read_text())
    if plan['batch']!=32:raise RuntimeError('Only B32 candidates accepted')
    lifecycle=h/f'lifecycle-{a.label}-b32.json'
    previous=json.loads(lifecycle.read_text()) if lifecycle.exists() else {}
    if not (previous.get('passed') and previous.get('deployment_attestation',{}).get('requested')==plan):
        run(f'lifecycle-{a.label}-b32','deployment/validate_lifecycle.py','--batch',32,'--deployment',path,'--output',lifecycle)
    if not json.loads(lifecycle.read_text())['passed']:raise RuntimeError('Lifecycle failed')
    capture=h/f'replay-acoustics-{a.label}-b32.pt';audit=h/f'audit-acoustics-{a.label}-b32.json'
    migration=json.loads((h/'migration-selected.json').read_text())
    keys=set(plan)|set(migration)
    runtime_changed=any(plan.get(k)!=migration.get(k) for k in keys-{'vocoder_plan','status'})
    ar_audit='audit-ar-b32.json'
    if runtime_changed:
        ar_capture=h/f'capture-ar-{a.label}-b32.pt';ar_path=h/f'audit-ar-{a.label}-b32.json'
        run(f'capture-ar-{a.label}-b32','scripts/audit_unified_ar.py','capture','--gpu',1,
            '--config','local_assets/runtime/runtime_fp32_b1.yaml','--manifest','deployment/history/validation-manifest.json',
            '--deployment',path,'--output',ar_capture)
        run(f'audit-ar-{a.label}-b32','scripts/audit_unified_ar.py','audit','--gpu',1,
            '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',ar_capture,'--output',ar_path)
        ar_audit=ar_path.name
    run(f'replay-acoustics-{a.label}-b32','scripts/audit_unified_acoustics.py','replay','--gpu',1,
        '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',h/'capture-acoustics-b32.pt',
        '--cfm-plan',plan['cfm_plan'],'--vocoder-plan',plan['vocoder_plan'],'--output',capture)
    run(f'audit-acoustics-{a.label}-b32','scripts/audit_unified_acoustics.py','audit','--gpu',1,
        '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',capture,'--output',audit)
    final=benchmark(32,path,f'final-{a.label}-int8-b32',final=True)
    counters=final['measured_counters']
    if not final['execution_pass'] or any(counters[k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('First-head fallback')
    if any(counters['head_graph_hits'][k]!=30 for k in ('cfm','vocoder')):raise RuntimeError('Missing graph coverage')
    evidence=dict(status='candidate_validated',batch=32,candidate=str(path),label=a.label,lifecycle=str(lifecycle),
        benchmark=f'final-{a.label}-int8-b32.json',acoustic_audit=str(audit),
        ar_reference=ar_audit,ar_note='Runtime-specific frozen AR audit' if runtime_changed else 'AR engines/runtime unchanged; migration audit retained',
        p50_ms=final['summary']['wave_admission_to_last_pcm_ms']['p50'],p95_ms=final['summary']['wave_admission_to_last_pcm_ms']['p95'])
    if a.select:
        plan['status']='validated_local_sm89_int8'
        selected=ROOT/'configs/hardware/sm89/indextts/int8_b32_selected.json';selected.write_text(json.dumps(plan,indent=2)+'\n')
        (h/'current-best-selected.json').write_text(json.dumps(plan,indent=2)+'\n');evidence['selected']=str(selected)
        (h/'current-best.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (h/f'accepted-{a.label}.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence),flush=True)


if __name__=='__main__':main()
