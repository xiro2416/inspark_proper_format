"""Accept a B32 compute baseline before transferring source application scheduling."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
HISTORY=ROOT/'deployment/b32/history'


def main():
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or Path(os.environ.get('INDEX_HISTORY_DIR','')).resolve()!=HISTORY:
        raise RuntimeError('Source deployment/b32/env.sh; dedicated B32 history and GPU1 required')
    from deployment.validate_matrix import run, benchmark, main as validate_matrix
    complete=HISTORY/'migration-complete.json'
    if complete.is_file():
        baseline=json.loads((HISTORY/'migration-selected.json').read_text())
        for component in ('target','draft','prefill','latent','cfm','vocoder'):
            plan_path=Path(baseline[component+'_plan']);plan=json.loads(plan_path.read_text())
            binary=Path(plan['engine']);binary=binary if binary.is_absolute() else plan_path.parent/binary
            with binary.open('rb') as stream:observed=hashlib.file_digest(stream,'sha256').hexdigest()
            if observed!=plan['sha256'] or plan['batch']!=32 or plan['sm']!=89:
                raise RuntimeError('Archived migration baseline identity changed; review before replacing it')
        print(json.dumps(dict(status='previously_validated_migration_preserving_current_best',batch=32)),flush=True)
        return
    from inspark_infer.runtime.unified_deployment import validate
    rows=[]
    for component in ('target','draft','prefill','latent','cfm-estimator','vocoder-gemm'):
        directory=ROOT/f'artifacts/sm89/int8_smoothquant/b32/{component}'
        plan=json.loads((directory/'model.plan.json').read_text())
        with (directory/'model.engine').open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert plan['batch']==32 and plan['sm']==89 and plan['sha256']==actual
        layers=json.loads((directory/'model.inspector.json').read_text())['Layers']
        integer=[layer for layer in layers if 'i8i32' in layer.get('TacticName','')]
        if not integer:raise RuntimeError('Missing inherited actual INT8 compute: '+component)
        rows.append(dict(component=component,batch=32,engine_sha256=actual,int8_tactic_layers=len(integer),
                         inspector=str(directory/'model.inspector.json'),
                         example_tactics=list(dict.fromkeys(layer['TacticName'] for layer in integer))[:3]))
    (HISTORY/'int8-kernel-evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    run('minimal-materialize-b32','deployment/materialize_plans.py','--batches',32,'--cfm-directory','cfm-estimator')
    configs=ROOT/'configs/hardware/sm89/indextts'
    plain=json.loads((configs/'int8_b32_framework_graph.json').read_text())
    plain.update(batch_conditions=False,latent_cached_prefix=False)
    validate(plain)
    minimal=configs/'int8_b32_minimal_plain.json'
    minimal.write_text(json.dumps(plain,indent=2)+'\n')
    result=benchmark(32,minimal,'minimal-compute-b32')
    counters=result['measured_counters']
    assert result['execution_pass'] and all(counters[k]==0 for k in
           ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks'))
    # Compare target-shaped real inputs with the original same-recipe reference
    # before migrating grouped conditions and cached latent prefix scheduling.
    for component in ('ar','acoustics'):
        script=f'scripts/audit_unified_{component}.py'
        capture=HISTORY/f'minimal-capture-{component}-b32.pt'
        audit=HISTORY/f'minimal-audit-{component}-b32.json'
        coverage=['--waves',1] if component=='acoustics' else []
        run(f'minimal-capture-{component}-b32',script,'capture','--gpu',1,
            '--config','local_assets/runtime/runtime_fp32_b1.yaml',
            '--manifest','deployment/history/validation-manifest.json',
            '--deployment',minimal,'--output',capture,*coverage)
        run(f'minimal-audit-{component}-b32',script,'audit','--gpu',1,
            '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',capture,'--output',audit)
    (HISTORY/'compute-baseline-passed.json').write_text(json.dumps(dict(batch=32,passed=True,
        deployment=str(minimal),execution='minimal-compute-b32.json',
        reference_audits=['minimal-audit-ar-b32.json','minimal-audit-acoustics-b32.json'],
        scope='compute validated before source application scheduling transfer'),indent=2)+'\n')
    sys.argv=['validate_matrix','--batches','32']
    validate_matrix()
    selected=configs/'int8_b32_selected.json'
    shutil.copy2(selected,HISTORY/'migration-selected.json')
    shutil.copy2(HISTORY/'final-int8-b32.json',HISTORY/'migration-final-int8-b32.json')
    lifecycle=json.loads((HISTORY/'lifecycle-selected-b32.json').read_text())
    archive=ROOT/'outputs/b32/migration-lifecycle'
    shutil.copytree(ROOT/'outputs/lifecycle/b32',archive,dirs_exist_ok=True)
    for group in ('first','replay_after_cancel','partial_batch'):
        for row in lifecycle.get(group,[]):
            row['wav']=str(archive/Path(row['wav']).name)
    (HISTORY/'migration-lifecycle-selected-b32.json').write_text(json.dumps(lifecycle,ensure_ascii=False,indent=2)+'\n')
    summary=json.loads((HISTORY/'validated-matrix.json').read_text())
    (HISTORY/'migration-complete.json').write_text(json.dumps(dict(status='migration_validated',batch=32,
        baseline=summary,selected='migration-selected.json',benchmark='migration-final-int8-b32.json',
        next_phase='optimization, separately requested by user'),indent=2)+'\n')
    print(json.dumps(dict(status='migration_validated',batch=32)),flush=True)


if __name__=='__main__':main()
