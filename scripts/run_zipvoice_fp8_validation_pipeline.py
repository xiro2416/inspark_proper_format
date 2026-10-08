"""Serial single-GPU validation/reference, CPU quality and matched E2E queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,environment,private_report,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-application',action='store_true');p.add_argument('--skip-reference',action='store_true')
    args=p.parse_args();env=environment();logs=ROOT/'outputs/fp8/pipeline';logs.mkdir(parents=True,exist_ok=True)
    gpu_python=ROOT/'.venv-zipvoice-fp8/bin/python';cpu_python=ROOT/'.venv-zipvoice-evaluation/bin/python'
    receipt=logs/'queue.json';stages=[]
    def run(name,script,python=gpu_python,extra=None,execution_env=env):
        command=[str(python),str(ROOT/'scripts'/script),*(extra or [])]
        if name=='migration-measurements':
            command=['taskset','-c',','.join(map(str,sorted(os.sched_getaffinity(0))[:16])),*command]
        entry=dict(stage=name,command=command,status='running',started=time.time());stages.append(entry);write(receipt,stages)
        print(json.dumps(dict(event='stage_started',stage=name)),flush=True)
        with (logs/f'{name}.log').open('w') as f:
            subprocess.run(command,env=execution_env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        entry.update(status='passed',completed=time.time());write(receipt,stages)
    if not args.skip_application:run('application','validate_zipvoice_fp8.py')
    if not args.skip_reference:run('fp32-reference','audit_zipvoice_fp8_reference.py',extra=['--gpu','3'])
    # CUDA visibility is empty for quality metrics. Separate CPU affinity avoids
    # imposing its eight-thread ASR/MOS load on the timed PCM worker cores.
    cpu_env={**env,'CUDA_VISIBLE_DEVICES':''}
    cores=sorted(os.sched_getaffinity(0));eval_cores=cores[len(cores)//2:len(cores)//2+16]
    quality_cmd=['taskset','-c',','.join(map(str,eval_cores)),str(cpu_python),str(ROOT/'scripts/evaluate_zipvoice_fp8.py')]
    with (logs/'quality.log').open('w') as quality_log:
        quality=subprocess.Popen(quality_cmd,env=cpu_env,cwd=ROOT,stdout=quality_log,stderr=subprocess.STDOUT)
        entry=dict(stage='quality',command=quality_cmd,pid=quality.pid,status='running');stages.append(entry);write(receipt,stages)
        run('migration-measurements','measure_zipvoice_fp8.py',extra=['--phase','migration'])
        # Observation only: locate exposed costs while the independent CPU
        # quality queue finishes. New candidates wait for all migrations.
        for batch in (1,16,64):
            run(f'baseline-profile-b{batch}','profile_zipvoice_fp8.py',extra=['--batch',str(batch)])
        while quality.poll() is None:
            print(json.dumps(dict(event='awaiting_cpu_quality',pid=quality.pid)),flush=True);time.sleep(30)
        if quality.returncode:raise RuntimeError('CPU quality evaluation failed; inspect quality.log')
        entry['status']='passed';write(receipt,stages)
    for batch in BATCHES:
        migration_path=private_report(batch,'006-migration');migration=json.loads(migration_path.read_text())
        application=json.loads(private_report(batch,'003-application').read_text())
        quality=json.loads(private_report(batch,'005-quality').read_text())
        if application['status']!='application_and_boundaries_passed_quality_audit_pending' or quality['status']!='paired_quality_metrics_complete':
            raise RuntimeError('Target evidence incomplete')
        migration.update(status='migration_accepted',migration_accepted=True,quality_status=quality['status'],
                         quality_report_sha256=sha(private_report(batch,'005-quality')))
        write(migration_path,migration)
        # Public aggregates contain neither raw evaluation text nor reference
        # voice material. Complete evidence is retained locally under outputs.
        public=dict(batch=batch,status='migration_accepted',recipe_sha256=sha(ROOT/'models/zipvoice/fp8/quantization.json'),
                    application_checks=application['status'],quality_summary=quality['summary'],
                    cases=migration['cases'],pcm_policy=migration['pcm_policy'],
                    private_evidence_hashes={name:sha(private_report(batch,name)) for name in ['003-application','004-fp32-reference','005-quality','006-migration']})
        write(ROOT/f'reports/sm120/zipvoice/fp8/b{batch}/history/001-migration-acceptance.json',public)
    write(ROOT/'reports/sm120/zipvoice/fp8/history/007-all-target-migration.json',dict(status='all_seven_migrations_accepted',batches=list(BATCHES)))
    print(json.dumps(dict(event='all_seven_migrations_accepted')),flush=True)


if __name__=='__main__':main()
