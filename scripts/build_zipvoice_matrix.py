"""Serial, resumable A_1007 build matrix. No performance acceptance is implied."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,default=[1,2,4,8,16,32,64])
    parser.add_argument('--skip-application-checks',action='store_true')
    args=parser.parse_args()
    lockpath=Path('/workspace/.cache/inspark/gpu-locks/1.lock')
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    lock=lockpath.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
         'TRITON_CACHE_DIR':str(ROOT/'.cache/triton'),'CUDA_CACHE_PATH':str(ROOT/'.cache/cuda'),
         'XDG_CACHE_HOME':str(ROOT/'.cache'),'TORCH_HOME':str(ROOT/'.cache/torch'),
         'HF_HOME':str(ROOT/'.cache/huggingface'),'HF_ENDPOINT':'https://hf-mirror.com',
         'TMPDIR':str(ROOT/'.cache/tmp')}
    Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
    records=[]
    for batch in args.batches:
        for kind in ('fm-inherited','fm-native','vocos','text'):
            target=ROOT/f'artifacts/zipvoice/a1007/b{batch}/{kind}'
            if kind=='text': target=target.parent/'text'
            report=target/'build.json'
            if report.exists():
                state=json.loads(report.read_text())
                if state.get('status')=='built_unvalidated':
                    records.append({'batch':batch,'kind':kind,'status':'existing_built_unvalidated'})
                    continue
                raise RuntimeError(f'Inspect incomplete build before resuming: {report}')
            graphs=ROOT/f'.work/zipvoice/b{batch}/graphs'
            cmd=[str(ROOT/'.venv-builder/bin/python'),str(ROOT/'scripts/build_zipvoice_engine.py'),
                 '--batch',str(batch),'--gpu','1','--no-torch','--onnx',str(graphs/f'{kind}.onnx'),
                 '--profile-json',str(graphs/f'{kind}-profile.json'),'--output',str(target)]
            if kind=='fm-inherited': cmd+=['--inherited']
            # Only copied caches are writable; TRT checks version compatibility.
            cache=ROOT/f'models/zipvoice/timing/b{16 if batch<=16 else batch}-{kind}.cache'
            if cache.exists():cmd+=['--timing-cache',str(cache)]
            logfile=ROOT/f'outputs/build-logs/b{batch}-{kind}.log'
            logfile.parent.mkdir(parents=True,exist_ok=True)
            print(json.dumps({'event':'build_start','batch':batch,'kind':kind,'log':str(logfile)}),flush=True)
            start=time.monotonic()
            with logfile.open('w') as log:
                subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            records.append({'batch':batch,'kind':kind,'seconds':time.monotonic()-start,'status':'built_unvalidated'})
            (ROOT/'reports/sm89/zipvoice/a1007/build-matrix.json').write_text(json.dumps(records,indent=2)+'\n')
        validation=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history/002-minimal-compute.json'
        previous=json.loads(validation.read_text()) if validation.exists() else {}
        identities={route:json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/{route}/build.json').read_text())['engine_sha256'] for route in ('fm-native','fm-inherited')}
        if previous.get('status')!='minimal_compute_passed_audio_quality_pending' or previous.get('engine_sha256')!=identities:
            with (ROOT/f'outputs/build-logs/b{batch}-minimal-compute.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_compute.py'),'--batch',str(batch)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not args.skip_application_checks:
            # The validation child acquires the same lease itself.
            fcntl.flock(lock,fcntl.LOCK_UN)
            with (ROOT/f'outputs/build-logs/b{batch}-application.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/run_zipvoice_validation.py'),'--batches',str(batch)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('Build matrix complete; correctness/migration/optimization acceptance still pending',flush=True)

if __name__=='__main__':main()
