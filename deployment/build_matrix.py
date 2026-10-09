"""Serial, resumable SM89 build of exact B1/B2/B4/B8/B16 component engines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--cfm-kind', choices=['full_solver','estimator'], default='estimator')
    parser.add_argument('--components', nargs='+', choices=['target','draft','prefill','latent','cfm','vocoder'], default=['target','draft','prefill','latent','cfm','vocoder'])
    parser.add_argument('--vocoder-representation', choices=['conv1d','conv2d','gemm'], default='gemm')
    args = parser.parse_args()
    if any(b not in (1, 2, 4, 8, 16, 32, 64, 128) for b in args.batches):
        parser.error('Batches must be drawn from 1/2/4/8/16/32')
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Source deployment/env.sh; only physical GPU1 is authorized')
    history = ROOT / 'deployment/history'; history.mkdir(exist_ok=True)
    for batch in args.batches:
        for component in args.components:
            component_directory='cfm-estimator' if component=='cfm' and args.cfm_kind=='estimator' else component
            if component=='vocoder' and args.vocoder_representation!='conv1d':component_directory='vocoder-'+args.vocoder_representation
            directory = ROOT / 'artifacts/sm89/int8_smoothquant' / f'b{batch}' / component_directory
            onnx, binary, plan = directory/'model.onnx', directory/'model.engine', directory/'model.plan.json'
            if plan.is_file() and binary.is_file():
                data = json.loads(plan.read_text())
                if (data['sha256']==digest(binary) and data['sm']==89
                        and data['gpu_name']=='NVIDIA GeForce RTX 4090' and data['batch']==batch
                        and data['optimization_level']==5 and data['tiling_optimization_level']=='full'
                        and data['max_num_tactics']==2147483646):
                    print(json.dumps(dict(stage='verified_existing_engine',batch=batch,component=component)),flush=True)
                    continue
                raise RuntimeError('Existing engine identity or build policy mismatch: '+str(plan))
            export = onnx.with_suffix('.export.json')
            if not export.is_file():
                command=[sys.executable,'deployment/export.py','--component',component,'--batch',str(batch)]
                if component=='cfm':command.extend(['--cfm-kind',args.cfm_kind])
                if component=='vocoder' and args.vocoder_representation!='conv1d':command.append('--vocoder-'+args.vocoder_representation)
                stage='export'
                log=ROOT/'.cache'/f'export-{component_directory}-b{batch}.log'
                print(json.dumps(dict(stage=stage,batch=batch,component=component,log=str(log))),flush=True)
                with log.open('w') as stream:
                    subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            record=json.loads(export.read_text())
            if record['onnx_sha256']!=digest(onnx):
                raise RuntimeError('ONNX identity mismatch: '+str(onnx))
            log=ROOT/'.cache'/f'build-{component_directory}-b{batch}.log'
            command=[sys.executable,'scripts/build_unified_onnx.py','--gpu','1','--onnx',str(onnx),
                     '--engine',str(binary),'--optimization-level','5','--tiling','full','--aux-streams','0']
            print(json.dumps(dict(stage='build',batch=batch,component=component,log=str(log))),flush=True)
            started=time.monotonic()
            with log.open('w') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print(json.dumps(dict(stage='built',batch=batch,component=component,seconds=time.monotonic()-started)),flush=True)
    print(json.dumps(dict(status='matrix_built_not_yet_validated',batches=args.batches)),flush=True)


if __name__=='__main__':
    main()
