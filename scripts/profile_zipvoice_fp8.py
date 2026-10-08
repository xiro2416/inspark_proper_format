"""Baseline compute review via CUDA Graph node tracing; diagnostics are not E2E."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,environment,write
from validate_zipvoice_fp8 import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=int,choices=BATCHES,required=True)
    p.add_argument('--variant',choices=['native','attention','attentiongeo'],default='native')
    args=p.parse_args()
    data=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text());case=data['primary_760']
    suffix='' if args.variant=='native' else '-'+args.variant
    out=ROOT/f'outputs/fp8/b{args.batch}/profile{suffix}';out.mkdir(parents=True,exist_ok=True)
    inv=out/'inventory.json';write(inv,inventory(args.batch,case['workload'],args.variant))
    command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
             '--cuda-graph-trace=node','--capture-range=nvtx','--nvtx-capture=zvoice_measure@*',
             '--capture-range-end=stop','--force-overwrite=true','--output='+str(out/'baseline'),
             str(ROOT/'.venv-zipvoice-fp8/bin/python'),'-m','inspark_infer.runtime.zipvoice_fp8.'+('route' if args.variant=='native' else 'attention_route'),
             '--batch',str(args.batch),'--gpu','3','--engine-manifest',str(inv),'--inputs',case['condition'],
             '--output',str(out/'run'),'--shared-context-workspace','--arena-istft','--include-input-transfer',
             '--disable-text-reuse','--nvtx-range','--warmup','2','--repetitions','3','--save-indices','0']
    from inspark_infer.runtime.device import GPULease
    with GPULease(3):
        env=environment();env['NSYS_NVTX_PROFILER_REGISTER_ONLY']='0'
        with (out/'nsys.log').open('w') as f:subprocess.run(command,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (out/'kernel-summary.csv').open('w') as f:
        subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(out/'baseline.nsys-rep')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'profile-scope.json',dict(status='profile_complete',batch=args.batch,
          scope='Three prepared760/full-text requests, CUDA Graph node trace; instrumented diagnostic timing, not production E2E'))
    print(json.dumps(dict(event='baseline_profile_complete',batch=args.batch)))


if __name__=='__main__':main()
