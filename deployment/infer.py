"""Select a verified local deployment, then retain the existing WAV/NDJSON API."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--precision',choices=['fp32','int8'],default='int8')
    p.add_argument('--batch',type=int,choices=[1,2,4,8,16,32,64,128],default=1)
    p.add_argument('--gpu',type=int,choices=[1],default=1)
    a,rest=p.parse_known_args()
    if '--help' in rest or '-h' in rest:
        print('Local defaults: INT8, physical GPU1, batch1. --batch {1,2,4,8,16,32,64,128}; --precision {int8,fp32}.')
    if a.precision=='fp32' or '--help' in rest or '-h' in rest:
        deployment=ROOT/'configs/current/reference.json'
    else:
        deployment=ROOT/'configs/hardware/sm89/indextts'/f'int8_b{a.batch}_selected.json'
        if not deployment.is_file():
            raise RuntimeError(f'INT8 B{a.batch} has not passed local deployment validation')
        plan=json.loads(deployment.read_text())
        if not plan['status'].startswith('validated_local_sm89'):
            raise RuntimeError('Candidate deployment has not passed validation')
    sys.argv=['inspark-local','--config',str(ROOT/'local_assets/runtime/runtime_fp32_b1.yaml'),
              '--deployment',str(deployment),'--gpu','1','--batch',str(a.batch),*rest]
    from inspark_infer.api.cli import main as infer
    infer()


if __name__=='__main__':main()
