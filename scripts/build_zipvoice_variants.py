"""Build prepared, isolated source geometry candidates serially on GPU1."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from run_zipvoice_validation import ROOT,environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,choices=(1,2,4,8,16,32,64),default=[1,2,4])
    parser.add_argument('--suffix',default='geo1')
    args=parser.parse_args()
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment()
    for batch in args.batches:
        prepared=ROOT/f'.work/zipvoice/b{batch}_{args.suffix}/preparation.json'
        preparation=json.loads(prepared.read_text())
        target=ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{args.suffix}'
        report=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history/016-{args.suffix}-minimal.json'
        built=target/'build.json'
        if built.exists():
            assert json.loads(built.read_text())['status']=='built_unvalidated'
        else:
            command=[str(ROOT/'.venv-builder/bin/python'),str(ROOT/'scripts/build_zipvoice_engine.py'),
                     '--batch',str(batch),'--gpu','1','--no-torch','--inherited',
                     '--onnx',str(ROOT/preparation['source_graph']),
                     '--plugin-code',str(ROOT/preparation['code']),
                     '--plugin-package',preparation['plugin_package'],
                     '--profile-json',str(ROOT/f'.work/zipvoice/b{batch}/graphs/fm-inherited-profile.json'),
                     '--timing-cache',str(ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-inherited/timing.cache'),
                     '--output',str(target)]
            if preparation.get('omit_inherited'):command+=['--omit-inherited',*preparation['omit_inherited']]
            print(json.dumps({'event':'variant_build_start','batch':batch,'suffix':args.suffix}),flush=True)
            with (ROOT/f'outputs/build-logs/b{batch}-{args.suffix}-build.log').open('w') as log:
                subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (ROOT/f'outputs/build-logs/b{batch}-{args.suffix}-minimal.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_compute.py'),
                            '--batch',str(batch),'--candidate',str(target),'--output',str(report)],
                           env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads(report.read_text())['status']=='minimal_compute_passed_audio_quality_pending'
        print(json.dumps({'event':'variant_built_minimal_passed','batch':batch,'suffix':args.suffix}),flush=True)


if __name__=='__main__':main()
