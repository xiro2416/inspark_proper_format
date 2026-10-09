"""Empty-cache/private-bundle validation against the retained target evidence."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import BATCHES,environment,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cache=args.cache_root.resolve()
    if not cache.is_relative_to('/workspace') or cache.exists():raise ValueError('Use a new empty workspace cache root')
    cache.mkdir(parents=True)
    selected=json.loads(args.selection.read_text());env=environment()
    env.update(HF_HUB_VERBOSITY='error',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME=str(cache/'huggingface'),HF_HUB_CACHE=str(cache/'huggingface/hub'))
    report=dict(status='running',selection_sha256=sha(args.selection),cache_root=str(cache),batches={})
    for batch in BATCHES:
        command=[str(ROOT/'.venv-zipvoice-fp8/bin/python'),'-m','inspark_infer.runtime.zipvoice_fp8_b128.worker',
                 'ensure','--precision','fp8','--batches',str(batch),'--gpu','3','--output-root',str(cache/'bundles')]
        download=subprocess.run(command,env=env,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if download.returncode:
            message=re.sub(r'(https?://[^\s"<>?]+)\?[^\s"<>]*',r'\1?[redacted]',download.stderr+download.stdout)
            raise RuntimeError(message)
        output=download.stdout
        downloaded=json.loads(output)['bundles'][0]
        proofs=[]
        for expected in selected['batches'][str(batch)]['fresh_cases']:
            if sha(ROOT/expected['condition'])!=expected['condition_sha256']:raise ValueError('Fresh validation input identity mismatch')
            destination=cache/f'inference/b{batch}'/str(len(proofs))
            workload=destination/'workload.json';write(workload,expected['workload'])
            cmd=[str(ROOT/'.venv-zipvoice-fp8/bin/python'),'-m','inspark_infer.runtime.zipvoice_fp8_b128.worker',
                 'infer','--precision','fp8','--batch',str(batch),'--gpu','3','--bundle',downloaded['bundle'],
                 '--inputs',str(ROOT/expected['condition']),'--workload',str(workload),'--output',str(destination),
                 '--seed',str(expected['seed']),'--functional-checks','--save-indices',*expected['wav_sha256'].keys()]
            if expected.get('text_reuse_disabled'):cmd.append('--disable-text-reuse')
            result=subprocess.run(cmd,env=env,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            safe_output=re.sub(r'(https?://[^\s"<>?]+)\?[^\s"<>]*',r'\1?[redacted]',result.stdout)
            destination.joinpath('download-run.log').write_text(safe_output)
            result.check_returncode()
            actual=json.loads((destination/'report.json').read_text())
            if not actual['graph_bitwise_guard'] or not actual['graph_wave_bitwise_guard']:raise ValueError('Downloaded graph guard failed')
            for row,digest in expected['wav_sha256'].items():
                if sha(destination/f'{int(row):04d}.wav')!=digest:raise ValueError('Downloaded complete PCM changed')
            proofs.append(dict(workload=expected['workload'],rows=list(expected['wav_sha256']),pcm_exact=True,
                               report_sha256=sha(destination/'report.json')))
        report['batches'][str(batch)]=dict(bundle_id=downloaded['bundle_id'],cases=proofs)
        write(args.output,report)
    reg=json.loads((ROOT/'configs/hardware/sm120/zipvoice_fp8_b128_registry.json').read_text())
    report.update(status='b128_fresh_fp8_bundle_validated',revision=reg['revision'])
    write(args.output,report);print(json.dumps(dict(status=report['status'],revision=report['revision'])))


if __name__=='__main__':main()
