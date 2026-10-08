"""Fresh private download, weight integrity and public-worker round trip for all batches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_zipvoice_validation import ROOT, environment, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-runs',type=Path,required=True,help='Accepted cases and selected PCM SHA256, keyed by batch')
    p.add_argument('--output',type=Path,required=True,help='New empty directory, including fresh download cache')
    args=p.parse_args()
    output=args.output.resolve();output.relative_to(ROOT)
    assert not output.exists(), 'Use a new empty location to prove download independence'
    expected=json.loads(args.expected_runs.read_text())
    assert set(expected['batches'])=={'1','2','4','8','16','32','64'}
    registry=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
    revision=registry['revision'];assert len(revision)==40
    output.mkdir(parents=True)
    env=environment();env['HF_HUB_CACHE']=str(output/'hub-cache')
    env['HF_ENDPOINT']='https://hf-mirror.com'
    env['HF_HUB_DISABLE_XET']='1'
    # Shared context was explicitly identified and retained. Do not claim ownership.
    env['ACC_GPU_ALLOW_SHARED']='1'
    sys.path.insert(0,str(ROOT/'src'))
    from inspark_infer.build.zipvoice import ensure
    os.environ.update(env)
    result={'status':'running','revision':revision,'batches':{},'expected_runs_sha256':sha(args.expected_runs)}
    report=output/'validation.json'
    def save():report.write_text(json.dumps(result,indent=2)+'\n')
    save()
    subprocess.run([sys.executable,str(ROOT/'scripts/download_zipvoice_weights.py'),
        '--output',str(output/'weights'),'--cache-dir',str(output/'hub-cache')],env=env,cwd=ROOT,check=True)
    result['weights_manifest_sha256']=sha(output/'weights/weights-manifest.json');save()
    for key,cases in expected['batches'].items():
        batch=int(key)
        assert {case['workload']['total_frames'] for case in cases}>={600,760,920}
        assert any(case.get('mixed_rows') for case in cases) or batch==1
        bundle,m=ensure(batch,output_root=output/'bundles')
        assert (bundle/'fetch-report.json').is_file(),'Fresh download was bypassed'
        assert json.loads((bundle/'fetch-report.json').read_text())['revision']==revision
        runs=[]
        for index,case in enumerate(cases):
            import time
            for attempt in range(10):
                row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True)
                memory,util,cap,enforced=map(float,row.split(','))
                assert memory<=4352 and cap==enforced==400, 'GPU1 preflight changed; preserve external workload'
                if util<=10:break
                time.sleep(1)
            else:raise RuntimeError('GPU1 has ongoing activity; preserve external workload')
            dest=output/f'b{batch}/case{index}';dest.mkdir(parents=True)
            workload=dest/'workload.json';workload.write_text(json.dumps(case['workload']))
            command=[sys.executable,'-m','inspark_infer.runtime.zipvoice.worker','infer',
                '--batch',key,'--gpu','1','--bundle',str(bundle),'--inputs',case['condition'],
                '--workload',str(workload),'--output',str(dest),'--seed',str(case['seed']),
                '--functional-checks','--save-indices',*case['pcm_sha256'].keys()]
            with (dest/'worker.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            proof=json.loads((dest/'report.json').read_text())
            assert proof['graph_bitwise_guard'] and all(x['pcm_items']==batch for x in proof['results'])
            assert proof['runner_sha256']==m['runner_sha256']
            assert proof['power_limits_before_w']==proof['power_limits_after_w']=='400.00, 400.00'
            assert proof['pcm_chunk']==m['pcm_policy']['chunk'] and proof['pcm_workers']==m['pcm_policy']['workers']
            if m['application'].endswith('_graph'):
                assert proof['graph_wave_bitwise_guard']
                assert proof['capture_domain']=='text,8FM+Euler,mel,Vocos,CENTER_ISTFT,RMS; fresh H2D/noise and ordered PCM outside'
            if case.get('mixed_rows'):assert not proof['text_reuse']
            assert proof['bundle_id']==m['bundle_id']
            for row,digest in case['pcm_sha256'].items():
                assert sha(dest/f'{int(row):04d}.wav')==digest,(batch,index,row,'Fresh public worker PCM differs')
            runs.append({'frames':case['workload']['total_frames'],'mixed_rows':case.get('mixed_rows',False),'all_rows_pcm_count':batch,'selected_pcm_exact':True,'report':str(dest/'report.json')})
        result['batches'][key]={'bundle_id':m['bundle_id'],'runs':runs};save()
    result['status']='all_batches_and_weights_fresh_download_validated';save()
    print(json.dumps({'status':result['status'],'revision':revision,'report':str(report)}))


if __name__=='__main__':main()
