"""Full EOS, request replay, cancellation/cleanup and ordered PCM on an exact batch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int,choices=[1,2,4,8,16,32,64,128],required=True)
    p.add_argument('--deployment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wav-directory',type=Path,help='Independent lifecycle output archive')
    a=p.parse_args()
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.pool import Pool
    cfg=load(ROOT/'local_assets/runtime/runtime_fp32_b1.yaml');cfg['max_batch']=a.batch
    texts=['小林正在检查样本。','小周正在记录结果。','小王正在测试语音。','小陈正在确认行程。']
    references=['male_news.wav','mingxiang.wav','paimeng5s.wav','positive.wav']
    report=dict(batch=a.batch,deployment=str(a.deployment.resolve()),physical_gpu=1,passed=False)
    with GPULease(1),Pool(cfg,1,1) as pool:
        for i,name in enumerate(references):pool.prepare_reference(f'voice-{i}',str(Path('/workspace/A_TEST_REF')/name))
        report['deployment_attestation']=pool.prepare_deployment(load_deployment(a.deployment))[0]
        def wave(label, count=None):
            ids=[f'{label}-{i}' for i in range(a.batch if count is None else count)]
            for i,ident in enumerate(ids):
                pool.create_session(ident,f'voice-{i%4}',seed=8100+i,emotion=[0.]*8)
                pool.push_text(ident,texts[i%4]);pool.finish_input(ident)
            while any(s['ready_heads'] or s['ready_tails'] for s in pool.states.values()):pool.run_ready()
            result=[]
            for i,ident in enumerate(ids):
                data=pool.result(ident)
                if not data['complete'] or data['error'] or not data['chunks']:raise RuntimeError('Incomplete/failed EOS: '+ident)
                pcm=np.concatenate([c['pcm'] for c in data['chunks']])
                if not np.isfinite(pcm).all() or not np.any(pcm):raise RuntimeError('Invalid/silent PCM: '+ident)
                indices=[c['index'] for c in data['chunks']]
                if indices!=list(range(len(indices))):raise RuntimeError('Unordered chunks')
                output=(a.wav_directory or ROOT/'outputs/lifecycle'/f'b{a.batch}')/f'{label}-{i}.wav';output.parent.mkdir(parents=True,exist_ok=True)
                sf.write(output,pcm,22050,subtype='PCM_16')
                result.append(dict(row=i,voice=references[i%4],text=texts[i%4],chunks=len(indices),
                                   seconds=len(pcm)/22050,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                   wav=str(output),peak=float(np.max(np.abs(pcm)))))
                pool.release(ident)
            return result
        report['first']=wave('first')
        # A cancelled request must not contaminate the next request's RNG/KV/PCM.
        pool.create_session('cancelled','voice-0',seed=8100,emotion=[0.]*8)
        pool.push_text('cancelled','小林正在检查样本，请仔细核对记录，然后准备下一步的报告。')
        pool.finish_input('cancelled')
        events=pool.run_ready()
        if not events:raise RuntimeError('Cancellation test did not execute an active request')
        report['cancelled_after_pcm']=len(events)
        pool.cancel('cancelled')
        report['replay_after_cancel']=wave('replay')
        if [r['pcm_sha256'] for r in report['first']] != [r['pcm_sha256'] for r in report['replay_after_cancel']]:
            raise RuntimeError('Same exact batch/inputs/seeds changed after cancellation/replay')
        if a.batch > 1:
            report['partial_batch']=wave('partial',max(1,a.batch//2))
        report['stats']=pool.stats()[0]
        if report['stats']['sessions']!=0 or report['stats']['active_rows']!=0:raise RuntimeError('Leaked request state')
        report['passed']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(batch=a.batch,passed=report['passed'],output=str(a.output))),flush=True)


if __name__=='__main__':main()
