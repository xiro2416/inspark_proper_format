"""WAV convenience mode and explicit NDJSON streaming-input protocol."""
import argparse,json,sys,time,base64
from pathlib import Path
from .config import load
from .runtime.device import GPULease
from .runtime.pool import Pool
from .streaming.splitter import split
from .runtime.graph_policy import BATCHES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/runtime.yaml');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--batch',type=int)
    p.add_argument('--deployment',help='Explicit offline deployment JSON; prepare before admitting requests')
    p.add_argument('--ref-audio',required=True);p.add_argument('--voice-id',default='reference')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--text');mode.add_argument('--stdin-stream',action='store_true')
    p.add_argument('--output');p.add_argument('--seed',type=int,default=0);p.add_argument('--emotion',type=float,nargs=8,default=[0.]*8)
    a=p.parse_args();cfg=load(a.config)
    if a.batch is not None and a.batch not in BATCHES:p.error('--batch must be one of '+str(BATCHES))
    if a.batch is not None:cfg['max_batch']=a.batch
    def emit(e):
        ch=e['chunk'];pcm=ch['pcm'].astype('<i2',copy=False).tobytes()
        print(json.dumps(dict(type='audio',id=e['request_id'],index=ch['index'],sample_rate=22050,sample_start=ch['sample_start'],sample_end=ch['sample_end'],pcm_s16le=base64.b64encode(pcm).decode(),complete=e['complete']),ensure_ascii=False),flush=True)
    with GPULease(a.gpu),Pool(cfg,a.gpu,a.workers) as pool:
        pool.prepare_reference(a.voice_id,a.ref_audio)
        if a.deployment:
            from .runtime.deployment import load as load_deployment
            manifest=pool.prepare_deployment(load_deployment(a.deployment))
            print(json.dumps(dict(deployment=manifest[0]['requested']['status'],precision=manifest[0]['resolved_precision'],online_learning=False)),file=sys.stderr)
        if a.text:
            parts=split(a.text)
            if not parts:raise ValueError('No spoken text')
            started=time.perf_counter();pool.create_session('request',a.voice_id,a.seed,a.emotion,started)
            pool.push_text('request',parts[0]);pool.run_ready(on_chunk=None if a.output else emit)
            pool.push_text('request',a.text[len(parts[0]):]);pool.finish_input('request')
            while any(s['ready_heads'] or s['ready_tails'] for s in pool.states.values()):pool.run_ready(on_chunk=None if a.output else emit)
            result=pool.result('request');assert result['complete']
            if a.output:
                import numpy as np,soundfile as sf
                path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
                sf.write(path,np.concatenate([c['pcm'] for c in result['chunks']]),22050,subtype='PCM_16')
                print(json.dumps(dict(output=str(path),first_ms=(result['chunks'][0]['ready']-started)*1000,total_ms=(result['chunks'][-1]['ready']-started)*1000)))
            pool.release('request')
        else:
            if a.output:raise ValueError('--output is only for --text mode')
            for line in sys.stdin:
                event=json.loads(line);op=event['op'];ident=event.get('id')
                if op=='open':pool.create_session(ident,a.voice_id,event.get('seed',a.seed),event.get('emotion',a.emotion),time.perf_counter())
                elif op=='text':pool.push_text(ident,event['text'])
                elif op=='end':pool.finish_input(ident)
                elif op=='cancel':pool.cancel(ident)
                elif op=='release':pool.release(ident)
                elif op=='run':pool.run_ready(on_chunk=emit)
                elif op=='tick':pool.tick(on_chunk=emit)
                elif op=='drain':
                    while any(s['ready_heads'] or s['ready_tails'] for s in pool.states.values()):pool.run_ready(on_chunk=emit)
                else:raise ValueError('Unknown op: '+str(op))
            if pool.owners:raise RuntimeError('Input ended with unreleased sessions; end, drain, then release each request')

if __name__=='__main__':main()
