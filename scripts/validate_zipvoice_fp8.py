"""Actual FP8 route checks: dynamic bounds, mixed rows, captures and matched FP32 audits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.build.zipvoice import gpu_info
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,environment,profiles,sha,workload,write,private_report


def inventory(batch,w,variant='native'):
    engines={}
    for key,kind,b in [('fm','fm',batch),('text','text',batch),('unique','text',1),('vocos','vocos',batch)]:
        actual_kind=kind if key!='fm' or variant=='native' else kind+'-'+variant
        folder=ROOT/f'artifacts/zipvoice/sm120/fp8/b{b}/{actual_kind}'
        info=json.loads((folder/'build.json').read_text())
        if info['status']!='built_unvalidated' or sha(folder/'engine.plan')!=info['engine_sha256']:
            raise ValueError('Unverified build identity: '+str(folder))
        engines[key]=dict(path=str(folder/'engine.plan'),sha256=info['engine_sha256'],
                          shape_profile={n:[v[k] for k in ('min','opt','max')] for n,v in profiles(b)[kind].items()},
                          source_sha256=info['source_sha256'])
    g=gpu_info(3)
    result=dict(physical_gpu=3,gpu_uuid=g['uuid'],compute_capability=[12,0],tensorrt='11.3.0.99',
                power_limit_w=600,workload={**w,'batch':batch},engines=engines,
                component_precisions=dict(fm='first4_fp32_last12_eligible_fp8',text='fp32',vocos='fp32',istft='fp32'),
                origin_mapping_source_sha256=sha(ROOT/'src/inspark_infer/models/zipvoice/packed.py'))
    if variant in ['attention','attentiongeo']:
        prefix=f'inspark_infer.ops.tensorrt.zipvoice_fp8.b{batch}.'+('geo.' if variant=='attentiongeo' else '')
        result['plugin_packages']=[prefix+p for p in ['normal_tf32_plugin','online_nonlinear_rna_plugin']]
        result['application']='attention_route'
    return result


def invoke(batch,case,output,execution='model-graph',repetitions=1,functional=True,
           chunk=None,workers=4,minimum_seconds=0,full_text=False,seed=9102,variant='native'):
    inv=output/'inventory.json';write(inv,inventory(batch,case['workload'],variant))
    runner='route' if variant=='native' else 'attention_route'
    cached_path=output/'report.json'
    if functional and cached_path.is_file() and (output/'selected-state.safetensors').is_file():
        cached=json.loads(cached_path.read_text())
        if (cached.get('status')=='complete_functional_only' and cached.get('runner_sha256')==sha(ROOT/f'src/inspark_infer/runtime/zipvoice_fp8/{runner}.py')
            and cached.get('input_sha256')==sha(case['condition']) and cached.get('engine_manifest_sha256')==sha(inv)
            and cached.get('execution')==execution and cached.get('seed')==seed
            and cached.get('text_reuse_disabled')==full_text and cached.get('repetitions')==repetitions
            and cached.get('pcm_chunk')==(chunk or {1:16,2:16,4:1,8:1,16:4,32:6,64:16}[batch])
            and cached.get('pcm_workers')==workers):
            return cached
    command=[str(ROOT/'.venv-zipvoice-fp8/bin/python'),'-m','inspark_infer.runtime.zipvoice_fp8.'+runner,
             '--batch',str(batch),'--engine-manifest',str(inv),'--inputs',str(case['condition']),
             '--gpu','3','--output',str(output),'--shared-context-workspace','--arena-istft',
             '--include-input-transfer','--execution',execution,'--seed',str(seed),
             '--warmup','2','--repetitions',str(repetitions),'--save-indices',
             *map(str,sorted({0,min(1,batch-1),batch-1})),'--dump-selected-state',
             '--pcm-chunk',str(chunk or {1:16,2:16,4:1,8:1,16:4,32:6,64:16}[batch]),
             '--pcm-workers',str(workers),'--minimum-seconds',str(minimum_seconds)]
    if functional:command+=['--functional-only']
    if full_text:command+=['--disable-text-reuse']
    write(output/'command.json',command)
    with (output/'run.log').open('w') as log:
        subprocess.run(command,env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((output/'report.json').read_text())
    assert all(x['pcm_items']==batch for x in report['results'])
    assert report['power_limits_before_w']==report['power_limits_after_w']=='600.00, 600.00'
    if execution=='model-graph':assert report['graph_bitwise_guard'] and report['graph_wave_bitwise_guard']
    return report


def boundaries(batch,base):
    import torch
    from safetensors.torch import load_file,save_file
    original=load_file(base['condition']);result=[]
    for frames,tokens in [(600,52),(600,141),(760,78),(920,52),(920,141)]:
        data={k:v.clone() for k,v in original.items()}
        data['token_ids']=torch.full((1,tokens),int(original['token_ids'][0,0]),dtype=torch.int64)
        data['token_ids'][0,-1]=original['token_ids'][0,-1]
        folder=ROOT/f'outputs/fp8/b{batch}/conditions';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'boundary-{frames}-{tokens}.safetensors';save_file(data,str(path))
        result.append(dict(condition=str(path),workload=workload(batch,frames,tokens),kind='synthetic_interface_only'))
    if batch>1:
        data={k:v.expand(batch,*v.shape[1:]).clone() for k,v in original.items()}
        # Same-length heterogeneous tokens exercise the actual full-batch encoder.
        data['token_ids'][1,:-1]=torch.flip(data['token_ids'][1,:-1],[0])
        path=folder/'mixed-interface.safetensors';save_file(data,str(path))
        result.append(dict(condition=str(path),workload={**base['workload'],'batch':batch},kind='synthetic_mixed_interface_only'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=list(BATCHES))
    p.add_argument('--limit-quality',type=int)
    p.add_argument('--skip-boundaries',action='store_true')
    args=p.parse_args()
    from inspark_infer.runtime.device import GPULease
    manifest=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text())
    quality=manifest['quality'][:args.limit_quality] if args.limit_quality else manifest['quality']
    with GPULease(3):
        for batch in args.batches:
            summary=dict(batch=batch,status='running',quality=[],interface=[],quality_metrics_pending=True)
            board=ROOT/f'reports/sm120/zipvoice/fp8/b{batch}'
            for i,case in enumerate(quality):
                dest=ROOT/f'outputs/fp8/b{batch}/quality/{i:03d}'
                invoke(batch,case,dest)
                summary['quality'].append(dict(case=i,language=case['language'],frames=case['total_frames'],
                    reference_wav=case['reference_wav'],reference_sha256=case['reference_sha256'],text=case['text'],
                    condition=case['condition'],condition_sha256=case['condition_sha256'],
                    rows={str(r):dict(path=str(dest/f'{r:04d}.wav'),sha256=sha(dest/f'{r:04d}.wav')) for r in sorted({0,min(1,batch-1),batch-1})},
                    report=str(dest/'report.json'),report_sha256=sha(dest/'report.json')))
                write(private_report(batch,'003-application'),summary)
                print(json.dumps({'event':'quality_route_pass','batch':batch,'case':i}),flush=True)
            if not args.skip_boundaries:
                for i,case in enumerate(boundaries(batch,quality[0])):
                    dest=ROOT/f'outputs/fp8/b{batch}/interface/{i:03d}'
                    report=invoke(batch,case,dest)
                    if case['kind'].startswith('synthetic_mixed'):assert not report['text_reuse']
                    summary['interface'].append(dict(kind=case['kind'],workload=case['workload'],report=str(dest/'report.json')))
            first=ROOT/f'outputs/fp8/b{batch}/controls/direct'
            invoke(batch,quality[0],first,execution='direct')
            full=ROOT/f'outputs/fp8/b{batch}/controls/full-text'
            invoke(batch,quality[0],full,full_text=True)
            from safetensors.torch import load_file
            import torch
            captured=load_file(str(ROOT/f'outputs/fp8/b{batch}/quality/000/selected-state.safetensors'))
            direct=load_file(str(first/'selected-state.safetensors'))
            assert all(torch.equal(captured[k],direct[k]) for k in captured)
            summary.update(status='application_and_boundaries_passed_quality_audit_pending',
                           direct_graph_all_selected_state_equal=True,full_text_control=str(full/'report.json'))
            write(private_report(batch,'003-application'),summary)


if __name__=='__main__':main()
