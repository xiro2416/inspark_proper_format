"""Run minimal FM checks followed by inherited application correctness, serially."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def environment():
    return {**os.environ,'CUDA_VISIBLE_DEVICES':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
            'INSPARK_REPO_ROOT':str(ROOT),'PYTHONPATH':str(ROOT/'src'),
            'XDG_CACHE_HOME':str(ROOT/'.cache'),'TRITON_CACHE_DIR':str(ROOT/'.cache/triton'),
            'CUDA_CACHE_PATH':str(ROOT/'.cache/cuda'),'TORCH_HOME':str(ROOT/'.cache/torch'),
            'HF_HOME':str(ROOT/'.cache/huggingface'),'TMPDIR':str(ROOT/'.cache/tmp')}

def inventory(batch,route,workload):
    engines={}
    package=None
    for key,kind,b in [('fm',f'fm-{route}',batch),('text','text',batch),('unique','text',1),('vocos','vocos',batch)]:
        directory=ROOT/f'artifacts/zipvoice/a1007/b{b}/{kind}'
        info=json.loads((directory/'build.json').read_text())
        if key=='fm':
            from inspark_infer.build.zipvoice import plugin_package
            package=plugin_package(batch,info.get('runtime_plugin_package'))
            plugin_dir=ROOT/'src'/Path(*package.split('.'))
            plugins={str(p):sha(p) for p in sorted(plugin_dir.glob('*.py'))}
        assert info['status']=='built_unvalidated'
        path=directory/'engine.plan';assert sha(path)==info['engine_sha256']
        engines[key]={'path':str(path),'sha256':info['engine_sha256'],'shape_profile':info['effective_shape_profile'],
                      'source_sha256':info['source_sha256'],'plugin_sources':plugins if key=='fm' and info.get('plugin_sources') else {}}
    return {'status':'validation_inventory','physical_gpu':1,'gpu_uuid':'GPU-7dab7d6b-ac8c-7ccc-6410-916d3b7689b3',
            'compute_capability':[8,9],'tensorrt':'11.3.0.99','workload':{**workload,'batch':batch},'engines':engines,'plugin_package':package,
            'origin_mapping_source_sha256':sha(ROOT/f'.work/zipvoice/b{batch}/code/position_fm_rewrite.py')}

def invoke(batch,route,case,dest,extra=(),repetitions=1,warmup=1,functional=True,application=None):
    dest.mkdir(parents=True,exist_ok=True)
    inv=inventory(batch,route,case['workload'])
    manifest=dest/'engines.json';manifest.write_text(json.dumps(inv,indent=2)+'\n')
    selected=sorted({0,min(1,batch-1),batch-1})
    route_module=application or ('a1007_delivery' if batch==32 else 'a1007')
    assert route_module in ('a1007','a1007_delivery','a1007_graph','a1007_delivery_graph')
    cmd=[str(ROOT/'.venv-zipvoice/bin/python'),'-m',f'inspark_infer.runtime.zipvoice.routes.{route_module}',
         '--batch',str(batch),'--engine-manifest',str(manifest),'--inputs',case['condition'],
         '--gpu','1','--output',str(dest),'--shared-context-workspace','--arena-istft','--include-input-transfer',
         '--warmup',str(warmup),'--repetitions',str(repetitions),'--seed','9102','--save-indices',*map(str,selected),
         '--dump-selected-state',*extra]
    if functional:cmd+=['--functional-only']
    saved=dest/'report.json'
    command=dest/'command.json'
    if functional and saved.exists() and command.exists() and json.loads(command.read_text())==cmd:
        previous=json.loads(saved.read_text())
        runner=ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{route_module}.py'
        if (previous.get('runner_sha256')==sha(runner) and previous.get('input_sha256')==sha(case['condition'])
            and previous.get('engine_manifest_sha256')==sha(manifest) and previous.get('graph_bitwise_guard')
            and all((dest/f'{i:04d}.wav').is_file() for i in selected)
            and (dest/'selected-state.safetensors').is_file()):
            return previous,selected
    (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (dest/'run.log').open('w') as log:
        subprocess.run(cmd,env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads((dest/'report.json').read_text())
    assert report['graph_bitwise_guard'] and report['graph_state_relative_l2']==0
    assert all(x['pcm_items']==batch for x in report['results'])
    assert report['power_limits_before_w']==report['power_limits_after_w']=='400.00, 400.00'
    return report,selected

def contract_cases(batch,manifest):
    """Synthetic interface boundaries are never used for audio quality claims."""
    import torch
    from safetensors.torch import load_file,save_file
    base=manifest['primary_760']
    original=load_file(base['condition'])
    folder=ROOT/f'outputs/zipvoice-validation/b{batch}/contract-inputs'
    folder.mkdir(parents=True,exist_ok=True)
    result=[]
    for frames,length in ((600,52),(600,141),(760,78),(761,79),(920,52),(920,141)):
        tensors={k:v.clone() for k,v in original.items()}
        token=original['token_ids'][0,0].item()
        tensors['token_ids']=torch.full((1,length),token,dtype=torch.int64)
        tensors['token_ids'][0,-1]=original['token_ids'][0,-1]
        dest=folder/f'synthetic-t{frames}-l{length}.safetensors'
        save_file(tensors,str(dest))
        result.append({'name':dest.stem,'kind':'synthetic_interface_only','condition':str(dest),
                       'workload':{**base['workload'],'batch':batch,'total_frames':frames,'target_frames':frames-375,'padded_tokens':length,'joint_tokens':length-1}})
    groups={}
    for case in manifest['legal_cases']:
        groups.setdefault((case['reference_index'],case['total_frames'],case['padded_tokens']),[]).append(case)
    pair=next(cases[:2] for cases in groups.values() if len({x['text'] for x in cases})>=2)
    if batch>1:
        rows=[load_file(pair[i%2]['condition']) for i in range(batch)]
        tensors={k:torch.cat([x[k] for x in rows],dim=0) for k in rows[0]}
        dest=folder/'mixed-complete-texts.safetensors';save_file(tensors,str(dest))
        result.append({'name':'mixed-complete-texts','kind':'natural_same_length_mixed','condition':str(dest),
                       'workload':{**pair[0]['workload'],'batch':batch},'texts':[pair[i%2]['text'] for i in range(batch)]})
    (folder/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16,32,64])
    p.add_argument('--minimal-only',action='store_true')
    args=p.parse_args()
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    for batch in args.batches:
        board=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'
        proof=board/'002-minimal-compute.json'
        previous=json.loads(proof.read_text()) if proof.exists() else {}
        identities={route:json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/{route}/build.json').read_text())['engine_sha256'] for route in ('fm-native','fm-inherited')}
        if previous.get('status')!='minimal_compute_passed_audio_quality_pending' or previous.get('engine_sha256')!=identities:
            with (ROOT/f'outputs/build-logs/b{batch}-minimal-compute.log').open('w') as log:
                subprocess.run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_compute.py'),'--batch',str(batch)],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        if args.minimal_only:continue
        summary={'status':'running','batch':batch,'quality_inputs':[],'cases':[]}
        for route in ('native','inherited'):
            for index,case in enumerate(manifest['quality_cases']):
                dest=ROOT/f'outputs/zipvoice-validation/b{batch}/{route}/real-{index:03d}'
                report,selected=invoke(batch,route,case,dest)
                for row in selected:
                    path=dest/f'{row:04d}.wav'
                    summary['quality_inputs'].append({'batch':batch,'route':route,'case':index,'row':row,'path':str(path),'wav_sha256':sha(path),
                        'target_text':case['text'],'reference_wav':case['reference_wav'],'reference_sha256':case['reference_sha256']})
                summary['cases'].append({'route':route,'index':index,'frames':case['total_frames'],'report':str(dest/'report.json'),'graph_direct_exact':True,'pcm_items':batch})
                (board/'003-application.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(f'B{batch} {route} real{index} passed',flush=True)
        summary['contract_cases']=[]
        from safetensors.torch import load_file
        import torch
        summary['matched_state_checks']=[]
        for index,case in enumerate(manifest['quality_cases']):
            native=load_file(str(ROOT/f'outputs/zipvoice-validation/b{batch}/native/real-{index:03d}/selected-state.safetensors'))
            inherited=load_file(str(ROOT/f'outputs/zipvoice-validation/b{batch}/inherited/real-{index:03d}/selected-state.safetensors'))
            for key in ('initial_state','text_condition','speech_condition','padding_mask','time_grid'):
                assert torch.equal(native[key],inherited[key]),(batch,index,key,'mapping mismatch')
            if batch>1:assert not torch.equal(native['initial_state'][0],native['initial_state'][-1])
            expected=native['final_state'];actual=inherited['final_state']
            summary['matched_state_checks'].append({'index':index,'original_conditions_noise_mask_grid_exact':True,
                'selected_state_relative_l2':float((actual-expected).norm()/expected.norm().clamp_min(1e-20))})
        for case in contract_cases(batch,manifest):
            dest=ROOT/f'outputs/zipvoice-validation/b{batch}/inherited'/case['name']
            report,_=invoke(batch,'inherited',case,dest)
            if case['kind']=='natural_same_length_mixed':assert not report['text_reuse']
            summary['contract_cases'].append({'kind':case['kind'],'name':case['name'],'report':str(dest/'report.json'),'passed':True})
            (board/'003-application.json').write_text(json.dumps(summary,indent=2)+'\n')
        fulltext=ROOT/f'outputs/zipvoice-validation/b{batch}/inherited/fulltext-control'
        report,_=invoke(batch,'inherited',manifest['primary_760'],fulltext,extra=['--disable-text-reuse'])
        assert not report['text_reuse']
        summary['full_text_control']=str(fulltext/'report.json')
        summary['status']='application_real_boundary_mixed_passed_quality_performance_pending'
        (board/'003-application.json').write_text(json.dumps(summary,indent=2)+'\n')
        (ROOT/f'outputs/zipvoice-validation/b{batch}/quality-inputs.json').write_text(json.dumps(summary['quality_inputs'],indent=2)+'\n')
        quality=board/'004-quality.json'
        inputs=ROOT/f'outputs/zipvoice-validation/b{batch}/quality-inputs.json'
        if not quality.exists() or json.loads(quality.read_text()).get('input_inventory_sha256')!=sha(inputs):
            cmd=[str(ROOT/'.venv-evaluation/bin/python'),str(ROOT/'scripts/evaluate_zipvoice_audio.py'),
                 '--device','cpu','--inputs',str(inputs),'--output',str(quality)]
            job_record=board/'004-quality-job.json'
            if job_record.exists():
                previous=json.loads(job_record.read_text())
                active=subprocess.run(['ps','-p',str(previous['pid']),'-o','args='],text=True,capture_output=True)
                if active.returncode==0 and str(inputs) in active.stdout and 'evaluate_zipvoice_audio.py' in active.stdout:
                    if previous.get('input_inventory_sha256',sha(inputs))!=sha(inputs):
                        raise RuntimeError('A quality job for different inputs is still active; wait for it before replacing its output')
                    print(f'B{batch} CPU quality evaluation already active: {previous["pid"]}',flush=True)
                    continue
            with (ROOT/f'outputs/build-logs/b{batch}-quality-cpu.log').open('w') as log:
                job=subprocess.Popen(cmd,cwd=ROOT,env={**environment(),'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job_record.write_text(json.dumps({'pid':job.pid,'command':cmd,'input_inventory_sha256':sha(inputs),'device':'cpu','status':'started_not_accepted'},indent=2)+'\n')
            print(f'B{batch} CPU quality evaluation launched: {job.pid}; GPU remains free for the next build',flush=True)

if __name__=='__main__':main()
