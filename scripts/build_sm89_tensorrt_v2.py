#!/usr/bin/env python3
"""Build static, whole-CFM and whole-BigVGAN TensorRT engines for SM89."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/common/runtime.yaml')
    parser.add_argument('--ref-audio',required=True)
    parser.add_argument('--batches',default='1,4,8,16')
    parser.add_argument('--output',default='artifacts/tensorrt_sm89_bf16_v2')
    args=parser.parse_args();batches=tuple(int(x) for x in args.batches.split(','))
    if not batches or any(x not in (1,4,8,16) for x in batches):raise ValueError('batches must be 1/4/8/16')
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one physical GPU')

    import tensorrt
    import torch_tensorrt
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine

    output=Path(args.output).resolve();staging=output.with_name(output.name+'.building')
    if output.exists() and not staging.exists():output.rename(staging)
    staging.mkdir(parents=True,exist_ok=True)
    timing_cache=staging/'timing.cache'
    config=load(args.config);config['max_batch']=max(batches);engine=Engine(config);entries=[]

    settings=dict(ir='dynamo',enabled_precisions={torch.float32,torch.bfloat16},
                  require_full_compilation=False,min_block_size=3,optimization_level=5,
                  workspace_size=8<<30,num_avg_timing_iters=5,timing_cache_path=str(timing_cache),
                  cache_built_engines=True,reuse_cached_engines=True,use_fast_partitioner=False)

    def compile_save(component,batch,module,inputs,frames):
        artifact=staging/f'{component}_b{batch}.ts'
        if artifact.exists():
            entry=dict(component=component,batch=batch,frames=frames,artifact=artifact.name,
                       sha256=sha256(artifact),compile_seconds=None,validation={'resumed':True})
            entries.append(entry);print(json.dumps(entry),flush=True);return
        started=time.perf_counter();compiled=torch_tensorrt.compile(module,inputs=list(inputs),**settings)
        torch.cuda.synchronize()
        with torch.inference_mode():
            expected=module(*inputs);actual=compiled(*inputs);torch.cuda.synchronize()
        validation=dict(max_abs=(expected-actual).abs().max().item(),
                        mean_abs=(expected-actual).abs().mean().item())
        torch_tensorrt.save(compiled,str(artifact),output_format='torchscript',inputs=list(inputs))
        entry=dict(component=component,batch=batch,frames=frames,artifact=artifact.name,
                   sha256=sha256(artifact),compile_seconds=time.perf_counter()-started,
                   validation=validation)
        entries.append(entry);print(json.dumps(entry),flush=True)
        del compiled,expected,actual;torch.cuda.empty_cache()

    try:
        engine.prepare_reference('reference',args.ref_audio)
        engine.prepare_precision('bf16',['target','draft','cfm','vocoder'],True)
        with torch.inference_mode():
            engine.student.model.requires_grad_(False);engine.tts.bigvgan.requires_grad_(False)
        voice=engine.model.bank.get('reference')['values'];plen=voice['voice.cache_mel'].shape[-1];frames=plen+52
        for batch in batches:
            mu=torch.cat((voice['voice.cache_s2mel_prompt'],voice['voice.cache_s2mel_prompt'].new_zeros(1,52,voice['voice.cache_s2mel_prompt'].shape[-1])),1).repeat(batch,1,1)
            x=mu.new_zeros(batch,80,frames);prompt=x.clone();prompt[:,:,:plen]=voice['voice.cache_mel']
            lengths=torch.full((batch,),frames,device=x.device,dtype=torch.long)
            cfm_inputs=(x,prompt,lengths,engine.student.times[0].expand(batch,-1),voice['voice.cache_s2mel_style'].repeat(batch,1),mu)
            compile_save('cfm',batch,engine.student.model,cfm_inputs,frames)
            vocoder_input=x[:,:,:52].clone()
            compile_save('vocoder',batch,engine.tts.bigvgan,(vocoder_input,),52)
        plan=dict(format_version=2,backend='Torch-TensorRT Dynamo',precision='bf16_fp32_interfaces',
                  sm=torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1],
                  gpu=torch.cuda.get_device_name(),max_batch=max(batches),batches=list(batches),
                  reference_prompt_frames=plen,cfm_frames=frames,torch_version=torch.__version__,
                  tensorrt_version=tensorrt.__version__,torch_tensorrt_version=torch_tensorrt.__version__,
                  tactic_search=dict(static_shapes=True,optimization_level=5,workspace_bytes=8<<30,
                                     num_avg_timing_iters=5,timing_cache=timing_cache.name),
                  engines=entries,fallbacks=[
                    {'scope':'AR Target/Draft and Slot KV','reason':'kept as the identical shared scheduler and BF16/cuBLAS baseline'},
                    {'scope':'non-selected capture batches','reason':'deployment loads the exact configured batch engine; other graph-capture shapes use eager only and are not measured'},
                  ],online_compilation=False,project_fusions_applied=False)
        (staging/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        if output.exists():shutil.rmtree(output)
        staging.rename(output);print(json.dumps({'output':str(output),'engines':len(entries)},ensure_ascii=False),flush=True)
    finally:engine.close()


if __name__=='__main__':run()
