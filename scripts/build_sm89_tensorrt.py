#!/usr/bin/env python3
"""Offline-build the fair BF16 TensorRT comparison engines on one visible GPU."""
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
    parser.add_argument('--max-batch',type=int,default=16)
    parser.add_argument('--output',default='artifacts/tensorrt_sm89_bf16')
    parser.add_argument('--rebuild-target',action='store_true')
    args=parser.parse_args()
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one physical GPU')
    if args.max_batch not in (1,4,8,16):raise ValueError('Validated comparison batches are 1/4/8/16')

    import tensorrt
    import torch_tensorrt
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine

    output=Path(args.output).resolve();staging=output.with_name(output.name+'.building')
    # A failed build keeps hashable completed engines and resumes explicitly.
    if output.exists() and not staging.exists():output.rename(staging)
    staging.mkdir(parents=True,exist_ok=True)
    config=load(args.config);config['max_batch']=args.max_batch
    engine=Engine(config);entries=[]

    if args.rebuild_target:
        for artifact in staging.glob('target_blocks_*_mlp.ts'):artifact.unlink()

    def compile_one(path,module,shape,opt_shape,operators):
        artifact=staging/(path.replace('.','_')+'.ts')
        profile={'min':list(shape[0]),'opt':list(opt_shape),'max':list(shape[1])}
        if artifact.exists():
            entry=dict(path=path,artifact=artifact.name,sha256=sha256(artifact),profile=profile,
                       operators=operators,compile_seconds=None,validation={'resumed':True})
            entries.append(entry);print(json.dumps(entry),flush=True);return
        spec=torch_tensorrt.Input(min_shape=shape[0],opt_shape=opt_shape,max_shape=shape[1],dtype=torch.float32)
        started=time.perf_counter()
        compiled=torch_tensorrt.compile(module.eval(),ir='dynamo',inputs=[spec],
            enabled_precisions={torch.float32,torch.bfloat16},require_full_compilation=True,min_block_size=1,
            cache_built_engines=True,reuse_cached_engines=True)
        example=torch.zeros(opt_shape,device='cuda',dtype=torch.float32)
        reference=module(example);actual=compiled(example);torch.cuda.synchronize()
        max_abs=(reference-actual).abs().max().item();mean_abs=(reference-actual).abs().mean().item()
        torch_tensorrt.save(compiled,str(artifact),output_format='torchscript',inputs=[example])
        entry=dict(path=path,artifact=artifact.name,sha256=sha256(artifact),
                   profile=profile,
                   operators=operators,compile_seconds=time.perf_counter()-started,
                   validation={'max_abs':max_abs,'mean_abs':mean_abs})
        entries.append(entry);print(json.dumps(entry),flush=True)

    try:
        # This is the common precision contract only. No CFM Triton fusion,
        # acoustic alias kernel or hand-written GEMM is installed here.
        engine.prepare_precision('bf16',['target','draft','cfm','vocoder'],True)
        for index,block in enumerate(engine.tts.gpt.gpt.h):
            compile_one(f'target.blocks.{index}.mlp',block.mlp,
                        ((1,8,1280),(args.max_batch,256,1280)),(min(8,args.max_batch),64,1280),
                        ['linear_bf16','gelu','linear_bf16'])
        for index,block in enumerate(engine.rt.engine.draft.layers):
            compile_one(f'draft.blocks.{index}.mlp',block.mlp,
                        ((1,7,1280),(args.max_batch,7,1280)),(min(8,args.max_batch),7,1280),
                        ['linear_bf16','gelu_tanh','linear_bf16'])
        for index,block in enumerate(engine.student.model.transformer.layers):
            compile_one(f'cfm.blocks.{index}.feed_forward',block.feed_forward,
                        ((1,32,512),(args.max_batch,512,512)),(min(8,args.max_batch),128,512),
                        ['linear_bf16','silu','linear_bf16','multiply','linear_bf16'])
        plan=dict(format_version=1,backend='Torch-TensorRT Dynamo',precision='bf16_fp32_interfaces',
                  sm=torch.cuda.get_device_capability()[0]*10+torch.cuda.get_device_capability()[1],
                  gpu=torch.cuda.get_device_name(),max_batch=args.max_batch,
                  torch_version=torch.__version__,tensorrt_version=tensorrt.__version__,
                  torch_tensorrt_version=torch_tensorrt.__version__,engines=entries,
                  fallbacks=[
                    {'scope':'target/draft attention and KV','reason':'stateful slot KV mutation belongs to shared scheduler'},
                    {'scope':'single projection GEMMs','reason':'isolated TensorRT engine overhead loses to common cuBLAS path'},
                    {'scope':'vocoder convolutions','reason':'kept for a separately validated whole-vocoder engine; no silent partition fallback'},
                  ],online_compilation=False,project_fusions_applied=False)
        (staging/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False))
        if output.exists():shutil.rmtree(output)
        staging.rename(output)
        print(json.dumps({'output':str(output),'engines':len(entries)},ensure_ascii=False),flush=True)
    finally:engine.close()


if __name__=='__main__':run()
