#!/usr/bin/env python3
"""Numerical, latency, and CUDA Graph validation for native TRT11.3 CFM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,default=6)
    parser.add_argument("--batch",type=int,default=8);parser.add_argument("--repeats",type=int,default=50)
    parser.add_argument("--config",default="configs/runtime.yaml")
    parser.add_argument("--reference",default="/workspace/index-tts/data/audio/old/mingxiang_gao.wav")
    parser.add_argument("--plan",default="artifacts/trt113_cfm/plan_b8.json")
    parser.add_argument("--json-out",default="outputs/trt113_cfm_b8_validation.json");args=parser.parse_args()
    from acc_infer_clear.runtime.device import select_gpu
    select_gpu(args.gpu)
    import torch
    from acc_infer_clear.config import load
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.runtime.graphs import capture
    from acc_infer_clear.tensorrt_backend.native113 import NativeCFMSolver113

    config=load(args.config);config["max_batch"]=args.batch;engine=Engine(config)
    try:
        engine.prepare_reference("reference",args.reference)
        engine.prepare_precision("bf16",["target","draft","cfm","vocoder"],True)
        eager=engine.student;voice=engine.model.bank.get("reference")["values"]
        prompt_frames=voice["voice.cache_mel"].shape[-1];frames=prompt_frames+52
        mu=torch.cat((voice["voice.cache_s2mel_prompt"],voice["voice.cache_s2mel_prompt"].new_zeros(1,52,512)),1).repeat(args.batch,1,1)
        x=mu.new_zeros(args.batch,80,frames);prompt=x.clone();prompt[:,:,:prompt_frames]=voice["voice.cache_mel"]
        lengths=torch.full((args.batch,),frames,device=x.device,dtype=torch.long)
        style=voice["voice.cache_s2mel_style"].repeat(args.batch,1)
        mask=(torch.arange(frames,device=x.device)[None,None]<prompt_frames).expand(args.batch,1,-1).clone()
        inputs=(x,prompt,lengths,style,mu,mask)
        native=NativeCFMSolver113(args.plan,eager)
        with torch.inference_mode():
            expected=eager(*inputs).clone();actual=native(*inputs).clone();torch.cuda.synchronize()
            graph=capture(native,inputs);graph_actual=graph(*inputs).clone();torch.cuda.synchronize()
            def timing(fn):
                for _ in range(5):fn(*inputs)
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):fn(*inputs)
                end.record();end.synchronize();return start.elapsed_time(end)/args.repeats
            eager_ms=timing(eager);native_ms=timing(native);graph_ms=timing(graph)
        def compare(value):
            delta=(expected-value).abs();a=expected.float().flatten();b=value.float().flatten()
            return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                        cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)))
        report=dict(batch=args.batch,frames=frames,prompt_frames=prompt_frames,
                    dtypes=[str(v.dtype) for v in inputs],eager_ms=eager_ms,native_ms=native_ms,
                    graph_ms=graph_ms,speedup=eager_ms/native_ms,native=compare(actual),
                    graph=compare(graph_actual),native_stats=native.stats())
        path=Path(args.json_out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:engine.close()


if __name__=="__main__":main()
