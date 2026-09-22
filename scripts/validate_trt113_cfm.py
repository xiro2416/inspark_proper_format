#!/usr/bin/env python3
"""Same-input pure PyTorch reference and CUDA Graph audit for native TRT11.3 CFM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trt113_validation import TOLERANCES, compare


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,default=6)
    parser.add_argument("--batch",type=int,default=8);parser.add_argument("--repeats",type=int,default=50)
    parser.add_argument("--seed",type=int,default=113)
    parser.add_argument("--skip-graph",action="store_true")
    parser.add_argument("--config",default="configs/runtime.yaml")
    parser.add_argument("--reference",default="/workspace/index-tts/data/audio/old/mingxiang_gao.wav")
    parser.add_argument("--plan",default="artifacts/trt113_cfm/plan_b8.json")
    parser.add_argument("--json-out",default="outputs/trt113_cfm_b8_validation.json");args=parser.parse_args()
    if args.batch <= 0 or args.repeats <= 0:
        parser.error("--batch and --repeats must be positive")
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        validate(args)


def validate(args):
    import torch
    torch.backends.cudnn.allow_tf32=False
    from acc_infer_clear.config import load
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.runtime.graphs import capture
    from acc_infer_clear.tensorrt_backend.native113 import NativeCFMSolver113

    config=load(args.config);config["max_batch"]=args.batch
    config["target_tf32"]=False
    engine=Engine(config)
    try:
        engine.prepare_reference("reference",args.reference)
        voice=engine.model.bank.get("reference")["values"]
        prompt_frames=voice["voice.cache_mel"].shape[-1];frames=prompt_frames+52
        plan=json.loads(Path(args.plan).read_text())
        if plan.get("batch")!=args.batch or plan.get("frames")!=frames:
            raise ValueError("CFM batch/prompt must match the plan; eager fallback is not native validation")
        with torch.cuda.stream(engine.model.stream),torch.inference_mode():
            mu=torch.cat((voice["voice.cache_s2mel_prompt"],voice["voice.cache_s2mel_prompt"].new_zeros(1,52,512)),1).repeat(args.batch,1,1)
            generator=torch.Generator(device=mu.device).manual_seed(args.seed)
            x=torch.randn(args.batch,80,frames,device=mu.device,generator=generator)
            prompt=torch.zeros_like(x);prompt[:,:,:prompt_frames]=voice["voice.cache_mel"]
            lengths=torch.full((args.batch,),frames,device=x.device,dtype=torch.long)
            style=voice["voice.cache_s2mel_style"].repeat(args.batch,1)
            mask=(torch.arange(frames,device=x.device)[None,None]<prompt_frames).expand(args.batch,1,-1).clone()
            inputs=(x,prompt,lengths,style,mu,mask)
            fp32_expected=engine.student(*inputs).clone()
            # Reference uses only PyTorch precision wrappers; no custom kernels.
            engine.prepare_precision("bf16",["cfm"],True)
            eager=engine.student;expected=eager(*inputs).clone()
            native=NativeCFMSolver113(args.plan,eager)
            actual=native(*inputs).clone();torch.cuda.synchronize()
            graph=None if args.skip_graph else capture(native,inputs)
            graph_actual=None if graph is None else graph(*inputs).clone();torch.cuda.synchronize()
            def timing(fn):
                for _ in range(5):fn(*inputs)
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):fn(*inputs)
                end.record();end.synchronize();return start.elapsed_time(end)/args.repeats
            eager_ms=timing(eager);native_ms=timing(native);graph_ms=None if graph is None else timing(graph)
            comparisons={"native_vs_fp32":compare(fp32_expected,actual,"bf16"),
                         "native_vs_bf16":compare(expected,actual,"bf16"),
                         "bf16_vs_fp32":compare(fp32_expected,expected,"bf16")}
            if graph_actual is not None:
                comparisons["graph_vs_fp32"]=compare(fp32_expected,graph_actual,"bf16")
                comparisons["graph_vs_bf16"]=compare(expected,graph_actual,"bf16")
        stats=native.stats();native_executed=stats["calls"]>0 and stats["fallbacks"]==0
        report=dict(batch=args.batch,frames=frames,prompt_frames=prompt_frames,seed=args.seed,
                    device=torch.cuda.get_device_name(),sm=list(torch.cuda.get_device_capability()),
                    torch=torch.__version__,tolerances=TOLERANCES,
                    reference="same-input PyTorch; fixed two-step solver; no custom kernels; TF32 disabled",
                    dtypes=[str(v.dtype) for v in inputs],eager_ms=eager_ms,native_ms=native_ms,
                    graph_ms=graph_ms,speedup=eager_ms/native_ms,comparisons=comparisons,
                    native=comparisons["native_vs_bf16"],graph=comparisons.get("graph_vs_bf16"),
                    native_stats=stats,native_executed=native_executed,
                    pass_gate=native_executed and all(item["pass_gate"] for item in comparisons.values()))
        path=Path(args.json_out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False))
        if not report["pass_gate"]:
            raise RuntimeError("CFM BF16 floating-point audit failed; report retains comparison-specific evidence")
    finally:engine.close()


if __name__=="__main__":main()
