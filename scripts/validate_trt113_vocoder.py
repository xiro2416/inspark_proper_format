#!/usr/bin/env python3
"""Numeric, latency and CUDA Graph gate for native TRT11.3 BigVGAN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspark_infer.guardrails.numerics import TOLERANCES, compare


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,default=6)
    parser.add_argument("--config",default="configs/common/runtime.yaml")
    parser.add_argument("--plan",default='artifacts/trt113_vocoder/plan_b8.json')
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--frames",type=int,default=52)
    parser.add_argument("--seed",type=int,default=113)
    parser.add_argument("--iterations",type=int,default=50)
    parser.add_argument("--skip-graph",action="store_true")
    parser.add_argument("--output",default="outputs/trt113_vocoder_b8_validation.json")
    args=parser.parse_args()
    if min(args.batch,args.frames,args.iterations)<=0:
        parser.error("--batch, --frames and --iterations must be positive")
    plan=json.loads(Path(args.plan).read_text())
    if plan.get("batch")!=args.batch or plan.get("frames")!=args.frames:
        raise ValueError("Vocoder batch/frames must match the plan; eager fallback is not native validation")
    from inspark_infer.runtime.device import GPULease, select_gpu
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        validate(args)


def validate(args):

    import torch
    torch.backends.cudnn.allow_tf32=False
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.ops.tensorrt.native113 import NativeVocoder113
    from inspark_infer.runtime.graphs import capture

    def bench(fn,value,n):
        for _ in range(5):fn(value)
        torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(n):fn(value)
        b.record();b.synchronize();return a.elapsed_time(b)/n
    cfg=load(args.config);cfg["max_batch"]=args.batch;cfg["target_tf32"]=False
    engine=Engine(cfg)
    try:
        with torch.cuda.stream(engine.model.stream),torch.inference_mode():
            gen=torch.Generator(device="cuda").manual_seed(args.seed)
            mel=torch.randn(args.batch,80,args.frames,device="cuda",generator=gen)
            fp32_expected=engine.vocoder(mel).clone()
            # BF16 precision wrappers call PyTorch convolutions. Keep the original
            # PyTorch alias-free activation; installing the candidate invalidates the reference.
            engine.prepare_precision("bf16",["vocoder"],True)
            eager=engine.vocoder;expected=eager(mel).clone()
            native=NativeVocoder113(args.plan,eager)
            actual=native(mel).clone();torch.cuda.synchronize()
            graph=None if args.skip_graph else capture(native,(mel,))
            graphed=None if graph is None else graph(mel).clone();torch.cuda.synchronize()
            comparisons={"native_vs_fp32":compare(fp32_expected,actual,"bf16"),
                         "native_vs_bf16":compare(expected,actual,"bf16"),
                         "bf16_vs_fp32":compare(fp32_expected,expected,"bf16")}
            if graphed is not None:
                comparisons["graph_vs_fp32"]=compare(fp32_expected,graphed,"bf16")
                comparisons["graph_vs_bf16"]=compare(expected,graphed,"bf16")
            metrics=comparisons["native_vs_bf16"]
            legacy_gate=metrics.get("cosine",-1)>=0.9999 and metrics.get("mean_abs",float("inf"))<=0.005
            report={"batch":args.batch,"frames":args.frames,"seed":args.seed,
                    "device":torch.cuda.get_device_name(),"sm":list(torch.cuda.get_device_capability()),
                    "torch":torch.__version__,"tolerances":TOLERANCES,
                    "reference":"same-input pure PyTorch BigVGAN; original PyTorch alias-free activation; TF32 disabled",
                    "comparisons":comparisons,"random":metrics,"graph":comparisons.get("graph_vs_bf16"),
                    "legacy_gate":{"cosine_min":0.9999,"mean_abs_max":0.005,"pass_gate":legacy_gate},
                    "eager_ms":bench(eager,mel,args.iterations),
                    "native_ms":bench(native,mel,args.iterations),
                    "graph_ms":None if graph is None else bench(graph,mel,args.iterations),
                    "native_stats":native.stats()}
        stats=report["native_stats"]
        report["native_executed"]=stats["calls"]>0 and stats["fallbacks"]==0
        report["pass_gate"]=(report["native_executed"] and legacy_gate
                             and all(item["pass_gate"] for item in comparisons.values()))
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2,allow_nan=False))
        if not report["pass_gate"]:
            raise RuntimeError("Vocoder BF16 floating-point audit failed; report retains comparison-specific evidence")
    finally:engine.close()

if __name__=="__main__":main()
