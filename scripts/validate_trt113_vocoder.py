#!/usr/bin/env python3
"""Numeric, latency and CUDA Graph gate for native TRT11.3 BigVGAN."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",type=int,default=6)
    parser.add_argument("--config",default="configs/runtime.yaml")
    parser.add_argument("--plan",default="artifacts/trt113_vocoder/plan_b8.json")
    parser.add_argument("--iterations",type=int,default=50)
    parser.add_argument("--skip-graph",action="store_true")
    parser.add_argument("--output",default="outputs/trt113_vocoder_b8_validation.json")
    args=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    import torch
    from acc_infer_clear.config import load
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.tensorrt_backend.native113 import NativeVocoder113
    from acc_infer_clear.runtime.graphs import capture

    def bench(fn,value,n):
        for _ in range(5):fn(value)
        torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(n):fn(value)
        b.record();b.synchronize();return a.elapsed_time(b)/n
    def compare(expected,actual):
        delta=(expected.float()-actual.float()).abs();x=expected.float().flatten();y=actual.float().flatten()
        signal=x.square().mean();noise=(x-y).square().mean()
        return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                    cosine=float(torch.nn.functional.cosine_similarity(x,y,dim=0)),
                    snr_db=float(10*torch.log10(signal.clamp_min(1e-30)/noise.clamp_min(1e-30))))

    cfg=load(args.config);cfg["max_batch"]=8
    with GPULease(args.gpu):
        engine=Engine(cfg)
        try:
            engine.prepare_precision("bf16",["target","draft","cfm","vocoder"],True)
            engine.prepare_acoustic_kernels(None,"alias")
            eager=engine.vocoder;native=NativeVocoder113(args.plan,eager)
            gen=torch.Generator(device="cuda").manual_seed(113)
            mel=torch.randn(8,80,52,device="cuda",generator=gen)
            with torch.inference_mode():
                expected=eager(mel).clone();actual=native(mel).clone();torch.cuda.synchronize()
                graph=None if args.skip_graph else capture(native,(mel,))
                graphed=None if graph is None else graph(mel).clone();torch.cuda.synchronize()
                report={"batch":8,"frames":52,"random":compare(expected,actual),
                        "graph":None if graphed is None else compare(expected,graphed),
                        "eager_ms":bench(eager,mel,args.iterations),
                        "native_ms":bench(native,mel,args.iterations),
                        "graph_ms":None if graph is None else bench(graph,mel,args.iterations),
                        "native_stats":native.stats()}
            # Hard failure: never publish the kind of 0.4--0.7 max error that
            # the old Torch-TensorRT builder silently accepted.
            Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
            if report["random"]["cosine"]<0.9999 or report["random"]["mean_abs"]>0.005:
                raise RuntimeError("TensorRT Vocoder numeric gate failed: "+json.dumps(report["random"]))
        finally:engine.close()

if __name__=="__main__":main()
