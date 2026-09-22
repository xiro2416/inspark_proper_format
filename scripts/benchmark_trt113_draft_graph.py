#!/usr/bin/env python3
"""Numerical and CUDA-event A/B for the complete Draft backbone."""
import argparse,json,os,statistics
from pathlib import Path

def measure(torch,fn,warmups=20,iterations=200):
    for _ in range(warmups):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(20):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(iterations//20):fn()
        end.record();end.synchronize();samples.append(start.elapsed_time(end)/(iterations//20))
    return {"mean_ms":statistics.fmean(samples),"median_ms":statistics.median(samples),"min_ms":min(samples),"max_ms":max(samples)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,default=6);p.add_argument("--batch",type=int,required=True)
    p.add_argument("--config",default="configs/runtime.yaml");p.add_argument("--deployment",default="configs/sm89_bf16_trt113_draft_lab.json");p.add_argument("--out",required=True);args=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.graphs import capture
    from acc_infer_clear.streaming.engine import Engine
    config=load_config(args.config);config["max_batch"]=args.batch
    with GPULease(args.gpu):
        engine=Engine(config)
        try:
            engine.prepare_deployment(load_deployment(args.deployment));draft=engine.rt.backbone;bank=draft.native_full_bank;b=args.batch
            if (b,128) not in bank.graphs:raise RuntimeError(f"No native Draft engine for B{b}")
            param=next(draft.model.parameters());anchors=torch.arange(b,device=param.device,dtype=torch.long)%8194
            positions=torch.arange(7,device=param.device)[None].expand(b,-1).clone();slots=torch.arange(b,device=param.device,dtype=torch.int32);lengths=torch.full((b,),121,device=param.device,dtype=torch.int32)
            with torch.inference_mode():
                torch.manual_seed(20260922);draft.pool.storage[:,:,:b,:,:128].zero_();bank.cache[:,:,:b].copy_(draft.pool.storage[:,:,:b,:,:128])
                current=capture(lambda aa,pp,ss,ll:draft.math(aa,pp,ss,ll,128),(anchors,positions,slots,lengths));native=bank.graphs[b,128]
                ch,cb=current(anchors,positions,slots,lengths);nh,nb=native(anchors,positions,slots,lengths);torch.cuda.synchronize()
            def diff(a,z):
                d=(a.float()-z.float()).abs();return {"max_abs":float(d.max()),"mean_abs":float(d.mean()),"cosine":float(torch.nn.functional.cosine_similarity(a.float().flatten(),z.float().flatten(),dim=0))}
            with torch.inference_mode():
                ct=measure(torch,lambda:current(anchors,positions,slots,lengths));nt=measure(torch,lambda:native(anchors,positions,slots,lengths))
            result={"batch":b,"current":ct,"native_trt113":nt,"speedup":ct["median_ms"]/nt["median_ms"],"delta_pct":(nt["median_ms"]/ct["median_ms"]-1)*100,"correctness":{"hidden":diff(ch,nh),"base":diff(cb,nb),"finite":{"current_hidden":bool(torch.isfinite(ch).all()),"native_hidden":bool(torch.isfinite(nh).all()),"current_base":bool(torch.isfinite(cb).all()),"native_base":bool(torch.isfinite(nb).all())}},"memory":{"allocated_mib":torch.cuda.memory_allocated()>>20,"reserved_mib":torch.cuda.memory_reserved()>>20},"trt_version":bank.backends[b].trt.__version__}
            path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
        finally:engine.close()
if __name__=="__main__":main()
