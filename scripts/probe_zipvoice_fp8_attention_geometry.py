"""Focused existing-kernel geometry probe; synthetic operator inputs, not audio/E2E."""
import argparse
import importlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=int,default=1)
    args=p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('Only physical GPU3')
    from inspark_infer.runtime.device import GPULease
    with GPULease(3):run(args)


def run(args):
    import torch
    import triton
    b=args.batch
    nonlinear=importlib.import_module(f'inspark_infer.ops.tensorrt.zipvoice_fp8.b{b}.online_nonlinear_rna_runtime_kernel').online_nonlinear_rna
    normal=importlib.import_module(f'inspark_infer.ops.tensorrt.zipvoice_fp8.b{b}.normal_tf32_runtime_kernel').online_branch_stats
    rows=[];torch.manual_seed(9100)
    for frames in [760,380,190]:
        q=torch.randn(4,b,frames,32,device='cuda')*.15;k=torch.randn(4,b,32,frames,device='cuda')*.15
        pq=torch.randn(4,b,frames,4,device='cuda')*.05;e=torch.randn(4,1,4,2*frames-1,device='cuda')*.05
        mask=torch.zeros(b,frames,device='cuda',dtype=torch.bool);mask[:,-11:]=True
        for name,kernel,nonlin in [('nonlinear',nonlinear,True),('normal',normal,False)]:
            value=torch.randn(1 if nonlin else 4,b,frames,384 if nonlin else 12,device='cuda')*.1
            output=torch.empty_like(value);stats=torch.empty(4,b,frames,2,device='cuda')
            reference=None
            for qb,kb,warps in [(32,16,4),(16,16,4),(16,32,4),(32,32,4),(64,32,4),(16,64,4),(32,64,4),(64,64,4)]:
                try:
                    launch=(triton.cdiv(frames,qb),b,1 if nonlin else 4)
                    def invoke():
                        if nonlin:return kernel[launch](q,k,pq,e,mask,value,output,frames,True,qb,kb,'tf32',True,num_warps=warps,num_stages=1,enable_fp_fusion=False)
                        return kernel[launch](q,k,pq,e,mask,value,output,stats,frames,False,qb,kb,'tf32',0,num_warps=warps,num_stages=1,enable_fp_fusion=False)
                    compiled=invoke();torch.cuda.synchronize()
                    if not torch.isfinite(output).all():raise RuntimeError('Nonfinite geometry output')
                    if reference is None:reference=output.clone()
                    difference=float((output-reference).abs().max())
                    for _ in range(5):invoke()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):invoke()
                    end.record();end.synchronize()
                    rows.append(dict(kind=name,batch=b,frames=frames,qb=qb,kb=kb,warps=warps,
                                     micro_ms=start.elapsed_time(end)/30,registers=compiled.n_regs,spills=compiled.n_spills,
                                     shared=compiled.metadata.shared,max_abs_vs_first_geometry=difference))
                except Exception as ex:rows.append(dict(kind=name,batch=b,frames=frames,qb=qb,kb=kb,warps=warps,error=str(ex)))
                write(ROOT/f'outputs/fp8/b{b}/reports/009-attention-geometry.json',dict(status='operator_geometry_probe',scope='Synthetic masked operator inputs only; no audio quality or E2E acceptance',rows=rows))
    print('GEOMETRY_DONE',b,flush=True)


if __name__=='__main__':main()
