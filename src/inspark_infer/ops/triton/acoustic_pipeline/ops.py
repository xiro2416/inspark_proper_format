"""Unchanged quantization/precision interfaces around explicit acoustic pipelines."""
import torch,triton
from inspark_infer.ops.triton.acoustic_pipeline.conv import conv_pipeline
from inspark_infer.ops.triton.fp8_conv import _partial_max, _scale
from inspark_infer.ops.triton.fp8_conv_ntc import _quant_ntc

def run(x,w,bias,ci,co,kw,stride,padding,dilation,plan,scales=None,return_kernel=False,transpose=False,output_padding=0):
    b,_,t=x.shape;to=(t-1)*stride-2*padding+dilation*(kw-1)+output_padding+1 if transpose else (t+2*padding-dilation*(kw-1)-1)//stride+1;fp8=scales is not None
    if fp8:
        x=x.contiguous();count=ci*t;parts=triton.cdiv(count,1024)
        partial=torch.empty(b,parts,device=x.device);sx=torch.empty(b,device=x.device);xx=torch.empty(b,t,ci,device=x.device,dtype=torch.float8_e4m3fn)
        _partial_max[(b,parts)](x,partial,count,parts,1024);bs=triton.next_power_of_2(parts)
        _scale[(b,)](partial,sx,parts,bs,num_warps=min(16,max(4,bs//2048)));_quant_ntc[(b,triton.cdiv(ci,32),triton.cdiv(t,32))](x,xx,sx,t,ci)
    else:
        xx=x.transpose(1,2).to(dtype=torch.bfloat16,memory_format=torch.contiguous_format);sx=w
    y=torch.empty(b,co,to,device=x.device,dtype=x.dtype)
    bm,bn,bk=plan['bm'],plan['bn'],plan['bk']
    k=conv_pipeline[(triton.cdiv(b*to,bm),triton.cdiv(co,bn))](xx,w,sx,scales if fp8 else w,bias if bias is not None else w,y,t,to,b*to,ci,co,kw,stride,padding,dilation,fp8,bias is not None,bm,bn,bk,plan['stages'],plan['swizzle'],plan['double'],transpose,plan.get('inner_double',False),num_warps=4)
    return (y,k) if return_kernel else y
