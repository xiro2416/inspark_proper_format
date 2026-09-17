"""Channels-contiguous activation loads while preserving original K reduction order."""
import torch
import triton
import triton.language as tl
from .fp8_conv import _partial_max,_scale
from .fp8_conv_ntc import _quant_ntc

@triton.jit(do_not_specialize=['T','TO','M'])
def _conv_ordered(X,W,SX,SW,BIAS,Y,T,TO,M,CI:tl.constexpr,CO:tl.constexpr,KW:tl.constexpr,WS:tl.constexpr,
                  STRIDE:tl.constexpr,PAD:tl.constexpr,DIL:tl.constexpr,TRANSPOSE:tl.constexpr,HAS_BIAS:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);batch=mi//TO;out_t=mi%TO;acc=tl.zeros((BM,BN),tl.float32)
    for part in range(tl.cdiv(CI*KW,BK)):
        red=part*BK+kk;channel=red//KW;tap=red%KW
        if TRANSPOSE:
            num=out_t[:,None]+PAD-tap[None,:]*DIL;time=num//STRIDE;aligned=num%STRIDE==0
        else:
            time=out_t[:,None]*STRIDE-PAD+tap[None,:]*DIL;aligned=tl.full((BM,BK),True,tl.int1)
        valid=(mi[:,None]<M)&(red[None,:]<CI*KW)&(time>=0)&(time<T)&aligned
        x=tl.load(X+(batch[:,None]*T+time)*CI+channel[None,:],valid,0.)
        w=tl.load(W+red[:,None]*WS+ni[None,:],(red[:,None]<CI*KW)&(ni[None,:]<CO),0.)
        acc=tl.dot(x,w,acc)
    y=acc*tl.load(SX+batch,mi<M,0.)[:,None]*tl.load(SW+ni,ni<CO,0.)[None,:]
    if HAS_BIAS:y+=tl.load(BIAS+ni,ni<CO,0.)[None,:]
    tl.store(Y+(batch[:,None]*CO+ni[None,:])*TO+out_t[:,None],y,(mi[:,None]<M)&(ni[None,:]<CO))

def conv1d_ordered(x,weight,scales,bias,ci,co,kw,stride,padding,dilation,transpose,output_padding,tile):
    x=x.contiguous();b,_,t=x.shape
    to=(t-1)*stride-2*padding+dilation*(kw-1)+output_padding+1 if transpose else (t+2*padding-dilation*(kw-1)-1)//stride+1
    count=ci*t;parts=triton.cdiv(count,1024);partial=torch.empty((b,parts),device=x.device);sx=torch.empty(b,device=x.device)
    q=torch.empty((b,t,ci),device=x.device,dtype=torch.float8_e4m3fn);y=torch.empty((b,co,to),device=x.device,dtype=x.dtype)
    _partial_max[(b,parts)](x,partial,count,parts,1024);bs=triton.next_power_of_2(parts)
    _scale[(b,)](partial,sx,parts,bs,num_warps=min(16,max(4,bs//2048)))
    _quant_ntc[(b,triton.cdiv(ci,32),triton.cdiv(t,32))](x,q,sx,t,ci)
    _conv_ordered[(triton.cdiv(b*to,tile.bm),triton.cdiv(co,tile.bn))](q,weight,sx,scales,bias if bias is not None else scales,y,
        t,to,b*to,ci,co,kw,weight.shape[1],stride,padding,dilation,transpose,bias is not None,tile.bm,tile.bn,tile.bk,
        num_warps=tile.warps,num_stages=tile.stages)
    return y
