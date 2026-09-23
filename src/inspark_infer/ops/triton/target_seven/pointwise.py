"""Offline-selected Target pointwise candidates. Unknown shapes retain upstream.

GELU preserves FP32 operation order, disables contraction and uses CUDA tanhf.
LayerNorm candidates keep FP32 statistics but change the reduction tree: they
must not be enabled merely because torch.allclose passes.
"""
import math
import torch
import triton as tr
import triton.language as tl
from triton.language.extra import cuda as cu


@tr.jit
def _gelu(X, Y, N:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,0)
    cube=(x*x)*x
    a=cube*0.044715
    b=x+a
    c=b*0.7978845608028654
    d=cu.libdevice.tanh(c)
    e=1.0+d
    half=0.5*x
    tl.store(Y+i,half*e,i<N)


def gelu(x, block=256, warps=4):
    assert x.dtype==torch.float32 and x.is_contiguous()
    y=torch.empty_like(x)
    _gelu[(tr.cdiv(x.numel(),block),)](x,y,x.numel(),block,num_warps=warps,enable_fp_fusion=False)
    return y


@tr.jit
def _ln_stats(X,S,N:tl.constexpr,K:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,K)
    x=tl.load(X+r*N+j,j<N,0).to(tl.float32)
    mean=tl.sum(x,0)/N
    z=tl.where(j<N,x-mean,0.)
    var=tl.sum(z*z,0)/N
    tl.store(S+2*r,mean); tl.store(S+2*r+1,var)


@tr.jit
def _ln_apply(X,W,B,S,Y,N:tl.constexpr,EPS:tl.constexpr,K:tl.constexpr):
    r=tl.program_id(0);j=tl.program_id(1)*K+tl.arange(0,K)
    x=tl.load(X+r*N+j,j<N,0)
    mean=tl.load(S+2*r);var=tl.load(S+2*r+1)
    w=tl.load(W+j,j<N,0);b=tl.load(B+j,j<N,0)
    y=((x-mean)*tl.rsqrt(var+EPS))*w+b
    tl.store(Y+r*N+j,y,j<N)


@tr.jit
def _ln(X,W,B,Y,N:tl.constexpr,EPS:tl.constexpr,K:tl.constexpr):
    r=tl.program_id(0);j=tl.arange(0,K)
    x=tl.load(X+r*N+j,j<N,0).to(tl.float32)
    mean=tl.sum(x,0)/N
    z=tl.where(j<N,x-mean,0.)
    var=tl.sum(z*z,0)/N
    w=tl.load(W+j,j<N,0);b=tl.load(B+j,j<N,0)
    y=((x-mean)*tl.rsqrt(var+EPS))*w+b
    tl.store(Y+r*N+j,y,j<N)


def layernorm(x,weight,bias,eps=1e-5,warps=4,split=False):
    assert x.dtype==torch.float32 and x.is_contiguous()
    n=x.shape[-1];m=x.numel()//n;y=torch.empty_like(x)
    if split:
        s=torch.empty((m,2),device=x.device,dtype=torch.float32)
        _ln_stats[(m,)](x,s,n,tr.next_power_of_2(n),num_warps=warps,enable_fp_fusion=False)
        _ln_apply[(m,tr.cdiv(n,256))](x,weight,bias,s,y,n,eps,256,num_warps=4,enable_fp_fusion=False)
    else:
        _ln[(m,)](x,weight,bias,y,n,eps,tr.next_power_of_2(n),num_warps=warps,enable_fp_fusion=False)
    return y


class PlannedPointwise(torch.nn.Module):
    """Explicit offline whitelist; no runtime tuning or implicit shape expansion."""
    def __init__(self,base,kind,entries):
        super().__init__();self.base=base;self.kind=kind
        self.entries={tuple(e['shape']):e['kwargs'] for e in entries if e.get('approved',False)}
    def forward(self,x):
        opts=self.entries.get(tuple(x.shape))
        if opts is None or x.dtype!=torch.float32 or not x.is_contiguous():return self.base(x)
        if self.kind=='gelu':return gelu(x,**opts)
        return layernorm(x,self.base.weight,self.base.bias,self.base.eps,**opts)
