"""CFM-only candidate: offline N,K FP8 weights; original row/channel scales."""
import torch
import triton
import triton.language as tl
from inspark_infer.ops.triton.fp8 import _quantize

@triton.jit(do_not_specialize=['M'])
def _gemm_col(A,W,SA,SB,BIAS,Y,M,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for step in range(tl.cdiv(K,BK)):
        k=step*BK+kk
        a=tl.load(A+mi[:,None]*K+k[None,:],(mi[:,None]<M)&(k[None,:]<K),0.)
        w=tl.load(W+ni[None,:]*K+k[:,None],(ni[None,:]<N)&(k[:,None]<K),0.)
        acc=tl.dot(a,w,acc)
    y=acc*tl.load(SA+mi,mi<M,0.)[:,None]*tl.load(SB+ni,ni<N,0.)[None,:]
    if HAS_BIAS:y+=tl.load(BIAS+ni,ni<N,0.)[None,:]
    tl.store(Y+mi[:,None]*N+ni[None,:],y,(mi[:,None]<M)&(ni[None,:]<N))

@triton.jit(do_not_specialize=['M'])
def _epilogue(X,SA,SB,BIAS,Y,M,N:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<M*N
    y=tl.load(X+i,valid,0.)*tl.load(SA+i//N,valid,0.)*tl.load(SB+i%N,valid,0.)
    if HAS_BIAS:y+=tl.load(BIAS+i%N,valid,0.)
    tl.store(Y+i,y,valid)

def col_linear(x,weight,scales,bias,tile,backend='triton',ones=None):
    x=x.contiguous();k=weight.shape[1];n=scales.numel();flat=x.view(-1,k);m=flat.shape[0]
    q=torch.empty((m,k),device=x.device,dtype=torch.float8_e4m3fn);sx=torch.empty(m,device=x.device)
    _quantize[(m,)](flat,q,sx,k,triton.next_power_of_2(k));y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    if backend=='triton':
        _gemm_col[(triton.cdiv(m,tile.bm),triton.cdiv(n,tile.bn))](q,weight,sx,scales,bias if bias is not None else scales,y,m,n,k,tile.bm,tile.bn,tile.bk,bias is not None,num_warps=tile.warps,num_stages=tile.stages)
    else:
        # No replacement of original row/channel scale semantics: raw FP32 GEMM then same epilogue.
        raw=torch._scaled_mm(q,weight[:n].t(),scale_a=ones,scale_b=ones,out_dtype=torch.float32,use_fast_accum=False)
        _epilogue[(triton.cdiv(m*n,256),)](raw,sx,scales,bias if bias is not None else scales,y,m,n,bias is not None,256)
    return y.view(*x.shape[:-1],n)
