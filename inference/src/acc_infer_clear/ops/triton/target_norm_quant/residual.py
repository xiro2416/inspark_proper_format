"""FP8 projection epilogue + residual fusion candidates."""
import torch
import triton
import triton.language as tl
from acc_infer_clear.ops.triton.fp8 import _quantize

@triton.jit(do_not_specialize=['M'])
def _gemm_col_residual(A,W,SA,SB,BIAS,R,Y,M,N:tl.constexpr,K:tl.constexpr,
                       BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for step in range(tl.cdiv(K,BK)):
        k=step*BK+kk
        a=tl.load(A+mi[:,None]*K+k[None,:],(mi[:,None]<M)&(k[None,:]<K),0.)
        w=tl.load(W+ni[None,:]*K+k[:,None],(ni[None,:]<N)&(k[:,None]<K),0.)
        acc=tl.dot(a,w,acc)
    value=acc*tl.load(SA+mi,mi<M,0.)[:,None]*tl.load(SB+ni,ni<N,0.)[None,:]
    if HAS_BIAS:value+=tl.load(BIAS+ni,ni<N,0.)[None,:]
    value+=tl.load(R+mi[:,None]*N+ni[None,:],(mi[:,None]<M)&(ni[None,:]<N),0.)
    tl.store(Y+mi[:,None]*N+ni[None,:],value,(mi[:,None]<M)&(ni[None,:]<N))

def col_linear_residual(x,residual,weight_col,scales,bias,tile):
    x=x.contiguous();residual=residual.contiguous();k=weight_col.shape[1];n=scales.numel();flat=x.view(-1,k);m=flat.shape[0]
    if residual.numel()!=m*n or residual.dtype!=torch.float32:raise ValueError('Residual mismatch')
    q=torch.empty((m,k),device=x.device,dtype=torch.float8_e4m3fn);sx=torch.empty(m,device=x.device);y=torch.empty((m,n),device=x.device)
    _quantize[(m,)](flat,q,sx,k,triton.next_power_of_2(k))
    _gemm_col_residual[(triton.cdiv(m,tile['bm']),triton.cdiv(n,tile['bn']))](q,weight_col,sx,scales,bias if bias is not None else scales,residual,y,m,n,k,tile['bm'],tile['bn'],tile['bk'],bias is not None,num_warps=tile['warps'],num_stages=tile['stages'])
    return y.view_as(residual)
