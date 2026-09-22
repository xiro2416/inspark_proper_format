"""Native W8A8 E4M3 GEMM with request-independent row scales.

Standalone Triton; weights packed offline. Full quantize+GEMM cost must be measured.
No eager-weight dequantization, materialized im2col, or online autotuning.
"""
import torch
import triton
import triton.language as tl
from acc_infer_clear.quantization.weights import pack_weight  # compatibility export

@triton.jit
def _quantize(X,Y,S,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    x=tl.load(X+row*K+col,col<K,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)/448.,1e-12)
    tl.store(Y+row*K+col,(x/scale).to(Y.dtype.element_ty),col<K)
    tl.store(S+row,scale)

@triton.jit(do_not_specialize=['M'])
def _gemm(A,B,SA,SB,BIAS,C,M,N:tl.constexpr,K:tl.constexpr,WS:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr,SPLITS:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN);ki=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    part=0
    if SPLITS>1:part=tl.program_id(2)
    for start in range(part,tl.cdiv(K,BK),SPLITS):
        kk=start*BK+ki
        a=tl.load(A+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K),0.)
        b=tl.load(B+kk[:,None]*WS+ni[None,:],(kk[:,None]<K)&(ni[None,:]<N),0.)
        acc=tl.dot(a,b,acc)
    if SPLITS==1:
        sa=tl.load(SA+mi,mi<M,0);sb=tl.load(SB+ni,ni<N,0)
        out=acc*sa[:,None]*sb[None,:]
        if HAS_BIAS:out+=tl.load(BIAS+ni,ni<N,0)[None,:]
        tl.store(C+mi[:,None]*N+ni[None,:],out,(mi[:,None]<M)&(ni[None,:]<N))
    else:tl.store(C+part*M*N+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))

@triton.jit(do_not_specialize=['M'])
def _reduce(P,SA,SB,BIAS,C,M,N:tl.constexpr,SPLITS:tl.constexpr,HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<M*N;value=tl.zeros((BLOCK,),tl.float32)
    for part in range(SPLITS):value+=tl.load(P+part*M*N+i,valid,0.)
    value*=tl.load(SA+i//N,valid,0.)*tl.load(SB+i%N,valid,0.)
    if HAS_BIAS:value+=tl.load(BIAS+i%N,valid,0.)
    tl.store(C+i,value,valid)

def linear(x,packed,scales,bias=None,tile=None,return_kernel=False):
    from acc_infer_clear.ops.planning.planner import Tile
    if tile is None:raise ValueError('Explicit offline-selected tile required')
    if tile.schedule!='tiled':raise ValueError('Unimplemented schedule')
    k,n=packed.shape[0],scales.numel();x=x.contiguous();flat=x.view(-1,k);m=flat.shape[0]
    q=torch.empty((m,k),device=x.device,dtype=torch.float8_e4m3fn)
    sx=torch.empty(m,device=x.device,dtype=torch.float32)
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _quantize[(m,)](flat,q,sx,k,triton.next_power_of_2(k))
    workspace=out if tile.split_k==1 else torch.empty((tile.split_k,m,n),device=x.device,dtype=torch.float32)
    compiled=_gemm[(triton.cdiv(m,tile.bm),triton.cdiv(n,tile.bn),tile.split_k)](
        q,packed,sx,scales,bias if bias is not None else scales,workspace,m,n,k,packed.shape[1],
        tile.bm,tile.bn,tile.bk,bias is not None,tile.split_k,num_warps=tile.warps,num_stages=tile.stages)
    if tile.split_k!=1:
        _reduce[(triton.cdiv(m*n,256),)](workspace,sx,scales,bias if bias is not None else scales,out,m,n,tile.split_k,bias is not None,256)
    output=out.view(*x.shape[:-1],n)
    return (output,compiled) if return_kernel else output
