"""Joint shared layout and explicit async/register pipelines for small-M W8A8.

Inspired by Marlin's scheduling principles, not its W4A16 dequant/reduction code.
Preserves unsplit FP32 accumulation, original scales and original output dtype.
"""
import torch,triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as ac
from acc_infer_clear.acoustic_pipeline.conv import mma
from acc_infer_clear.kernels.fp8 import _quantize

@g.jit
def enqueue(A,W,SA,SB,mi,kk,part,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BK:gl.constexpr,BN:gl.constexpr):
    rk=part*BK+kk
    ap=A+mi[:,None]*K+rk[None,:];am=(mi[:,None]<M)&(rk[None,:]<K)
    LB:gl.constexpr=gl.BlockedLayout([16,1],[8,4],[1,4],[0,1])
    nn=gl.program_id(1)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,LB))
    wk=part*BK+gl.arange(0,BK,layout=gl.SliceLayout(1,LB))
    wp=W+nn[None,:]*K+wk[:,None];wm=(nn[None,:]<N)&(wk[:,None]<K)
    ac.async_copy_global_to_shared(SA,ap,am);ac.async_copy_global_to_shared(SB,wp,wm);ac.commit_group()

@g.jit
def pipeline_gemm(A,W,AS,WS,B,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,HAS_BIAS:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,P:gl.constexpr,SWIZZLE:gl.constexpr,INNER:gl.constexpr,DOUBLE:gl.constexpr):
    L:gl.constexpr=gl.BlockedLayout([1,16],[4,8],[4,1],[1,0])
    MM:gl.constexpr=gl.NVMMADistributedLayout(version=[2,0],warps_per_cta=[1,4],instr_shape=[16,8])
    DA:gl.constexpr=gl.DotOperandLayout(0,MM,4);DB:gl.constexpr=gl.DotOperandLayout(1,MM,4)
    PH:gl.constexpr=128//BK if BK<=128 else 1;MAX:gl.constexpr=(BK//16 if BK<=128 else 8) if SWIZZLE else 1
    LA:gl.constexpr=gl.SwizzledSharedLayout(16,PH,MAX,[1,0]);LB:gl.constexpr=gl.SwizzledSharedLayout(16,PH,MAX,[0,1])
    sa=(gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],LA))
    sb=(gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],LB))
    mi=gl.program_id(0)*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,L));kk=gl.arange(0,BK,layout=gl.SliceLayout(0,L));acc=gl.full((BM,BN),0,gl.float32,MM)
    ITER:gl.constexpr=(K+BK-1)//BK
    if DOUBLE:
        gl.static_assert(P>=2)
        for i in gl.static_range(P):enqueue(A,W,sa[i],sb[i],mi,kk,i,M,N,K,BK,BN)
        ac.wait_group(P-1);gl.thread_barrier();a=sa[0].load(DA);b=sb[0].load(DB)
        for cycle in range((ITER+P-1)//P):
            for slot in gl.static_range(P):
                step=cycle*P+slot
                if step<ITER:
                    gl.thread_barrier();enqueue(A,W,sa[slot],sb[slot],mi,kk,step+P,M,N,K,BK,BN)
                    ac.wait_group(P-1);gl.thread_barrier()
                    an=sa[(slot+1)%P].load(DA);bn=sb[(slot+1)%P].load(DB)
                    acc=mma(a,b,acc);a=an;b=bn
    else:
        for i in gl.static_range(P-1):enqueue(A,W,sa[i],sb[i],mi,kk,i,M,N,K,BK,BN)
        for cycle in range((ITER+P-1)//P):
            for slot in gl.static_range(P):
                step=cycle*P+slot
                if step<ITER:
                    gl.thread_barrier();enqueue(A,W,sa[(slot+P-1)%P],sb[(slot+P-1)%P],mi,kk,step+P-1,M,N,K,BK,BN)
                    ac.wait_group(P-1);gl.thread_barrier()
                    if INNER:
                        a=sa[slot].slice(0,32,1).load(DA);b=sb[slot].slice(0,32,0).load(DB)
                        for sub in gl.static_range(1,BK//32):
                            an=sa[slot].slice(sub*32,32,1).load(DA);bn=sb[slot].slice(sub*32,32,0).load(DB)
                            acc=mma(a,b,acc);a=an;b=bn
                        acc=mma(a,b,acc)
                    else:acc=mma(sa[slot].load(DA),sb[slot].load(DB),acc)
    ac.wait_group(0);gl.thread_barrier()
    O:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[4,1],[1,0]);acc=gl.convert_layout(acc,O)
    mm=gl.program_id(0)*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,O));nn=gl.program_id(1)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,O))
    result=acc*gl.load(AS+mm,mm<M,0)[:,None]*gl.load(WS+nn,nn<N,0)[None,:]
    if HAS_BIAS:result=result+gl.load(B+nn,nn<N,0)[None,:]
    gl.store(Y+mm[:,None]*N+nn[None,:],result,(mm[:,None]<M)&(nn[None,:]<N))

def run(x,weight,scales,bias,plan,return_kernel=False):
    x=x.contiguous();k=weight.shape[1];n=scales.numel();flat=x.view(-1,k);m=flat.shape[0]
    q=torch.empty(m,k,device=x.device,dtype=torch.float8_e4m3fn);sx=torch.empty(m,device=x.device);y=torch.empty(m,n,device=x.device,dtype=x.dtype)
    _quantize[(m,)](flat,q,sx,k,triton.next_power_of_2(k))
    bm,bn,bk=plan.get('bm',16),plan.get('bn',32),plan.get('bk',128)
    compiled=pipeline_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](q,weight,sx,scales,bias if bias is not None else scales,y,m,n,k,bias is not None,bm,bn,bk,plan['stages'],plan.get('swizzle',True),plan.get('inner',False),plan.get('double',False),num_warps=4)
    y=y.view(*x.shape[:-1],n)
    return (y,compiled) if return_kernel else y
