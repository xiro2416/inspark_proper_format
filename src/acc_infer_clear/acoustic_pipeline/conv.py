"""SM120 implicit convolution with explicit async staging and XOR shared layout.

Pinned Triton3.5 Gluon binding: MMAv2 is emitted directly (FP8 QMMA / BF16 HMMA).
No materialized im2col. Per-sample and per-output scales remain unchanged.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as ac
from triton.experimental.gluon.language._core import builtin, tensor
from triton._C.libtriton import ir

@builtin
def mma(a,b,c,_semantic=None):
    # Gluon3.5 exposes the layouts but not a public MMAv2 convenience function.
    return tensor(_semantic.builder.create_dot(a.handle,b.handle,c.handle,ir.INPUT_PRECISION.IEEE,0),c.type)

@g.jit
def enqueue(X,W,aa,bb,mi,ni,kk,part,T:gl.constexpr,TO:gl.constexpr,M:gl.constexpr,CI:gl.constexpr,CO:gl.constexpr,KW:gl.constexpr,S:gl.constexpr,PAD:gl.constexpr,DIL:gl.constexpr,BK:gl.constexpr,TRANSPOSE:gl.constexpr):
    red=part*BK+kk;tap=red//CI;channel=red%CI
    if TRANSPOSE:
        numerator=(mi%TO)[:,None]+PAD-tap[None,:]*DIL;time=numerator//S;aligned=numerator%S==0
    else:
        time=(mi%TO)[:,None]*S-PAD+tap[None,:]*DIL;aligned=True
    ap=X+((mi//TO)[:,None]*T+time)*CI+channel[None,:]
    am=(mi[:,None]<M)&(red[None,:]<CI*KW)&(time>=0)&(time<T)&aligned
    WB:gl.constexpr=X.dtype.element_ty.primitive_bitwidth//8
    LB:gl.constexpr=gl.BlockedLayout([16//WB,1],[4,8],[1,4],[0,1])
    nr=gl.program_id(1)*bb.shape[1]+gl.arange(0,bb.shape[1],layout=gl.SliceLayout(0,LB))
    kr=part*BK+gl.arange(0,BK,layout=gl.SliceLayout(1,LB))
    wp=W+nr[None,:]*(CI*KW)+kr[:,None]
    wm=(nr[None,:]<CO)&(kr[:,None]<CI*KW)
    ac.async_copy_global_to_shared(aa,ap,am)
    ac.async_copy_global_to_shared(bb,wp,wm)
    ac.commit_group()

@g.jit
def conv_pipeline(X,W,SX,SW,BIAS,Y,T:gl.constexpr,TO:gl.constexpr,M:gl.constexpr,CI:gl.constexpr,CO:gl.constexpr,KW:gl.constexpr,S:gl.constexpr,PAD:gl.constexpr,DIL:gl.constexpr,
                  FP8:gl.constexpr,HAS_BIAS:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,P:gl.constexpr,SWIZZLE:gl.constexpr,DOUBLE:gl.constexpr,TRANSPOSE:gl.constexpr=False,INNER_DOUBLE:gl.constexpr=False):
    BYTES:gl.constexpr=1 if FP8 else 2
    L:gl.constexpr=gl.BlockedLayout([1,16//BYTES],[8,4],[4,1],[1,0])
    MM:gl.constexpr=gl.NVMMADistributedLayout(version=[2,0],warps_per_cta=[4,1],instr_shape=[16,8])
    DA:gl.constexpr=gl.DotOperandLayout(0,MM,4//BYTES)
    DB:gl.constexpr=gl.DotOperandLayout(1,MM,4//BYTES)
    PER:gl.constexpr=128//(BK*BYTES) if BK*BYTES<=128 else 1
    MAX:gl.constexpr=(BK*BYTES//16 if BK*BYTES<=128 else 8) if SWIZZLE else 1
    SL:gl.constexpr=gl.SwizzledSharedLayout(16//BYTES,PER,MAX,[1,0])
    WL:gl.constexpr=gl.SwizzledSharedLayout(16//BYTES,PER,MAX,[0,1])
    mi=gl.program_id(0)*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,L))
    ni=gl.program_id(1)*BN+gl.arange(0,BN,layout=gl.SliceLayout(1,L))
    kk=gl.arange(0,BK,layout=gl.SliceLayout(0,L))
    # Separate 2D descriptors: avoid Gluon3.5's rank-mismatched indexed swizzle descriptor.
    sa=(gl.allocate_shared_memory(X.dtype.element_ty,[BM,BK],SL),gl.allocate_shared_memory(X.dtype.element_ty,[BM,BK],SL),gl.allocate_shared_memory(X.dtype.element_ty,[BM,BK],SL),gl.allocate_shared_memory(X.dtype.element_ty,[BM,BK],SL))
    sb=(gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],WL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],WL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],WL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],WL))
    acc=gl.full((BM,BN),0,gl.float32,MM)
    ITER:gl.constexpr=(CI*KW+BK-1)//BK
    if DOUBLE:
        gl.static_assert(P>=2)
        for i in gl.static_range(P):enqueue(X,W,sa[i],sb[i],mi,ni,kk,i,T,TO,M,CI,CO,KW,S,PAD,DIL,BK,TRANSPOSE)
        ac.wait_group(P-1);gl.thread_barrier()
        a=sa[0].load(DA);b=sb[0].load(DB)
        for cycle in range((ITER+P-1)//P):
            for slot in gl.static_range(P):
                step=cycle*P+slot
                if step<ITER:
                    gl.thread_barrier()
                    enqueue(X,W,sa[slot],sb[slot],mi,ni,kk,step+P,T,TO,M,CI,CO,KW,S,PAD,DIL,BK,TRANSPOSE)
                    ac.wait_group(P-1);gl.thread_barrier()
                    an=sa[(slot+1)%P].load(DA);bn=sb[(slot+1)%P].load(DB)
                    acc=mma(a,b,acc);a=an;b=bn
    else:
        for i in gl.static_range(P-1):enqueue(X,W,sa[i],sb[i],mi,ni,kk,i,T,TO,M,CI,CO,KW,S,PAD,DIL,BK,TRANSPOSE)
        for cycle in range((ITER+P-1)//P):
            for slot in gl.static_range(P):
                step=cycle*P+slot
                if step<ITER:
                    gl.thread_barrier()
                    enqueue(X,W,sa[(slot+P-1)%P],sb[(slot+P-1)%P],mi,ni,kk,step+P-1,T,TO,M,CI,CO,KW,S,PAD,DIL,BK,TRANSPOSE)
                    ac.wait_group(P-1);gl.thread_barrier()
                    if INNER_DOUBLE:
                        FK:gl.constexpr=32 if FP8 else 16
                        a=sa[slot].slice(0,FK,1).load(DA);b=sb[slot].slice(0,FK,0).load(DB)
                        for sub in gl.static_range(1,BK//FK):
                            an=sa[slot].slice(sub*FK,FK,1).load(DA);bn=sb[slot].slice(sub*FK,FK,0).load(DB)
                            acc=mma(a,b,acc);a=an;b=bn
                        acc=mma(a,b,acc)
                    else:
                        a=sa[slot].load(DA);b=sb[slot].load(DB)
                        acc=mma(a,b,acc)
    ac.wait_group(0);gl.thread_barrier()
    O:gl.constexpr=gl.BlockedLayout([4,1],[8,4],[4,1],[0,1])
    acc=gl.convert_layout(acc,O)
    m=gl.program_id(0)*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,O));n=gl.program_id(1)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,O))
    if FP8:
        acc=acc*gl.load(SX+m//TO,m<M,0)[:,None]*gl.load(SW+n,n<CO,0)[None,:]
    else:acc=acc.to(gl.bfloat16).to(gl.float32)
    if HAS_BIAS:acc=acc+gl.load(BIAS+n,n<CO,0).to(gl.float32)[None,:]
    if not FP8:acc=acc.to(gl.bfloat16).to(gl.float32)
    gl.store(Y+((m//TO)[:,None]*CO+n[None,:])*TO+(m%TO)[:,None],acc,(m[:,None]<M)&(n[None,:]<CO))
