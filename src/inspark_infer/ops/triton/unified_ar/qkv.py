"""Unified QKV GEMM with fused per-row/head E4M3 output epilogues.

BN32 partitions each head into two independently scaled blocks while preserving
one kernel family and fixed two-stage schedule across the supported M values.
"""
import torch
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as ac
from inspark_infer.ops.triton.target_full_m.tiled import enqueue, consume, normalize_plan


@g.jit
def output_fp8(ACC, AS, WS, BIAS, Y, SY, row_start,
               M:gl.constexpr,N:gl.constexpr,ROWS:gl.constexpr,BN:gl.constexpr,
               WARPS:gl.constexpr,HAS_BIAS:gl.constexpr):
    O:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[WARPS,1],[1,0])
    acc=gl.convert_layout(ACC,O)
    mm=row_start+gl.arange(0,ROWS,layout=gl.SliceLayout(1,O))
    nn=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,O))
    result=acc*gl.load(AS+mm,mm<M,0.)[:,None]*gl.load(WS+nn,nn<N,0.)[None,:]
    if HAS_BIAS:result+=gl.load(BIAS+nn,nn<N,0.)[None,:]
    scale=gl.maximum(gl.max(gl.abs(result),axis=1)/448.,1e-12)
    gl.store(Y+mm[:,None]*N+nn[None,:],result/scale[:,None],(mm[:,None]<M)&(nn[None,:]<N))
    gl.store(SY+mm*(N//BN)+gl.program_id(0),scale,mm<M)


@g.jit
def output_target_slots(ACC,AS,WS,BIAS,QO,SQ,PK,SK,PV,SV,SLOTS,LENGTHS,row_start,
                        M:gl.constexpr,N:gl.constexpr,ROWS:gl.constexpr,BN:gl.constexpr,
                        WARPS:gl.constexpr,H:gl.constexpr,NQ:gl.constexpr,D:gl.constexpr,
                        CAP:gl.constexpr,HAS_BIAS:gl.constexpr,V_FP32:gl.constexpr):
    O:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[WARPS,1],[1,0])
    acc=gl.convert_layout(ACC,O);tile=gl.program_id(0);BLOCKS:gl.constexpr=D//BN;kind=tile//(H*BLOCKS);head=(tile//BLOCKS)%H;part=tile%BLOCKS
    mm=row_start+gl.arange(0,ROWS,layout=gl.SliceLayout(1,O));dd=gl.arange(0,BN,layout=gl.SliceLayout(0,O))
    nn=tile*BN+dd;result=acc*gl.load(AS+mm,mm<M,0.)[:,None]*gl.load(WS+nn,nn<N,0.)[None,:]
    if HAS_BIAS:result+=gl.load(BIAS+nn,nn<N,0.)[None,:]
    scale=gl.maximum(gl.max(gl.abs(result),axis=1)/448.,1e-12);quant=result/scale[:,None]
    request=mm//NQ;token=mm%NQ;slot=gl.load(SLOTS+request,request<M//NQ,0);dest=gl.load(LENGTHS+request,request<M//NQ,0)+token
    valid=(mm[:,None]<M)&(dd[None,:]<D)
    qoff=((request[:,None]*H+head)*NQ+token[:,None])*D+part*BN+dd[None,:]
    kvoff=((slot[:,None]*H+head)*CAP+dest[:,None])*D+part*BN+dd[None,:]
    soff=((slot*H+head)*CAP+dest)*BLOCKS+part
    gl.store(QO+qoff,quant,valid&(kind==0));gl.store(PK+kvoff,quant,valid&(kind==1))
    if V_FP32:gl.store(PV+kvoff,result,valid&(kind==2))
    else:gl.store(PV+kvoff,quant,valid&(kind==2))
    gl.store(SQ+((request*H+head)*NQ+token)*BLOCKS+part,scale,(mm<M)&(kind==0))
    gl.store(SK+soff,scale,(mm<M)&(kind==1))
    if not V_FP32:gl.store(SV+soff,scale,(mm<M)&(kind==2))


@g.jit
def output_qkv_mixed(ACC,AS,WS,BIAS,QK,SQK,V,row_start,
                     M:gl.constexpr,N:gl.constexpr,ROWS:gl.constexpr,BN:gl.constexpr,
                     WARPS:gl.constexpr,H:gl.constexpr,D:gl.constexpr,QLEN:gl.constexpr,HAS_BIAS:gl.constexpr):
    O:gl.constexpr=gl.BlockedLayout([1,4],[4,8],[WARPS,1],[1,0])
    acc=gl.convert_layout(ACC,O);tile=gl.program_id(0);BLOCKS:gl.constexpr=D//BN;kind=tile//(H*BLOCKS);head=(tile//BLOCKS)%H;part=tile%BLOCKS
    mm=row_start+gl.arange(0,ROWS,layout=gl.SliceLayout(1,O));dd=gl.arange(0,BN,layout=gl.SliceLayout(0,O));nn=tile*BN+dd
    result=acc*gl.load(AS+mm,mm<M,0.)[:,None]*gl.load(WS+nn,nn<N,0.)[None,:]
    if HAS_BIAS:result+=gl.load(BIAS+nn,nn<N,0.)[None,:]
    scale=gl.maximum(gl.max(gl.abs(result),axis=1)/448.,1e-12);valid=(mm[:,None]<M)&(dd[None,:]<D)
    qkoff=(((kind*M+mm[:,None])*H+head)*D+part*BN+dd[None,:]);voff=((mm[:,None]*H+head)*D+part*BN+dd[None,:])
    gl.store(QK+qkoff,result/scale[:,None],valid&(kind<2));gl.store(V+voff,result,valid&(kind==2))
    if QLEN:
        B:gl.constexpr=M//QLEN;request=mm//QLEN;token=mm%QLEN;scale_offset=((((kind*B+request)*H+head)*QLEN+token)*BLOCKS+part)
    else:scale_offset=((kind*M+mm)*H+head)*BLOCKS+part
    gl.store(SQK+scale_offset,scale,(mm<M)&(kind<2))


@g.jit
def full_m_qkv_fp8(A,W,AS,WS,BIAS,Y,SY,M:gl.constexpr,N:gl.constexpr,
                    K:gl.constexpr,BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,
                    P:gl.constexpr,WARPS:gl.constexpr,WM:gl.constexpr,WN:gl.constexpr,
                    SWIZZLE:gl.constexpr,HAS_BIAS:gl.constexpr):
    gl.static_assert(BN==64 and N%BN==0)
    MM:gl.constexpr=gl.NVMMADistributedLayout(version=[2,0],warps_per_cta=[WM,WN],instr_shape=[16,8])
    PER:gl.constexpr=128//BK if BK<=128 else 1;MAX:gl.constexpr=(BK//16 if BK<=128 else 8) if SWIZZLE else 1
    AL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[1,0]);BL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[0,1])
    sa=(gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),
        gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),
        gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL))
    sb=(gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),
        gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),
        gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL))
    accs=(gl.full((BM,BN),0,gl.float32,MM),);ITER:gl.constexpr=(K+BK-1)//BK
    for part in gl.static_range(P-1):enqueue(A,W,sa[part],sb[part],part,M,N,K,BM,BN,BK,WARPS,'','')
    for cycle in range((ITER+P-1)//P):
        for slot in gl.static_range(P):
            step=cycle*P+slot
            if step<ITER:
                gl.thread_barrier();enqueue(A,W,sa[(slot+P-1)%P],sb[(slot+P-1)%P],step+P-1,M,N,K,BM,BN,BK,WARPS,'','')
                ac.wait_group(P-1);gl.thread_barrier();accs=consume(sa[slot],sb[slot],accs,BM,BK,MM,0)
    ac.wait_group(0);gl.thread_barrier();output_fp8(accs[0],AS,WS,BIAS,Y,SY,0,M,N,BM,BN,WARPS,HAS_BIAS)


@g.jit
def full_m_target_slots(A,W,AS,WS,BIAS,QO,SQ,PK,SK,PV,SV,SLOTS,LENGTHS,
                        M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,BM:gl.constexpr,
                        BN:gl.constexpr,BK:gl.constexpr,P:gl.constexpr,WARPS:gl.constexpr,
                        WM:gl.constexpr,WN:gl.constexpr,SWIZZLE:gl.constexpr,
                        H:gl.constexpr,NQ:gl.constexpr,D:gl.constexpr,CAP:gl.constexpr,
                        HAS_BIAS:gl.constexpr,V_FP32:gl.constexpr):
    gl.static_assert(D%BN==0 and N==3*H*D and M%NQ==0)
    MM:gl.constexpr=gl.NVMMADistributedLayout(version=[2,0],warps_per_cta=[WM,WN],instr_shape=[16,8])
    PER:gl.constexpr=128//BK if BK<=128 else 1;MAX:gl.constexpr=(BK//16 if BK<=128 else 8) if SWIZZLE else 1
    AL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[1,0]);BL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[0,1])
    sa=(gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),
        gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),
        gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL))
    sb=(gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),
        gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),
        gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL))
    accs=(gl.full((BM,BN),0,gl.float32,MM),);ITER:gl.constexpr=(K+BK-1)//BK
    for part in gl.static_range(P-1):enqueue(A,W,sa[part],sb[part],part,M,N,K,BM,BN,BK,WARPS,'','')
    for cycle in range((ITER+P-1)//P):
        for slot in gl.static_range(P):
            step=cycle*P+slot
            if step<ITER:
                gl.thread_barrier();enqueue(A,W,sa[(slot+P-1)%P],sb[(slot+P-1)%P],step+P-1,M,N,K,BM,BN,BK,WARPS,'','')
                ac.wait_group(P-1);gl.thread_barrier();accs=consume(sa[slot],sb[slot],accs,BM,BK,MM,0)
    ac.wait_group(0);gl.thread_barrier()
    output_target_slots(accs[0],AS,WS,BIAS,QO,SQ,PK,SK,PV,SV,SLOTS,LENGTHS,0,M,N,BM,BN,WARPS,H,NQ,D,CAP,HAS_BIAS,V_FP32)


@g.jit
def full_m_qkv_mixed(A,W,AS,WS,BIAS,QK,SQK,V,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,
                     BM:gl.constexpr,BN:gl.constexpr,BK:gl.constexpr,P:gl.constexpr,
                     WARPS:gl.constexpr,WM:gl.constexpr,WN:gl.constexpr,SWIZZLE:gl.constexpr,
                     H:gl.constexpr,D:gl.constexpr,QLEN:gl.constexpr,HAS_BIAS:gl.constexpr):
    gl.static_assert(D%BN==0 and N==3*H*D)
    MM:gl.constexpr=gl.NVMMADistributedLayout(version=[2,0],warps_per_cta=[WM,WN],instr_shape=[16,8])
    PER:gl.constexpr=128//BK if BK<=128 else 1;MAX:gl.constexpr=(BK//16 if BK<=128 else 8) if SWIZZLE else 1
    AL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[1,0]);BL:gl.constexpr=gl.SwizzledSharedLayout(16,PER,MAX,[0,1])
    sa=(gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL),gl.allocate_shared_memory(A.dtype.element_ty,[BM,BK],AL))
    sb=(gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL),gl.allocate_shared_memory(W.dtype.element_ty,[BK,BN],BL))
    accs=(gl.full((BM,BN),0,gl.float32,MM),);ITER:gl.constexpr=(K+BK-1)//BK
    for part in gl.static_range(P-1):enqueue(A,W,sa[part],sb[part],part,M,N,K,BM,BN,BK,WARPS,'','')
    for cycle in range((ITER+P-1)//P):
        for slot in gl.static_range(P):
            step=cycle*P+slot
            if step<ITER:
                gl.thread_barrier();enqueue(A,W,sa[(slot+P-1)%P],sb[(slot+P-1)%P],step+P-1,M,N,K,BM,BN,BK,WARPS,'','')
                ac.wait_group(P-1);gl.thread_barrier();accs=consume(sa[slot],sb[slot],accs,BM,BK,MM,0)
    ac.wait_group(0);gl.thread_barrier();output_qkv_mixed(accs[0],AS,WS,BIAS,QK,SQK,V,0,M,N,BM,BN,WARPS,H,D,QLEN,HAS_BIAS)


def run_prequantized(q,sx,weight_col,weight_scales,bias,plan):
    m,k=q.shape;n=weight_scales.numel();p=normalize_plan(m,plan)
    if p['bn']!=64 or p['schedule']!='full' or n%64:raise ValueError('Experiment requires full schedule and BN64/head64')
    y=torch.empty((m,n),device=q.device,dtype=torch.float8_e4m3fn);sy=torch.empty((m,n//64),device=q.device,dtype=torch.float32)
    full_m_qkv_fp8[(n//64,)](q,weight_col,sx,weight_scales,bias if bias is not None else weight_scales,y,sy,
        m,n,k,p['bm'],p['bn'],p['bk'],p['stages'],p['warps'],p['wm'],p['wn'],p['swizzle'],bias is not None,
        num_warps=p['warps'])
    return y,sy


def run_target_slots(q,sx,weight_col,weight_scales,bias,plan,pk,sk,pv,sv,slots,lengths,nq=8,v_fp32=False):
    m,k=q.shape;n=weight_scales.numel();p=normalize_plan(m,plan);h=n//(3*64);blocks=64//p['bn']
    if p['schedule']!='full' or 64%p['bn'] or m!=slots.numel()*nq:raise ValueError('Target slot specialization mismatch')
    qo=torch.empty((slots.numel(),h,nq,64),device=q.device,dtype=torch.float8_e4m3fn);sq=torch.empty((*qo.shape[:-1],blocks),device=q.device,dtype=torch.float32)
    full_m_target_slots[(n//p['bn'],)](q,weight_col,sx,weight_scales,bias if bias is not None else weight_scales,
        qo,sq,pk,sk,pv,sv,slots,lengths,m,n,k,p['bm'],p['bn'],p['bk'],p['stages'],p['warps'],p['wm'],p['wn'],p['swizzle'],
        h,nq,64,pk.shape[-2],bias is not None,bool(v_fp32),num_warps=p['warps'])
    return qo,sq


def unified_plan(m,bn=32):
    bm=max(16,1<<(int(m)-1).bit_length());warps=4 if bm<=32 else 8
    return dict(mode='tiled',bm=bm,bn=int(bn),bk=128,stages=2,warps=warps,wm=2,wn=warps//2,schedule='full',swizzle=True)


def run_qkv_mixed(q,sx,weight_col,weight_scales,bias,bn=32,qlen=None):
    m,k=q.shape;n=weight_scales.numel();h=n//(3*64);plan=unified_plan(m,bn);p=normalize_plan(m,plan);blocks=64//p['bn']
    qk=torch.empty((2,m,h,64),device=q.device,dtype=torch.float8_e4m3fn)
    sqk=torch.empty((2,m//qlen,h,qlen,blocks) if qlen else (2,m,h,blocks),device=q.device,dtype=torch.float32);v=torch.empty((m,h,64),device=q.device,dtype=torch.float32)
    full_m_qkv_mixed[(n//p['bn'],)](q,weight_col,sx,weight_scales,bias if bias is not None else weight_scales,qk,sqk,v,m,n,k,p['bm'],p['bn'],p['bk'],p['stages'],p['warps'],p['wm'],p['wn'],p['swizzle'],h,64,0 if qlen is None else int(qlen),bias is not None,num_warps=p['warps'])
    return qk,sqk,v
