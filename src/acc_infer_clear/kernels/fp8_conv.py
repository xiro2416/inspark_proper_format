"""Native FP8 implicit Conv1d/ConvTranspose1d, without materialized unfold/fold."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['COUNT','PARTS'])
def _partial_max(X,P,COUNT,PARTS,BLOCK:tl.constexpr):
    b=tl.program_id(0);part=tl.program_id(1);i=part*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+b*COUNT+i,i<COUNT,0.).to(tl.float32)
    tl.store(P+b*PARTS+part,tl.max(tl.abs(x),0))

@triton.jit(do_not_specialize=['PARTS'])
def _scale(P,S,PARTS,BLOCK:tl.constexpr):
    b=tl.program_id(0);i=tl.arange(0,BLOCK)
    maximum=tl.max(tl.load(P+b*PARTS+i,i<PARTS,0.),0)
    tl.store(S+b,tl.maximum(maximum/448.,1e-12))

@triton.jit(do_not_specialize=['COUNT'])
def _quant(X,Y,S,COUNT,BLOCK:tl.constexpr):
    b=tl.program_id(0);i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+b*COUNT+i,i<COUNT,0.).to(tl.float32);scale=tl.load(S+b)
    tl.store(Y+b*COUNT+i,(x/scale).to(Y.dtype.element_ty),i<COUNT)

@triton.jit(do_not_specialize=['T','TO','M'])
def _conv(X,W,SX,SW,BIAS,Y,T,TO,M,CI:tl.constexpr,CO:tl.constexpr,KW:tl.constexpr,WS:tl.constexpr,
          STRIDE:tl.constexpr,PAD:tl.constexpr,DIL:tl.constexpr,TRANSPOSE:tl.constexpr,
          HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK);batch=mi//TO;out_t=mi%TO;acc=tl.zeros((BM,BN),tl.float32)
    for part in range(tl.cdiv(CI*KW,BK)):
        reduction=part*BK+kk;channel=reduction//KW;tap=reduction%KW
        if TRANSPOSE:
            numer=out_t[:,None]+PAD-tap[None,:]*DIL
            time=numer//STRIDE;aligned=numer%STRIDE==0
        else:
            time=out_t[:,None]*STRIDE-PAD+tap[None,:]*DIL;aligned=tl.full((BM,BK),True,tl.int1)
        valid=(mi[:,None]<M)&(reduction[None,:]<CI*KW)&(time>=0)&(time<T)&aligned
        a=tl.load(X+(batch[:,None]*CI+channel[None,:])*T+time,valid,0.)
        w=tl.load(W+reduction[:,None]*WS+ni[None,:],(reduction[:,None]<CI*KW)&(ni[None,:]<CO),0.)
        acc=tl.dot(a,w,acc)
    result=acc*tl.load(SX+batch,mi<M,0.)[:,None]*tl.load(SW+ni,ni<CO,0.)[None,:]
    if HAS_BIAS:result+=tl.load(BIAS+ni,ni<CO,0.)[None,:]
    tl.store(Y+(batch[:,None]*CO+ni[None,:])*TO+out_t[:,None],result,(mi[:,None]<M)&(ni[None,:]<CO))

def conv1d(x,weight,scales,bias,ci,co,kw,stride,padding,dilation,transpose,output_padding,tile):
    x=x.contiguous();b,channels,t=x.shape
    if channels!=ci:raise ValueError('Input channels changed')
    out_t=(t-1)*stride-2*padding+dilation*(kw-1)+output_padding+1 if transpose else (t+2*padding-dilation*(kw-1)-1)//stride+1
    if out_t<=0:raise ValueError('Non-positive output extent')
    count=ci*t;parts=triton.cdiv(count,1024)
    partial=torch.empty(b,parts,device=x.device,dtype=torch.float32)
    sx=torch.empty(b,device=x.device,dtype=torch.float32);q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    _partial_max[(b,parts)](x,partial,count,parts,1024)
    scale_block=triton.next_power_of_2(parts)
    _scale[(b,)](partial,sx,parts,scale_block,num_warps=min(16,max(4,scale_block//2048)))
    _quant[(b,parts)](x,q,sx,count,1024)
    y=torch.empty(b,co,out_t,device=x.device,dtype=x.dtype)
    _conv[(triton.cdiv(b*out_t,tile.bm),triton.cdiv(co,tile.bn))](
        q,weight,sx,scales,bias if bias is not None else scales,y,t,out_t,b*out_t,ci,co,kw,weight.shape[1],
        stride,padding,dilation,transpose,bias is not None,tile.bm,tile.bn,tile.bk,
        num_warps=tile.warps,num_stages=tile.stages)
    return y

def prepare_scale_kernels(device):
    """All reduction bucket sizes within current model bounds, before serving."""
    for power in range(17):
        parts=2**power
        partial=torch.zeros(1,parts,device=device);scale=torch.empty(1,device=device)
        _scale[(1,)](partial,scale,parts,parts,num_warps=min(16,max(4,parts//2048)))
