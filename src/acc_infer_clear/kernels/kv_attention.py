"""Request-slot causal verification attention; no packed history materialization."""
import torch
import triton
import triton.language as tl

@triton.jit
def _append(K,V,PK,PV,SLOTS,LENGTHS,
            K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
            V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
            H:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);idx=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    d=idx%D;q=(idx//D)%Q;h=idx//(D*Q);valid=idx<H*Q*D
    k=tl.load(K+row*K0+h*K1+q*K2+d*K3,valid,0)
    v=tl.load(V+row*V0+h*V1+q*V2+d*V3,valid,0)
    dest=((slot*H+h)*CAP+length+q)*D+d
    tl.store(PK+dest,k,valid);tl.store(PV+dest,v,valid)

@triton.jit
def _attention(QT,PK,PV,KEEP,SLOTS,LENGTHS,OUT,
               Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
               H:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,
               LIMIT:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    qi=tl.arange(0,16);di=tl.arange(0,D);ki=tl.arange(0,BN)
    q=tl.load(QT+row*Q0+head*Q1+qi[:,None]*Q2+di[None,:]*Q3,qi[:,None]<Q,0)
    mx=tl.full((16,),float('-inf'),tl.float32);den=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,D),tl.float32)
    # Device length selects actual iterations; graph launch and addresses stay fixed.
    for start in range(0,tl.minimum(length+Q,LIMIT),BN):
        pos=start+ki
        valid=pos<length+Q
        keep=tl.load(KEEP+slot*CAP+pos,valid,0)>0
        base=PK+((slot*H+head)*CAP+pos[None,:])*D+di[:,None]
        k=tl.load(base,valid[None,:],0)
        score=tl.dot(q,k,input_precision='tf32x3')*(D**-.5)
        causal=(pos[None,:]<=length+qi[:,None])&keep[None,:]
        score=tl.where(causal,score,float('-inf'))
        nxt=tl.maximum(mx,tl.max(score,1));nxt=tl.where(nxt==float('-inf'),0.,nxt)
        alpha=tl.exp(mx-nxt)
        prob=tl.exp(score-nxt[:,None])
        value=tl.load(PV+((slot*H+head)*CAP+pos[:,None])*D+di[None,:],valid[:,None],0)
        acc=acc*alpha[:,None]+tl.dot(prob.to(value.dtype),value,input_precision='tf32x3')
        den=den*alpha+tl.sum(prob,1);mx=nxt
    result=acc/den[:,None]
    tl.store(OUT+((row*H+head)*Q+qi[:,None])*D+di[None,:],result,qi[:,None]<Q)

def append(k,v,pk,pv,slots,lengths):
    b,h,q,d=k.shape
    _append[(b,triton.cdiv(h*q*d,256))](k,v,pk,pv,slots,lengths,*k.stride(),*v.stride(),h,q,d,pk.shape[-2],256)

def attention(q,pk,pv,keep,slots,lengths,limit):
    b,h,n,d=q.shape
    if d!=64 or n!=8:raise ValueError('Validated specialization is Q8/head_dim64')
    out=torch.empty((b,h,n,d),device=q.device,dtype=q.dtype)
    _attention[(b,h)](q,pk,pv,keep,slots,lengths,out,*q.stride(),h,n,d,pk.shape[-2],limit,64,num_warps=4,num_stages=2)
    return out
