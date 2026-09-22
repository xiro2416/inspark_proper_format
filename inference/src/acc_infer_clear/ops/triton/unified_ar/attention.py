"""Unified FP8 Q/K attention with FP32 Softmax, V cache and P×V.

Q/K/V use E4M3 with one FP32 dequant scale per token/head vector. QK and PV
use native FP8 dot products with FP32 accumulators. The online softmax state is
always FP32. MODE=1 keeps PV on the original FP32 path; MODE=2 quantizes each
softmax tile after folding the per-token V scales into its probabilities.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _target_fp8(Q, SQ, PK, SK, PV, SV, KEEP, SLOTS, LENGTHS, OUT,
                Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
                H:tl.constexpr,NQ:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,
                LIMIT:tl.constexpr,BN:tl.constexpr,MODE:tl.constexpr,
                CONSUMER_LAYOUT:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    qi=tl.arange(0,16);di=tl.arange(0,D);ki=tl.arange(0,BN)
    q=tl.load(Q+row*Q0+head*Q1+qi[:,None]*Q2+di[None,:]*Q3,qi[:,None]<NQ,0.)
    sq=tl.load(SQ+(row*H+head)*NQ+qi,qi<NQ,1.)
    mx=tl.full((16,),float('-inf'),tl.float32);den=tl.zeros((16,),tl.float32)
    acc=tl.zeros((16,D),tl.float32)
    for start in range(0,tl.minimum(length+NQ,LIMIT),BN):
        pos=start+ki;valid=pos<length+NQ
        keep=tl.load(KEEP+slot*CAP+pos,valid,0)>0
        k=tl.load(PK+((slot*H+head)*CAP+pos[None,:])*D+di[:,None],valid[None,:],0.)
        sk=tl.load(SK+(slot*H+head)*CAP+pos,valid,1.)
        score=tl.dot(q,k)*sq[:,None]*sk[None,:]*(D**-.5)
        causal=(pos[None,:]<=length+qi[:,None])&keep[None,:]
        score=tl.where(causal,score,float('-inf'))
        nxt=tl.maximum(mx,tl.max(score,1));nxt=tl.where(nxt==float('-inf'),0.,nxt)
        alpha=tl.exp(mx-nxt);prob=tl.exp(score-nxt[:,None])
        value=tl.load(PV+((slot*H+head)*CAP+pos[:,None])*D+di[None,:],valid[:,None],0.)
        if MODE==1:
            update=tl.dot(prob.to(value.dtype),value,input_precision='tf32x3')
        elif MODE==3:
            sv=tl.load(SV+(slot*H+head)*CAP+pos,valid,1.)
            value32=value.to(tl.float32)*sv[:,None]
            update=tl.dot(prob,value32,input_precision='tf32x3')
        else:
            sv=tl.load(SV+(slot*H+head)*CAP+pos,valid,1.)
            weighted=prob*sv[None,:]
            ps=tl.maximum(tl.max(tl.abs(weighted),1)/448.,1e-12)
            p8=(weighted/ps[:,None]).to(value.dtype)
            update=tl.dot(p8,value)*ps[:,None]
        acc=acc*alpha[:,None]+update
        den=den*alpha+tl.sum(prob,1);mx=nxt
    result=acc/den[:,None]
    if CONSUMER_LAYOUT:offset=((row*NQ+qi[:,None])*H+head)*D+di[None,:]
    else:offset=((row*H+head)*NQ+qi[:,None])*D+di[None,:]
    tl.store(OUT+offset,result,qi[:,None]<NQ)


def quantize_vectors(x):
    """Quantize the last dimension independently; experimental setup only."""
    scale=x.float().abs().amax(-1).div(448.).clamp_min_(1e-12)
    return (x/scale.unsqueeze(-1)).to(torch.float8_e4m3fn),scale


def quantize_slot_cache(cache,slots,limit):
    """Preserve production slot/capacity layout while quantizing used prefixes."""
    out=torch.zeros(cache.shape,device=cache.device,dtype=torch.float8_e4m3fn)
    scales=torch.ones(cache.shape[:-1],device=cache.device,dtype=torch.float32)
    for slot in slots.detach().cpu().tolist():
        q,s=quantize_vectors(cache[slot,:,:limit])
        out[slot,:,:limit].copy_(q);scales[slot,:,:limit].copy_(s)
    return out,scales


@triton.jit
def _append_fp8(K,V,SK,SV,PK,PV,PSK,PSV,SLOTS,LENGTHS,
                K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
                H:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr):
    index=tl.program_id(0);row=index//(H*Q);head=(index//Q)%H;token=index%Q;dim=tl.arange(0,D)
    slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row);dest=length+token
    base=row*K0+head*K1+token*K2+dim*K3;target=((slot*H+head)*CAP+dest)*D+dim
    tl.store(PK+target,tl.load(K+base));tl.store(PV+target,tl.load(V+base))
    scale_offset=(slot*H+head)*CAP+dest+dim*0
    tl.store(PSK+scale_offset,tl.load(SK+(row*H+head)*Q+token),mask=dim==0)
    tl.store(PSV+scale_offset,tl.load(SV+(row*H+head)*Q+token),mask=dim==0)


def append_fp8(k8,v8,sk,sv,pk8,pv8,psk,psv,slots,lengths):
    b,h,q,d=k8.shape
    _append_fp8[(b*h*q,)](k8,v8,sk,sv,pk8,pv8,psk,psv,slots,lengths,*k8.stride(),h,q,d,pk8.shape[-2],num_warps=1)


@triton.jit
def _append_qk_fp8(K,SK,V,PK,PSK,PV,SLOTS,LENGTHS,
                   K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
                   V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
                   H:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr):
    index=tl.program_id(0);row=index//(H*Q);head=(index//Q)%H;token=index%Q;dim=tl.arange(0,D)
    slot=tl.load(SLOTS+row);dest=tl.load(LENGTHS+row)+token
    koff=row*K0+head*K1+token*K2+dim*K3;voff=row*V0+head*V1+token*V2+dim*V3;target=((slot*H+head)*CAP+dest)*D+dim
    tl.store(PK+target,tl.load(K+koff));tl.store(PV+target,tl.load(V+voff))
    tl.store(PSK+(slot*H+head)*CAP+dest+dim*0,tl.load(SK+(row*H+head)*Q+token),mask=dim==0)


def append_qk_fp8(k8,sk,v,pk8,psk,pv,slots,lengths):
    b,h,q,d=k8.shape
    _append_qk_fp8[(b*h*q,)](k8,sk,v,pk8,psk,pv,slots,lengths,*k8.stride(),*v.stride(),h,q,d,pk8.shape[-2],num_warps=1)


def target_attention(q8,sq,pk8,sk,pv,sv,keep,slots,lengths,limit,mode,
                     consumer_layout=False):
    if mode not in (1,2,3):raise ValueError('mode is 1 (FP8 QK), 2 (FP8 QK+PV), or 3 (FP8 QK + dequant V)')
    b,h,n,d=q8.shape
    if q8.dtype!=torch.float8_e4m3fn or pk8.dtype!=torch.float8_e4m3fn:
        raise ValueError('Q/K must be E4M3')
    if mode in (2,3) and pv.dtype!=torch.float8_e4m3fn:raise ValueError('Quantized V modes require E4M3 V')
    if mode==1 and pv.dtype!=torch.float32:raise ValueError('QK-only control expects FP32 V')
    out=torch.empty((b,n,h,d) if consumer_layout else (b,h,n,d),device=q8.device,dtype=torch.float32)
    _target_fp8[(b,h)](q8,sq,pk8,sk,pv,sv,keep,slots,lengths,out,*q8.stride(),
        h,n,d,pk8.shape[-2],limit,64,mode,consumer_layout,num_warps=4,num_stages=1)
    return out


@triton.jit
def _target_fp8_block32(Q,SQ,PK,SK,PV,KEEP,SLOTS,LENGTHS,OUT,
                        Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
                        H:tl.constexpr,NQ:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,
                        LIMIT:tl.constexpr,BN:tl.constexpr,CONSUMER_LAYOUT:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1);slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    qi=tl.arange(0,16);di=tl.arange(0,32);ki=tl.arange(0,BN)
    q0=tl.load(Q+row*Q0+head*Q1+qi[:,None]*Q2+di[None,:]*Q3,qi[:,None]<NQ,0.)
    q1=tl.load(Q+row*Q0+head*Q1+qi[:,None]*Q2+(di[None,:]+32)*Q3,qi[:,None]<NQ,0.)
    sq0=tl.load(SQ+((row*H+head)*NQ+qi)*2,qi<NQ,1.);sq1=tl.load(SQ+((row*H+head)*NQ+qi)*2+1,qi<NQ,1.)
    mx=tl.full((16,),float('-inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32);od=tl.arange(0,D)
    for start in range(0,tl.minimum(length+NQ,LIMIT),BN):
        pos=start+ki;valid=pos<length+NQ;keep=tl.load(KEEP+slot*CAP+pos,valid,0)>0
        k0=tl.load(PK+((slot*H+head)*CAP+pos[None,:])*D+di[:,None],valid[None,:],0.)
        k1=tl.load(PK+((slot*H+head)*CAP+pos[None,:])*D+(di[:,None]+32),valid[None,:],0.)
        sk0=tl.load(SK+((slot*H+head)*CAP+pos)*2,valid,1.);sk1=tl.load(SK+((slot*H+head)*CAP+pos)*2+1,valid,1.)
        score=(tl.dot(q0,k0)*sq0[:,None]*sk0[None,:]+tl.dot(q1,k1)*sq1[:,None]*sk1[None,:])*(D**-.5)
        score=tl.where((pos[None,:]<=length+qi[:,None])&keep[None,:],score,float('-inf'))
        nxt=tl.maximum(mx,tl.max(score,1));nxt=tl.where(nxt==float('-inf'),0.,nxt);alpha=tl.exp(mx-nxt);prob=tl.exp(score-nxt[:,None])
        value=tl.load(PV+((slot*H+head)*CAP+pos[:,None])*D+od[None,:],valid[:,None],0.)
        acc=acc*alpha[:,None]+tl.dot(prob,value,input_precision='tf32x3');den=den*alpha+tl.sum(prob,1);mx=nxt
    if CONSUMER_LAYOUT:offset=((row*NQ+qi[:,None])*H+head)*D+od[None,:]
    else:offset=((row*H+head)*NQ+qi[:,None])*D+od[None,:]
    tl.store(OUT+offset,acc/den[:,None],qi[:,None]<NQ)


def quantize_blocks32(x):
    shape=x.shape;blocks=x.reshape(*shape[:-1],2,32);scale=blocks.float().abs().amax(-1).div(448.).clamp_min_(1e-12)
    return (blocks/scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(shape),scale


def quantize_slot_cache_blocks32(cache,slots,limit):
    out=torch.zeros(cache.shape,device=cache.device,dtype=torch.float8_e4m3fn);scales=torch.ones((*cache.shape[:-1],2),device=cache.device,dtype=torch.float32)
    for slot in slots.detach().cpu().tolist():
        q,s=quantize_blocks32(cache[slot,:,:limit]);out[slot,:,:limit].copy_(q);scales[slot,:,:limit].copy_(s)
    return out,scales


def target_attention_block32(q8,sq,pk8,sk,pv,keep,slots,lengths,limit,consumer_layout=False):
    b,h,n,d=q8.shape;out=torch.empty((b,n,h,d) if consumer_layout else (b,h,n,d),device=q8.device,dtype=torch.float32)
    _target_fp8_block32[(b,h)](q8,sq,pk8,sk,pv,keep,slots,lengths,out,*q8.stride(),h,n,d,pk8.shape[-2],limit,64,consumer_layout,num_warps=4,num_stages=1)
    return out


@triton.jit
def _draft_fp8_block32(Q,SQ,K,SK,V,CK,SCK,CV,SLOTS,LENGTHS,OUT,
                       S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,
                       H:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,LIMIT:tl.constexpr,
                       BN:tl.constexpr,CONSUMER_LAYOUT:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1);slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    qi=tl.arange(0,16);di=tl.arange(0,32);ki=tl.arange(0,BN);od=tl.arange(0,D)
    q0=tl.load(Q+row*S0+head*S1+qi[:,None]*S2+di[None,:]*S3,qi[:,None]<7,0.)
    q1=tl.load(Q+row*S0+head*S1+qi[:,None]*S2+(di[None,:]+32)*S3,qi[:,None]<7,0.)
    sq0=tl.load(SQ+((row*H+head)*7+qi)*2,qi<7,1.);sq1=tl.load(SQ+((row*H+head)*7+qi)*2+1,qi<7,1.)
    mx=tl.full((16,),float('-inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for start in range(0,tl.minimum(length+7,LIMIT),BN):
        pos=start+ki;valid=pos<length+7;history=pos<length;fresh=pos-length
        ck0=tl.load(CK+((slot*H+head)*CAP+pos[None,:])*D+di[:,None],history[None,:],0.)
        ck1=tl.load(CK+((slot*H+head)*CAP+pos[None,:])*D+(di[:,None]+32),history[None,:],0.)
        nk0=tl.load(K+row*S0+head*S1+fresh[None,:]*S2+di[:,None]*S3,(~history)[None,:]&valid[None,:],0.)
        nk1=tl.load(K+row*S0+head*S1+fresh[None,:]*S2+(di[:,None]+32)*S3,(~history)[None,:]&valid[None,:],0.)
        csk0=tl.load(SCK+((slot*H+head)*CAP+pos)*2,history,1.);csk1=tl.load(SCK+((slot*H+head)*CAP+pos)*2+1,history,1.)
        nsk0=tl.load(SK+((row*H+head)*7+fresh)*2,(~history)&valid,1.);nsk1=tl.load(SK+((row*H+head)*7+fresh)*2+1,(~history)&valid,1.)
        kk0=tl.where(history[None,:],ck0,nk0);kk1=tl.where(history[None,:],ck1,nk1);ks0=tl.where(history,csk0,nsk0);ks1=tl.where(history,csk1,nsk1)
        score=(tl.dot(q0,kk0)*sq0[:,None]*ks0[None,:]+tl.dot(q1,kk1)*sq1[:,None]*ks1[None,:])*(D**-.5)
        score=tl.where(valid[None,:],score,float('-inf'));nxt=tl.maximum(mx,tl.max(score,1));alpha=tl.exp(mx-nxt);prob=tl.exp(score-nxt[:,None])
        cv=tl.load(CV+((slot*H+head)*CAP+pos[:,None])*D+od[None,:],history[:,None],0.)
        nv=tl.load(V+row*S0+head*S1+fresh[:,None]*S2+od[None,:]*S3,(~history)[:,None]&valid[:,None],0.)
        value=tl.where(history[:,None],cv,nv)
        acc=acc*alpha[:,None]+tl.dot(prob,value,input_precision='tf32x3');den=den*alpha+tl.sum(prob,1);mx=nxt
    if CONSUMER_LAYOUT:offset=((row*7+qi[:,None])*H+head)*D+od[None,:]
    else:offset=((row*H+head)*7+qi[:,None])*D+od[None,:]
    tl.store(OUT+offset,acc/den[:,None],qi[:,None]<7)


def draft_attention_block32(q8,sq,k8,sk,v,ck8,sck,cv,slots,lengths,limit,consumer_layout=False):
    b,h,n,d=q8.shape
    if n!=7 or d!=64:raise ValueError('Draft FP8 specialization is Q7/D64')
    out=torch.empty((b,n,h,d) if consumer_layout else (b,h,n,d),device=q8.device,dtype=torch.float32)
    _draft_fp8_block32[(b,h)](q8,sq,k8,sk,v,ck8,sck,cv,slots,lengths,out,*q8.stride(),h,d,ck8.shape[-2],limit,64,consumer_layout,num_warps=4,num_stages=1)
    return out
