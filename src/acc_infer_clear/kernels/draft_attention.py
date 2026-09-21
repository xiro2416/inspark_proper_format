"""Non-causal Q7 attention to request-owned context plus7 query keys/values."""
import torch
import triton
import triton.language as tl

@triton.jit
def _draft(Q,K,V,CK,CV,SLOTS,LENGTHS,O,
           S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,
           H:tl.constexpr,D:tl.constexpr,CAP:tl.constexpr,LIMIT:tl.constexpr,BN:tl.constexpr,
           CONSUMER_LAYOUT:tl.constexpr):
    row=tl.program_id(0);h=tl.program_id(1);slot=tl.load(SLOTS+row);length=tl.load(LENGTHS+row)
    qi=tl.arange(0,16);di=tl.arange(0,D);ki=tl.arange(0,BN)
    q=tl.load(Q+row*S0+h*S1+qi[:,None]*S2+di[None,:]*S3,qi[:,None]<7,0.)
    mx=tl.full((16,),float('-inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,D),tl.float32)
    for start in range(0,tl.minimum(length+7,LIMIT),BN):
        pos=start+ki;valid=pos<length+7
        ck=tl.load(CK+((slot*H+h)*CAP+pos[None,:])*D+di[:,None],pos[None,:]<length,0.)
        nk=tl.load(K+row*S0+h*S1+(pos[None,:]-length)*S2+di[:,None]*S3,(pos[None,:]>=length)&valid[None,:],0.)
        score=tl.dot(q,ck+nk,input_precision='tf32x3')*(D**-.5)
        score=tl.where(valid[None,:],score,float('-inf'))
        nxt=tl.maximum(mx,tl.max(score,1));alpha=tl.exp(mx-nxt);p=tl.exp(score-nxt[:,None])
        cv=tl.load(CV+((slot*H+h)*CAP+pos[:,None])*D+di[None,:],pos[:,None]<length,0.)
        nv=tl.load(V+row*S0+h*S1+(pos[:,None]-length)*S2+di[None,:]*S3,(pos[:,None]>=length)&valid[:,None],0.)
        acc=acc*alpha[:,None]+tl.dot(p,cv+nv,input_precision='tf32x3')
        den=den*alpha+tl.sum(p,1);mx=nxt
    if CONSUMER_LAYOUT:offset=((row*7+qi[:,None])*H+h)*D+di[None,:]
    else:offset=((row*H+h)*7+qi[:,None])*D+di[None,:]
    tl.store(O+offset,acc/den[:,None],qi[:,None]<7)

def attention(q,k,v,ck,cv,slots,lengths,limit,consumer_layout=False):
    b,h,n,d=q.shape
    assert n==7 and d==64 and q.stride()==k.stride()==v.stride()
    out=torch.empty((b,n,h,d) if consumer_layout else (b,h,n,d),device=q.device,dtype=q.dtype)
    _draft[(b,h)](q,k,v,ck,cv,slots,lengths,out,*q.stride(),h,d,ck.shape[-2],limit,64,consumer_layout,num_warps=4,num_stages=2)
    return out
