"""Fuse normalization, sampled group mass, acceptance and compact decisions."""
import torch
import triton
import triton.language as tl

@triton.jit
def _accept(L,P,T,GD,AD,COUNTS,TGROUPS,MEMBERS,WEIGHTS,SIZES,Q,A,PACK,
            V:tl.constexpr,TG:tl.constexpr,GM:tl.constexpr,TEMP:tl.constexpr,
            BV:tl.constexpr,BG:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BV)
    logits=tl.load(L+row*V+i,i<V,float('-inf')).to(tl.float32)/TEMP
    mx=tl.max(logits,0);exp=tl.exp(logits-mx);den=tl.sum(exp,0)
    q=exp/den
    p=tl.load(P+row*V+i,i<V,0.).to(tl.float32);ps=tl.maximum(tl.sum(p,0),1e-12)
    token=tl.load(T+row);count=tl.load(COUNTS+token)
    choice=tl.floor(tl.load(GD+row)*count).to(tl.int32)
    group=tl.load(TGROUPS+token*TG+choice)
    j=tl.arange(0,BG);members=tl.load(MEMBERS+group*GM+j,j<GM,0)
    weights=tl.load(WEIGHTS+group*GM+j,j<GM,0.)
    gl=tl.load(L+row*V+members)/TEMP
    gp=tl.load(P+row*V+members)/ps
    qm=tl.sum(tl.exp(gl-mx)/den*weights,0);pm=tl.sum(gp*weights,0)
    accept=tl.minimum(qm/tl.maximum(pm,1e-12),1.)
    exact_q=tl.exp(tl.load(L+row*V+token)/TEMP-mx)/den
    exact_p=tl.load(P+row*V+token)/ps
    exact=tl.minimum(exact_q/tl.maximum(exact_p,1e-12),1.)
    flag=tl.load(AD+row)<accept;size=tl.load(SIZES+group)
    tl.store(Q+row*V+i,q,i<V);tl.store(A+row,accept)
    col=tl.arange(0,8)
    value=tl.where(col==0,flag.to(tl.float32),tl.where(col==1,token.to(tl.float32),tl.where(col==2,size.to(tl.float32),tl.where(col==3,accept,exact))))
    tl.store(PACK+row*5+col,value,col<5)

def acceptance(groups,temperature,logits,p,tokens,group_draws,accept_draws):
    logits=logits.contiguous();p=p.contiguous();tokens=tokens.contiguous()
    b,k,v=logits.shape;gm=groups.group_members.shape[1]
    q=torch.empty_like(logits,dtype=torch.float32);a=torch.empty((b,k),device=p.device,dtype=torch.float32)
    packed=torch.empty((b,k,5),device=p.device,dtype=torch.float32)
    _accept[(b*k,)](logits,p,tokens,group_draws,accept_draws,groups.token_group_counts,
                    groups.token_groups,groups.group_members,groups.group_weights,groups.group_sizes,
                    q,a,packed,v,groups.token_groups.shape[1],gm,temperature,triton.next_power_of_2(v),
                    triton.next_power_of_2(gm),num_warps=8)
    return q,a,packed

@triton.jit
def _prefix_plan(PACK,REMAINING,CURRENT,COUNTS,ENDS,CORRECTIONS,RESIDUALS,
                 K:tl.constexpr,EOS:tl.constexpr,MAX_TOKENS:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,8);valid=(j<K)&(j<tl.load(REMAINING+row))
    flag=tl.load(PACK+(row*K+j)*5,mask=j<K,other=0.)!=0
    token=tl.load(PACK+(row*K+j)*5+1,mask=j<K,other=0.).to(tl.int32)
    stop=valid&((~flag)|(token==EOS));first=tl.min(tl.where(stop,j,K),axis=0)
    has=first<K;sf=tl.sum(tl.where(j==first,flag.to(tl.int32),0),axis=0)!=0
    st=tl.sum(tl.where(j==first,token,0),axis=0);end=has&sf&(st==EOS)
    remaining=tl.load(REMAINING+row);n=tl.where(has,first+end.to(tl.int32),remaining)
    correction=(~end)&((tl.load(CURRENT+row)+n)<MAX_TOKENS)
    tl.store(COUNTS+row,n);tl.store(ENDS+row,end)
    tl.store(CORRECTIONS+row,correction);tl.store(RESIDUALS+row,correction&(n<remaining))

def prefix_plan(packed,remaining,current,eos,max_tokens):
    """Compact device commit plan; packed acceptance decisions never leave GPU."""
    if packed.ndim!=3 or packed.shape[1]!=7 or packed.shape[2]!=5:
        raise ValueError('Expected packed [B,7,5] acceptance decisions')
    b=packed.shape[0];device=packed.device
    remaining=remaining.to(device=device,dtype=torch.int32).contiguous()
    current=current.to(device=device,dtype=torch.int32).contiguous()
    counts=torch.empty(b,device=device,dtype=torch.int32)
    ends=torch.empty(b,device=device,dtype=torch.bool)
    corrections=torch.empty(b,device=device,dtype=torch.bool)
    residuals=torch.empty(b,device=device,dtype=torch.bool)
    _prefix_plan[(b,)](packed,remaining,current,counts,ends,corrections,residuals,
                       7,int(eos),int(max_tokens),num_warps=1)
    return counts,ends,corrections,residuals
