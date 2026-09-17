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
