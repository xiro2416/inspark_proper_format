"""One-launch scatter of concatenated Draft context K/V into request slots."""
import torch
import triton
import triton.language as tl

@triton.jit
def _scatter(K,V,POOL,SRC,LENGTHS,SLOTS,DEST,B:tl.constexpr,L:tl.constexpr,
             S:tl.constexpr,H:tl.constexpr,T:tl.constexpr,C:tl.constexpr,D:tl.constexpr,
             MAX_COMMIT:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(0);part=tl.program_id(1);i=part*BLOCK+tl.arange(0,BLOCK)
    request=group%B;kv=(group//B)%2;layer=group//(B*2)
    token=i//(H*D);hd=i%(H*D);head=hd//D;dim=hd%D
    length=tl.load(LENGTHS+request);valid=(token<length)&(head<H)&(dim<D)
    source=tl.load(SRC+request)+token;slot=tl.load(SLOTS+request);dest=tl.load(DEST+request)+token
    source_offset=(((layer*H+head)*T+source)*D+dim)
    value=tl.load(tl.where(kv==0,K,V)+source_offset,mask=valid,other=0.)
    target=(((((layer*2+kv)*S+slot)*H+head)*C+dest)*D+dim)
    tl.store(POOL+target,value,mask=valid)

def scatter(keys,values,pool,source_offsets,lengths,slots,destinations,max_commit=8):
    if keys.shape!=values.shape or keys.ndim!=5 or keys.shape[1]!=1:
        raise ValueError('Expected stacked K/V [layers,1,heads,total,dim]')
    layers,_,heads,total,dim=keys.shape
    if pool.ndim!=6 or pool.shape[:2]!=(layers,2) or pool.shape[3]!=heads or pool.shape[-1]!=dim:
        raise ValueError('Draft pool layout mismatch')
    b=lengths.numel()
    if any(x.numel()!=b for x in (source_offsets,slots,destinations)):
        raise ValueError('Scatter metadata mismatch')
    block=256;parts=triton.cdiv(heads*max_commit*dim,block)
    _scatter[(layers*2*b,parts)](keys,values,pool,source_offsets,lengths,slots,destinations,
        b,layers,pool.shape[2],heads,total,pool.shape[4],dim,max_commit,block,num_warps=4)
