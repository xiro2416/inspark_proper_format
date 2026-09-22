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
    # The same projections update a canonical K2048 pool and an optional K128
    # TRT mirror. Long-context fallback must never write past the compact arena.
    valid=valid&(source>=0)&(source<T)&(slot>=0)&(slot<S)&(dest>=0)&(dest<C)
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

@triton.jit
def _scatter_layer(K,V,POOL,SRC,LENGTHS,SLOTS,DEST,
                   K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
                   V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
                   LAYER:tl.constexpr,B:tl.constexpr,S:tl.constexpr,H:tl.constexpr,
                   C:tl.constexpr,D:tl.constexpr,MAX_COMMIT:tl.constexpr,BLOCK:tl.constexpr):
    group=tl.program_id(0);part=tl.program_id(1);i=part*BLOCK+tl.arange(0,BLOCK)
    request=group%B;kv=group//B;token=i//(H*D);hd=i%(H*D);head=hd//D;dim=hd%D
    length=tl.load(LENGTHS+request);valid=(token<length)&(head<H)&(dim<D)
    source=tl.load(SRC+request)+token;slot=tl.load(SLOTS+request);dest=tl.load(DEST+request)+token
    valid=valid&(slot>=0)&(slot<S)&(dest>=0)&(dest<C)
    value=tl.load(tl.where(kv==0,K+head*K1+source*K2+dim*K3,V+head*V1+source*V2+dim*V3),mask=valid,other=0.)
    target=(((((LAYER*2+kv)*S+slot)*H+head)*C+dest)*D+dim)
    tl.store(POOL+target,value,mask=valid)

def scatter_layer(key,value,pool,layer,source_offsets,lengths,slots,destinations,max_commit=8):
    if key.shape!=value.shape or key.ndim!=4 or key.shape[0]!=1:raise ValueError('Expected K/V [1,heads,total,dim]')
    _,heads,_,dim=key.shape;b=lengths.numel();block=256;parts=triton.cdiv(heads*max_commit*dim,block)
    _scatter_layer[(2*b,parts)](key,value,pool,source_offsets,lengths,slots,destinations,
        *key.stride(),*value.stride(),int(layer),b,pool.shape[2],heads,pool.shape[4],dim,max_commit,block,num_warps=4)

@triton.jit
def _scatter_layer_k8_v32(K,V,PK,SK,PV,SRC,LENGTHS,SLOTS,DEST,
                          K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
                          V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
                          B:tl.constexpr,H:tl.constexpr,CAP:tl.constexpr,D:tl.constexpr,
                          MAX_COMMIT:tl.constexpr):
    index=tl.program_id(0);request=index//(H*MAX_COMMIT);head=(index//MAX_COMMIT)%H;token=index%MAX_COMMIT;dim=tl.arange(0,D)
    length=tl.load(LENGTHS+request);valid=token<length;source=tl.load(SRC+request)+token;slot=tl.load(SLOTS+request);dest=tl.load(DEST+request)+token
    k=tl.load(K+head*K1+source*K2+dim*K3,valid,0.).to(tl.float32);v=tl.load(V+head*V1+source*V2+dim*V3,valid,0.)
    blocks=k.reshape((2,32));scale=tl.maximum(tl.max(tl.abs(blocks),1)/448.,1e-12);target=((slot*H+head)*CAP+dest)*D+dim
    tl.store(PK+target,(blocks/scale[:,None]).reshape((D,)),valid);tl.store(PV+target,v,valid)
    part=tl.arange(0,2);tl.store(SK+((slot*H+head)*CAP+dest)*2+part,scale,valid)

def scatter_layer_k8_v32(key,value,pk,sk,pv,source_offsets,lengths,slots,destinations,max_commit=8):
    if key.shape!=value.shape or key.shape[0]!=1 or key.shape[-1]!=64:raise ValueError('Expected [1,H,T,64]')
    _,h,_,d=key.shape;b=lengths.numel()
    _scatter_layer_k8_v32[(b*h*max_commit,)](key,value,pk,sk,pv,source_offsets,lengths,slots,destinations,
        key.stride(1),key.stride(2),key.stride(3),value.stride(1),value.stride(2),value.stride(3),b,h,pk.shape[-2],d,max_commit,num_warps=2)
