"""One masked BF16 append write, preserving the provider's clamped-index policy.

Provider indices increase across Q then repeat capacity-1. Only valid append positions write. Clamped invalid positions never write,
avoiding the original CUDA scatter race at K-1. Valid provider positions are
unique by construction; rejected speculative positions are gated by real KV
lengths on subsequent reads.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _write(CACHE, APPEND, INDEX, VALID, B:tl.constexpr,H:tl.constexpr,
           K:tl.constexpr,Q:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,
           IB:tl.constexpr,IQ:tl.constexpr,VB:tl.constexpr,VQ:tl.constexpr,
           BLOCK:tl.constexpr):
    o=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=o%D;q=o//D%Q;h=o//(D*Q)%H;b=o//(D*Q*H)%B;p=o//(D*Q*H*B)
    pos=tl.load(INDEX+b*IB+q*IQ,mask=o<TOTAL,other=0)
    valid=tl.load(VALID+b*VB+q*VQ,mask=o<TOTAL,other=0)
    keep=(o<TOTAL)&valid
    value=tl.load(APPEND+o,mask=keep,other=0)
    dest=(((p*B+b)*H+h)*K+pos)*D+d
    tl.store(CACHE+dest,value,mask=keep)


def write_target_kv(cache,append,indices,valid):
    if (cache.ndim!=6 or not cache.is_contiguous() or not append.is_contiguous()
            or cache.dtype!=torch.bfloat16 or append.dtype!=cache.dtype
            or cache.device!=append.device or cache.device.type!='cuda'):
        raise ValueError('Fused Target KV needs contiguous CUDA BF16 slabs')
    layers,planes,b,h,k,d=cache.shape;q=append.shape[-2]
    if tuple(append.shape)!=(layers,planes,b,h,q,d) or tuple(valid.shape)!=(b,q):
        raise ValueError('Target KV dimensions differ')
    if (tuple(indices.shape)!=(b,h,q,d) or indices.stride(1)!=0 or indices.stride(3)!=0
            or indices.dtype!=torch.int64 or valid.dtype!=torch.bool
            or indices.device!=cache.device or valid.device!=cache.device):
        raise ValueError('Expected broadcast per-request monotone Q positions')
    _write[(triton.cdiv(append.numel(),1024),)](cache,append,indices,valid,b,h,k,q,d,append.numel(),
        indices.stride(0),indices.stride(2),valid.stride(0),valid.stride(1),BLOCK=1024,num_warps=4)
