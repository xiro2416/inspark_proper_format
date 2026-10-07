"""Native logical slot arena backed by per-layer contiguous TRT cache storage."""
import torch


def arena(layers,slots,heads,capacity,dimension,*,device,dtype):
    storage=torch.zeros(layers,slots,heads,capacity,dimension,device=device,dtype=dtype)
    # Native worker indexes [slot,layer,position,head,dimension]. Existing torch
    # index_put/copy operators respect strides; no new math kernel is introduced.
    return storage.permute(1,0,3,2,4)


def install(worker,batch,capacity):
    old=worker._ctx_k_buf
    if worker._max_ctx!=capacity or old.shape[0]!=batch+1 or old.shape[2]<capacity:
        raise ValueError('Head arena geometry differs from the fixed context limit')
    # The bridge builds noise K/V inside the Draft engine, never in the native
    # pool. Native committed writes clamp to max_ctx-1. Its +block slack is
    # therefore unconsumed in this fixed x/cache-only forward-provider route.
    slots,layers,_,heads,dimension=old.shape
    worker._ctx_k_buf=arena(layers,slots,heads,capacity,dimension,device=old.device,dtype=old.dtype)
    worker._ctx_v_buf=arena(layers,slots,heads,capacity,dimension,device=old.device,dtype=old.dtype)
    for pool in (worker._ctx_k_buf,worker._ctx_v_buf):
        for layer in range(layers):
            if not pool[:batch,layer].transpose(1,2).is_contiguous():
                raise ValueError('Arena cannot directly bind the static TRT head-major cache')
