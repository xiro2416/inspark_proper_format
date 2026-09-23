"""Request-local host dispatch must use real native graphs, with safe fallback."""
from types import SimpleNamespace as NS

import pytest
import torch

from inspark_infer.ops.tensorrt.native113 import NativeTargetFullBank113
from inspark_infer.runtime.indextts2.slot_target import SlotKV, SlotTarget


def target_pool(batch=2):
    pool=SlotTarget.__new__(SlotTarget)
    pool.capacity=256;pool.max_slots=4;pool.max_batch=2
    pool.storage=torch.full((1,2,4,1,256,2),9.1251)
    pool.keep=torch.zeros(4,256,dtype=torch.int32)
    pool.generations=[1]*4;pool.leased=[True]*4
    pool.graph_sealed=True;pool.graphs={};pool.graph_hits=0;pool.native_full_steps=0
    pool.calls=pool.rows=0;pool.shapes={}
    pool.generic_calls=0
    def generic(x,slots,lengths,limit):
        pool.generic_calls+=1
        return x+1,x+2,x+3
    pool.math=generic
    bank=NativeTargetFullBank113.__new__(NativeTargetFullBank113)
    bank.target=pool;bank.graphs={};bank.backends={}
    mirror=torch.full((1,2,4,1,128,2),-100.,dtype=torch.bfloat16)
    imports=[]
    def import_slot(packed,row,slot,length):
        imports.append((row,slot,length))
        mirror[:,:,slot,:,:length].copy_(packed[:,:,row,:,:length])
    backend=NS(cache=mirror,import_slot=import_slot)
    def graph(x,slots,lengths):
        for row,(slot,length) in enumerate(zip(slots.tolist(),lengths.tolist())):
            assert torch.equal(mirror[:,:,row,:,:length],pool.storage[:,:,slot,:,:length].bfloat16())
            mirror[:,:,row,:,length:length+8].fill_(3.)
        return x+10,x+20,x+30
    bank.backends[batch]=backend;bank.graphs[batch,128]=graph
    pool.native_full_bank=bank
    return pool,mirror,imports


def jobs(pool,slots,lengths):
    return [(torch.zeros(1,8,2),SlotKV(pool,slot,length,1),torch.ones(1,length+8),None)
            for slot,length in zip(slots,lengths)]


@pytest.mark.parametrize('lengths',[[20,40],[120,119]])
def test_native_dispatch_refreshes_and_exports_canonical_cache(lengths):
    pool,mirror,imports=target_pool()
    tail=pool.storage[:,:,:,:,128:].clone()
    result=pool(jobs(pool,[0,1],lengths))
    assert pool.native_full_steps==1 and pool.generic_calls==0
    assert imports==[(0,0,lengths[0]),(1,1,lengths[1])]
    for row,length in enumerate(lengths):
        assert torch.equal(pool.storage[:,:,row,:,:length+8],mirror[:,:,row,:,:length+8].float())
    assert torch.equal(pool.storage[:,:,:,:,128:],tail)
    assert all(torch.equal(row[0],torch.full((1,8,2),10.)) for row in result)
    # A generic/batch-switch call may change valid history: the next native
    # call must not reuse its stale mirror. Returning tensors are request-owned.
    pool.storage[:,:,:2,:,:4].fill_(17.)
    old=result[0][0].clone()
    pool(jobs(pool,[0,1],lengths))
    assert torch.equal(mirror[:,:,:2,:,:4],torch.full_like(mirror[:,:,:2,:,:4],17.))
    assert torch.equal(result[0][0],old)


@pytest.mark.parametrize('slots,lengths',[
    ([1,0],[80,80]),([0,2],[80,80]),([0],[80]),
])
def test_nonidentity_or_short_batch_packs_without_touching_inactive_canonical_slots(slots,lengths):
    pool,mirror,imports=target_pool()
    before=pool.storage.clone()
    pool(jobs(pool,slots,lengths))
    assert pool.native_full_steps==1 and pool.generic_calls==0
    assert imports==[(slot,row,length) for row,(slot,length) in enumerate(zip(slots,lengths))]
    for slot in set(range(pool.max_slots))-set(slots):
        assert torch.equal(pool.storage[:,:,slot],before[:,:,slot])


def test_long_batch_uses_generic():
    pool,mirror,imports=target_pool()
    before=mirror.clone()
    pool(jobs(pool,[0,1],[121,80]))
    assert pool.native_full_steps==0 and pool.generic_calls==1
    assert imports==[] and torch.equal(mirror,before)


def test_stale_lease_fails_before_native_write():
    pool,mirror,imports=target_pool()
    pool.leased[1]=False
    with pytest.raises(RuntimeError,match='Stale'):
        pool(jobs(pool,[0,1],[80,80]))
    assert not imports
