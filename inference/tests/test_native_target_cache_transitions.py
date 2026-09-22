"""CPU sentinels for host/device entry cache ownership; no CUDA/TRT execution."""
from types import SimpleNamespace as NS

import pytest
import torch

from acc_infer_clear.ops.tensorrt.native113 import NativeTargetFullBank113
from acc_infer_clear.runtime.indextts2.device_round import DeviceRoundHead
from acc_infer_clear.runtime.indextts2.slot_target import SlotKV,SlotTarget
from acc_infer_clear.runtime.pool import _engine_stats


def fixture():
    pool=SlotTarget.__new__(SlotTarget)
    pool.capacity=256;pool.max_slots=16;pool.max_batch=8
    pool.storage=torch.full((1,2,16,1,256,2),-123.)
    pool.keep=torch.zeros(16,256,dtype=torch.int32)
    pool.free=list(reversed(range(16)));pool.generations=[0]*16;pool.leased=[False]*16
    pool.native_attention=None;pool.graph_sealed=True;pool.graphs={};pool.graph_hits=0
    pool.native_full_steps=0;pool.calls=pool.rows=0;pool.shapes={};pool.generic_calls=0
    bank=NativeTargetFullBank113.__new__(NativeTargetFullBank113)
    bank.target=pool;bank.backends={};bank.graphs={};pool.native_full_bank=bank
    for batch in (1,4,8):
        backend=NS(cache=torch.full((1,2,16,1,128,2),-99.,dtype=torch.bfloat16),
                   imports=[],calls=0,outputs=tuple(torch.empty(batch,8,2) for _ in range(3)))
        def importer(packed,row,slot,length,backend=backend):
            backend.imports.append((row,slot,length))
            backend.cache[:,:,slot,:,:length].copy_(packed[:,:,row,:,:length])
        backend.import_slot=importer
        def graph(x,slots,lengths,backend=backend,batch=batch):
            backend.calls+=1
            for slot,length in zip(slots.tolist(),lengths.tolist()):
                assert torch.equal(backend.cache[:,:,slot,:,:length],pool.storage[:,:,slot,:,:length].bfloat16())
                backend.cache[:,:,slot,:,length:length+8].fill_(batch+slot+backend.calls)
            for index,out in enumerate(backend.outputs):out.fill_(100*backend.calls+10*batch+index)
            return backend.outputs
        bank.backends[batch]=backend;bank.graphs[batch,128]=graph
    def generic(x,slots,lengths,limit):
        pool.generic_calls+=1
        for slot,length in zip(slots.tolist(),lengths.tolist()):
            pool.storage[:,:,slot,:,length:length+8].fill_(73.25+slot)
        return x+1,x+2,x+3
    pool.math=generic
    caches=[]
    for slot in range(8):
        length=16+slot
        kv=pool.import_cache(torch.full((1,2,1,1,length,2),slot+11.1251),0,length)
        pool.keep[kv.slot,:length].fill_(1);caches.append(kv)
    for backend in bank.backends.values():backend.imports.clear()
    return pool,caches


def jobs(caches):
    return [(torch.zeros(1,8,2),kv,torch.ones(1,kv.length+8),None) for kv in caches]


def update(caches,outputs,accepted=3):
    for index,out in enumerate(outputs):
        out[1].crop(caches[index].length+accepted);caches[index]=out[1]


def test_batch_switch_refreshes_selected_mirror_and_preserves_all_output_ownership():
    pool,caches=fixture();retained=[]
    for batch in (8,4,1,4):
        inactive=pool.storage[:,:,batch:].clone();tail=pool.storage[:,:,:,:,128:].clone()
        other={b:backend.cache.clone() for b,backend in pool.native_full_bank.backends.items() if b!=batch}
        outputs=pool(jobs(caches[:batch]))
        for row in outputs:
            for index in (0,2,3):retained.append((row[index],row[index].clone()))
        update(caches,outputs)
        assert torch.equal(pool.storage[:,:,batch:],inactive)
        assert torch.equal(pool.storage[:,:,:,:,128:],tail)
        assert all(torch.equal(pool.native_full_bank.backends[b].cache,value) for b,value in other.items())
        assert all(torch.equal(value,saved) for value,saved in retained)
    assert pool.native_full_steps==pool.calls==4 and pool.generic_calls==0


def test_nonidentity_fallback_updates_canonical_then_native_reimports_accepted_prefix():
    pool,caches=fixture();update(caches,pool(jobs(caches[:4])))
    stale=pool.native_full_bank.backends[4].cache.clone()
    indices=(2,0,3,1);reordered=[caches[index] for index in indices]
    outputs=pool(jobs(reordered));update(reordered,outputs)
    for index,kv in zip(indices,reordered):caches[index]=kv
    assert pool.generic_calls==1
    assert torch.equal(pool.native_full_bank.backends[4].cache,stale)
    pool(jobs(caches[:4]))  # The simulated graph asserts all valid imports.
    assert pool.native_full_steps==2 and pool.calls==3


def test_cancel_reuse_refreshes_every_mirror_and_rejects_old_generation():
    pool,caches=fixture();update(caches,pool(jobs(caches[:4])))
    old=SlotKV(pool,0,caches[0].length,caches[0].generation)
    pool.release(caches[0])
    replacement=pool.import_cache(torch.full((1,2,1,1,5,2),41.75),0,5)
    assert replacement.slot==0 and replacement.generation!=old.generation
    caches[0]=replacement
    for backend in pool.native_full_bank.backends.values():
        assert torch.equal(backend.cache[:,:,0,:,:5],torch.full_like(backend.cache[:,:,0,:,:5],41.75))
    before=pool.storage.clone()
    with pytest.raises(RuntimeError,match='Stale'):pool(jobs([old]))
    assert torch.equal(pool.storage,before)
    pool(jobs(caches[:1]));pool(jobs(caches[:4]))
    replacement.check()


def test_long_fallback_consumes_last_native_kv_and_keeps_mirror_out_of_scope():
    pool,caches=fixture();caches[0].length=120
    pool.storage[:,:,0,:,:120].fill_(19.25)
    output=pool(jobs(caches[:1]))[0];output[1].crop(123)
    prefix=pool.storage[:,:,0,:,:123].clone();mirror=pool.native_full_bank.backends[1].cache.clone()
    pool(jobs([output[1]]))
    assert pool.generic_calls==1 and pool.native_full_steps==1
    assert torch.equal(pool.storage[:,:,0,:,:123],prefix)
    assert torch.equal(pool.native_full_bank.backends[1].cache,mirror)


def device_runtime(pool,caches):
    batch=len(caches)
    rows=[NS(kv=kv,past_length=kv.length,cache=NS(length=kv.length,pool_slot=kv.slot),
             codes=[torch.tensor([1])],mel_length=0,done=False) for kv in caches]
    pool.graphs[batch,128]=lambda *args:None
    runtime=NS(device=torch.device('cpu'),target=pool,accept=NS(fused=True),
        residual=NS(device_failures=torch.tensor(0)),
        context=NS(graph_scatter=True,graphs={batch*8:None},pool=NS(max_slots=16)),
        backbone=NS(graphs={(batch,128):None},native_eligible=lambda *args:False),
        proposal=NS(graphs={batch:None}),engine=NS(target=NS(gpt=NS(stop_mel_token=8193))))
    return runtime,rows


@pytest.mark.parametrize('batch',[1,4,8])
def test_host_to_device_entry_refreshes_only_selected_valid_prefix_without_rng_draws(batch):
    pool,caches=fixture();update(caches,pool(jobs(caches[:8])))
    # Another engine or a generic host call can leave this backend stale.
    backend=pool.native_full_bank.backends[batch];backend.cache.fill_(-87.)
    for kv in caches[:batch]:pool.storage[:,:,kv.slot,:,:kv.length].fill_(kv.slot+37.1251)
    others={b:value.cache.clone() for b,value in pool.native_full_bank.backends.items() if b!=batch}
    backend.imports.clear();runtime,rows=device_runtime(pool,caches[:batch]);canonical=pool.storage.clone()
    runner=DeviceRoundHead(runtime,rows,100)
    assert runner.native_target is pool.native_full_bank
    assert backend.imports==[(kv.slot,kv.slot,kv.length) for kv in caches[:batch]]
    for kv in caches[:batch]:
        assert torch.equal(backend.cache[:,:,kv.slot,:,:kv.length],canonical[:,:,kv.slot,:,:kv.length].bfloat16())
        assert (backend.cache[:,:,kv.slot,:,kv.length:]==-87).all()
    assert torch.equal(pool.storage,canonical)
    assert all(torch.equal(pool.native_full_bank.backends[b].cache,value) for b,value in others.items())
    assert torch.equal(runner.generator.get_state(),torch.Generator().manual_seed(0xD3C0A117).get_state())
    assert not torch.cuda.is_initialized()


def test_pool_counts_add_host_and_device_without_counting_preparation_or_fallback_as_native():
    pool,caches=fixture();pool(jobs(caches[:4]));pool(jobs([caches[1]]));pool(jobs(caches[:1]))
    engine=NS(rt=NS(target=pool,proposal=NS(calls=3),backbone=NS(calls=3,device_steps=5,native_full_steps=6),
                   native_target_steps=4,device_target_steps=5),
        sessions={},config={'max_batch':8},device_round_attempts=1,device_round_successes=1,device_round_fallbacks=0)
    result=_engine_stats(engine)
    assert result['target_calls']==3 and result['device_target_steps']==5
    assert result['native_target_steps']==6  # Two host successes plus four device steps.
    assert result['native_target_steps']<=result['target_calls']+result['device_target_steps']
    assert result['target_batch_counts']=={'4':1,'1':2}
