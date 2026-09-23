from types import SimpleNamespace
import torch

from inspark_infer.ops.tensorrt.native113 import NativeDraftFullBank113
from inspark_infer.runtime.indextts2.slot_draft import SlotDraft


def test_draft_cache_mirror_is_not_gated_by_committed_token_bucket():
    bank=SimpleNamespace(backends={8:object()})
    assert NativeDraftFullBank113.enabled_for_total(bank,8)
    assert NativeDraftFullBank113.enabled_for_total(bank,16)
    assert NativeDraftFullBank113.enabled_for_total(bank,56)
    assert NativeDraftFullBank113.enabled_for_total(bank,64)


def test_draft_cache_mirror_is_disabled_without_an_engine():
    bank=SimpleNamespace(backends={})
    assert not NativeDraftFullBank113.enabled_for_total(bank,64)


def test_short_kv_bucket_uses_masked_k128_engine_without_truncation():
    used=[]
    graph=lambda anchors,positions,slots,lengths: (
        used.append((slots.tolist(),lengths.tolist())) or torch.zeros(1,7,2),
        torch.zeros(1,7,3))
    draft=SlotDraft.__new__(SlotDraft)
    draft.pool=SimpleNamespace(storage=torch.empty(1),check=lambda cache:None)
    draft.native_full_bank=SimpleNamespace(graphs={(1,128):graph},host_engine_batch=lambda batch,length:1)
    draft.graphs={(1,64):lambda *args: (_ for _ in ()).throw(AssertionError('eager graph used'))}
    draft.math=lambda *args: (_ for _ in ()).throw(AssertionError('eager math used'))
    draft.step=torch.arange(7)[None]
    draft.calls=draft.rows=draft.graph_hits=draft.native_full_steps=0
    draft.execution_buckets={}
    draft.native_compare=[]
    cache=SimpleNamespace(length=20,pool_slot=0)
    jobs=[dict(cache=cache,first_position=5,anchor_token=torch.tensor(3))]
    result=draft(jobs)
    assert len(result)==1 and used==[([0],[20])]
    assert draft.native_full_steps==1
    assert draft.execution_buckets=={'b1_kv64_native_engine128':1}


def test_packed_draft_uses_selected_slots_and_restores_mirror():
    canonical=torch.arange(8,dtype=torch.float32).reshape(1,2,4,1,1,1).expand(1,2,4,1,128,1).clone()
    bank=NativeDraftFullBank113.__new__(NativeDraftFullBank113)
    bank.cache=torch.full_like(canonical,-1)
    seen=[]
    def graph(anchors,positions,slots,lengths):
        seen.append((anchors.tolist(),slots.tolist(),lengths.tolist(),bank.cache[:,:,0,0,0,0].tolist()))
        return torch.arange(4,dtype=torch.float32).reshape(4,1,1),torch.arange(4,dtype=torch.float32).reshape(4,1,1)
    bank.graphs={(4,128):graph}
    anchors=torch.tensor([7,8]);positions=torch.zeros(2,7,dtype=torch.long)
    slots=torch.tensor([3,1],dtype=torch.int32);lengths=torch.tensor([20,30],dtype=torch.int32)
    hidden,base=bank.run_packed(4,anchors,positions,slots,lengths,canonical)
    assert seen[0][:3]==([7,8,0,0],[0,1,2,3],[20,30,0,0])
    assert torch.equal(bank.cache[:,:,:4],canonical[:,:,:4])
    assert hidden.flatten().tolist()==[0,1] and base.flatten().tolist()==[0,1]
