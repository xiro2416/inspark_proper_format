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
    draft.native_full_bank=SimpleNamespace(graphs={(1,128):graph})
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
