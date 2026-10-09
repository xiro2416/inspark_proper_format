"""Frozen audit crops request axes identically for every tensor and case."""
import pytest
import torch
from scripts.audit_ready_ar_bucket import slice_frozen


def payload():
    rows=torch.arange(64)
    return dict(schema=1,kind='unified_frozen_first_ar_round',batch=64,cases=[dict(id=str(i)) for i in range(64)],
        inputs=dict(anchors=rows,target_keep=rows[:,None],draft_cache=rows.view(1,1,64,1,1,1),target_cache=rows.view(1,1,64,1,1,1)),
        actual=dict(draft_hidden=rows[:,None,None],target_kv_append=rows.view(1,1,64,1,1,1)))


def test_noncontiguous_cache_and_row_axes_agree():
    index=torch.arange(63,15,-1)
    inputs,actual,cases=slice_frozen(payload(),index)
    assert torch.equal(inputs['anchors'],index)
    assert torch.equal(inputs['draft_cache'].flatten(),index)
    assert torch.equal(inputs['target_cache'].flatten(),index)
    assert torch.equal(actual['target_kv_append'].flatten(),index)
    assert [case['id'] for case in cases]==list(map(str,index.tolist()))


@pytest.mark.parametrize('indices',[torch.arange(47),torch.zeros(48,dtype=torch.long),torch.arange(48).float(),torch.arange(-1,47),torch.arange(17,65)])
def test_bad_request_mapping_rejected(indices):
    with pytest.raises(ValueError):slice_frozen(payload(),indices)


def test_bad_source_batch_and_tensor_axis_rejected():
    p=payload();p['batch']=48
    with pytest.raises(ValueError):slice_frozen(p,torch.arange(48))
    p=payload();p['inputs']['anchors']=torch.arange(32)
    with pytest.raises(ValueError,match='request axis'):slice_frozen(p,torch.arange(48))
