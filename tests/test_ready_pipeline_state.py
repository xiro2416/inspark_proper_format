"""Request identity/RNG and immutable-source guarantees across compaction."""
from types import SimpleNamespace
import pytest
import torch
from deployment.ready_pipeline.scheduler import FIELDS,state_tensors,transfer
from inspark_infer.ops.trtllm.runtime import FrameworkRoundRuntime


def bank(n):
    r=SimpleNamespace(batch=n,eos=999)
    for field in FIELDS:
        dtype=torch.bool if field in ('done','ready','active') else torch.long
        shape=(n,8) if field in ('tokens','accepted') else (n,)
        setattr(r,field,torch.arange(torch.tensor(shape).prod()).reshape(shape).to(dtype))
    r.draws=SimpleNamespace(values=torch.arange(n*12,dtype=torch.float).reshape(n,3,4))
    r.failures=torch.tensor(0);r.capacity_failures=torch.tensor(0);r.status=torch.tensor(0)
    p=SimpleNamespace(target_cache=torch.arange(2*2*n*3,dtype=torch.float).reshape(2,2,n,3),target_keep=torch.ones(n,3,dtype=torch.bool),draft_cache=torch.arange(n*6,dtype=torch.float).reshape(1,2,n,3))
    return SimpleNamespace(runtime=r,provider=p)


def test_noncontiguous_compaction_keeps_request_history_rng_and_source():
    source,dest=bank(6),bank(4);indices=torch.tensor([5,1,3])
    before=[x.clone() for x,_ in state_tensors(source)]
    transfer(source,dest,indices)
    for (src,axis),(dst,_),old in zip(state_tensors(source),state_tensors(dest),before):
        assert torch.equal(src,old)
        assert torch.equal(dst.narrow(axis,0,3),old.index_select(axis,indices))
    assert dest.runtime.ready[3:].all() and dest.runtime.done[3:].all()
    assert dest.runtime.token_lengths[3:].eq(1).all()
    assert dest.runtime.last[3:].eq(999).all()
    assert dest.runtime.accepted[3:].eq(-1).all()
    assert dest.runtime.active[3:].eq(False).all()


def test_two_moves_preserve_original_identity_not_compacted_row_seed():
    source,middle,last=bank(6),bank(4),bank(2)
    transfer(source,middle,torch.tensor([5,1,3]));transfer(middle,last,torch.tensor([2,0]))
    for (src,axis),(dst,_) in zip(state_tensors(source),state_tensors(last)):
        assert torch.equal(dst,src.index_select(axis,torch.tensor([3,5])))


@pytest.mark.parametrize('indices',[torch.tensor([0,0]),torch.tensor([0,1,2]),torch.tensor([[0]]),torch.tensor([0.])])
def test_invalid_mapping_rejected_before_copy(indices):
    with pytest.raises(ValueError):transfer(bank(4),bank(2),indices)


def test_partial_observation_preserves_ready_eos_and_reports_failure():
    graph=SimpleNamespace(replay=lambda:None)
    runtime=SimpleNamespace(graph=graph,status=torch.tensor(0),ready=torch.tensor([True,False,True]),graph_burst=2)
    result=FrameworkRoundRuntime.advance_burst(runtime)
    assert result['ready']==[1,0,1] and result['launched_rounds']==2
    runtime.status.fill_(4)
    with pytest.raises(RuntimeError,match='failed'):FrameworkRoundRuntime.advance_burst(runtime)


def test_alias_bank_rejected_without_destroying_committed_state():
    source=bank(4);before=source.runtime.tokens.clone()
    with pytest.raises(ValueError,match="itself"):transfer(source,source,torch.tensor([0,1]))
    assert torch.equal(source.runtime.tokens,before)


def test_terminal_eos_state_and_request_draws_survive_mapping():
    source,dest=bank(6),bank(4)
    source.runtime.done[5]=True;source.runtime.ready[5]=True;source.runtime.last[5]=source.runtime.eos
    draws=source.runtime.draws.values[5].clone();rounds=source.runtime.rounds[5].clone()
    transfer(source,dest,torch.tensor([5,1]))
    assert dest.runtime.done[0] and dest.runtime.ready[0]
    assert dest.runtime.last[0]==source.runtime.eos
    assert torch.equal(dest.runtime.draws.values[0],draws)
    assert torch.equal(dest.runtime.rounds[0],rounds)


def test_failed_pipeline_cannot_retry_stale_rows_and_releases_model_lock():
    from contextlib import nullcontext
    from inspark_infer.runtime.engine import Engine
    e=Engine.__new__(Engine);row=SimpleNamespace();session=dict(chunks=[],_row=row,error=None,complete=False,parts=['test'])
    released=[];e.closed=False;e.config={'max_batch':64};e.sessions={'x':session};e.rt=SimpleNamespace()
    e.ready=lambda:[s for s in e.sessions.values() if not s['error']]
    e.torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _:nullcontext()),inference_mode=nullcontext)
    e.model=SimpleNamespace(stream=None,_acquire=lambda _:None,_release=lambda:released.append(True))
    e.unified_first_chunk=SimpleNamespace(failure_count=0)
    def fail(*_):raise RuntimeError('status=4')
    e.head_ready_pipeline=SimpleNamespace(run=fail)
    with pytest.raises(RuntimeError,match='status=4'):e.run_ready()
    assert released==[True] and e.unified_first_chunk.failure_count==1
    assert session['error'] and '_row' in session
    assert e.run_ready()==[]
