"""CPU-only request mapping, recipe binding and serial publication invariants."""
from types import SimpleNamespace as NS
import hashlib
import json
import pytest
import torch
from inspark_infer.runtime.ready_scheduler import (
    ReadyPipeline, validate_manifest, validate_component_identity,
    transfer_native, eager_reference, advance_native,
)
from inspark_infer.ops.trtllm.compact_tail import state_pairs


def runtime(n):
    r = NS(batch=n)
    for name in ('tokens','accepted','past','draft_lengths','mel_lengths','done','active','next_tokens'):
        setattr(r,name,torch.arange(n*3).reshape(n,3) if name in ('tokens','accepted','next_tokens') else torch.arange(n))
    r.draws=NS(values=torch.arange((n+1)*8).reshape(n+1,2,4))
    r.provider=NS(target_cache=torch.arange(2*2*n*3).reshape(2,2,n,3),target_keep=torch.arange(n*3).reshape(n,3))
    r.worker=NS(_ctx_k_buf=torch.arange((n+1)*6).reshape(n+1,2,3),_ctx_v_buf=torch.arange((n+1)*6).reshape(n+1,2,3))
    r.policy=NS()
    for name in ('lengths','rounds','last','ready','last_committed','tokens','probabilities','has_proposal'):
        setattr(r.policy,name,torch.arange((n+1)*3).reshape(n+1,3) if name in ('tokens','probabilities') else torch.arange(n+1))
    r.failures=torch.tensor(0);r.capacity_failures=torch.tensor(0);r.status=torch.tensor(0)
    return r


def test_two_noncontiguous_moves_keep_all_request_state():
    source,middle,last=runtime(6),runtime(4),runtime(2)
    before=[x.clone() for x,_ in state_pairs(source)]
    transfer_native(source,middle,torch.tensor([5,1,3]));transfer_native(middle,last,torch.tensor([2,0]))
    for ((src,axis),(dst,_),old) in zip(state_pairs(source),state_pairs(last),before):
        assert torch.equal(src,old)
        assert torch.equal(dst.narrow(axis,0,2),old.index_select(axis,torch.tensor([3,5])))
    assert middle.policy.ready[3:].all() and middle.done[3:].all()
    assert middle.active[3:].eq(0).all() and middle.draws.values[3:].eq(0).all()
    assert len(state_pairs(source)) == 21


@pytest.mark.parametrize('idx',[torch.tensor([0,0]),torch.tensor([[0]]),torch.tensor([0.]),torch.tensor([0,1,2]),torch.tensor([-1]),torch.tensor([4])])
def test_invalid_mapping_rejected_before_mutation(idx):
    src,dst=runtime(4),runtime(2);old=dst.tokens.clone()
    with pytest.raises(ValueError):transfer_native(src,dst,idx)
    assert torch.equal(dst.tokens,old)


def test_self_and_geometry_move_rejected():
    source=runtime(4)
    with pytest.raises(ValueError,match='itself'):transfer_native(source,source,torch.tensor([0]))
    dst=runtime(2);dst.policy.probabilities=torch.zeros(3,4)
    with pytest.raises(ValueError,match='geometry'):transfer_native(source,dst,torch.tensor([0]))


@pytest.mark.parametrize('batch,buckets',[(32,[32,16,8]),(64,[64,48,32,16,8]),(128,[128,64,48,32,16,8])])
def test_manifest_exact_inventory(batch,buckets):
    manifest=dict(schema=1,admission_batch=batch,acoustic_batch=16,ar_buckets={str(b):'/tmp/b'+str(b) for b in buckets},acoustic_deployment='/tmp/a16')
    assert validate_manifest(manifest) is manifest
    manifest['ar_buckets'].pop('8')
    with pytest.raises(ValueError,match='inventory'):validate_manifest(manifest)


@pytest.mark.parametrize('key,value',[('schema',2),('admission_batch',16),('acoustic_batch',8),('late_verify_after',10),('graph_burst_rounds',4)])
def test_invalid_schedule_rejected(key,value):
    m=dict(schema=1,admission_batch=32,acoustic_batch=16,ar_buckets={str(b):'/tmp' for b in [32,16,8]},acoustic_deployment='/tmp/a')
    m[key]=value
    with pytest.raises(ValueError):validate_manifest(m)


def test_wrapper_eager_resolution():
    eager=object();inner=NS(eager=eager)
    assert eager_reference(NS(fallback=NS(fallback=inner))) is eager
    circular=NS();circular.fallback=circular
    with pytest.raises(ValueError,match='Circular'):eager_reference(circular)


def recipe_fixture(tmp_path,name,global_scheme,specs):
    path=tmp_path/name;path.write_text(json.dumps(dict(scheme=global_scheme,role_specs=specs)))
    selected={k:v for k,v in specs.items() if k.startswith('target.')}
    metadata=dict(trt='11.3',sm=120,gpu_name='RTX6000D',provenance=dict(model_sources=[dict(role='target',sha256='model')]),
        quantization_recipe=dict(calibration=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        role_specs_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()))
    return metadata,dict(calibration=str(path))


def test_component_identity_uses_real_roles_not_global_recipe(tmp_path):
    a,pa=recipe_fixture(tmp_path,'a.json','nvfp4',{'target.x':{'precision':'nvfp4','scale':0.5}})
    b,pb=recipe_fixture(tmp_path,'b.json','nvfp4_fp8',{'target.x':{'precision':'nvfp4','scale':0.5},'vocoder.x':{'precision':'fp8'}})
    validate_component_identity(a,b,pa,pb,'target')
    # Matching declared role hashes cannot conceal a changed actual scale.
    payload=json.loads((tmp_path/'b.json').read_text());payload['role_specs']['target.x']['scale']=0.6
    (tmp_path/'b.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='actual data'):validate_component_identity(a,b,pa,pb,'target')


def test_component_identity_rejects_valid_but_different_arithmetic(tmp_path):
    a,pa=recipe_fixture(tmp_path,'a.json','nvfp4',{'target.x':{'precision':'nvfp4','scale':0.5}})
    b,pb=recipe_fixture(tmp_path,'b.json','fp8',{'target.x':{'precision':'fp8','scale':0.5}})
    with pytest.raises(ValueError,match='arithmetic'):validate_component_identity(a,b,pa,pb,'target')


def test_observation_uses_global_late_round_and_fails_status():
    calls=[];r=NS(graph=NS(replay=lambda:calls.append('full')),verify_graph=NS(replay=lambda:calls.append('verify')),
        late_verify_after=12,graph_burst=2,status=torch.tensor(0),ready=torch.tensor([False,True]))
    assert advance_native(r,10)['launched_rounds']==2
    assert advance_native(r,12)['deferred']
    assert calls==['full','verify']
    r.status.fill_(4)
    with pytest.raises(RuntimeError,match='failed'):advance_native(r,13)


@pytest.mark.parametrize('count',[1,15,17,32])
@pytest.mark.parametrize('mode',['A','C','barrier16'])
def test_serial_publication_exact_groups_and_tail_padding(monkeypatch,count,mode):
    from inspark_infer.runtime import ready_scheduler as module
    timeline=[]
    r=NS(batch=32,tokens=torch.arange(32*4).reshape(32,4),accepted=torch.ones(32,64,dtype=torch.long),
         token_lengths=torch.full((32,),3),past=torch.full((32,),40),draft_lengths=torch.full((32,),40),
         rounds=torch.full((32,),2),done=torch.zeros(32,dtype=torch.bool),last=torch.ones(32,dtype=torch.long),
         ready=torch.zeros(32,dtype=torch.bool))
    controller=NS(runtime=r,supports=lambda rows:True,begin=lambda rows:None,
        calls=0,total_launched_rounds=0,total_status_reads=0,total_status_wait_ms=0.,total_graph_replays=0,
        total_target_enqueues=0,total_draft_enqueues=0,total_prime_enqueues=0)
    rows=[NS() for _ in range(count)]
    owner={id(row):dict(case=dict(id=str(i),voice_id='v'),rounds=0,error=None,chunks=[],complete=False,_row=row)
           for i,row in enumerate(rows)}
    released=[]
    e=NS(model=NS(bank=NS(get=lambda _:dict(values={'voice.cache_mel':torch.zeros(1,80,258)})),stream=object()),
        rt=NS(native_target_steps=0,backbone=NS()),device_round_attempts=0,device_round_successes=0,
        device_round_status_reads=0,device_round_status_wait_ms=0.,device_round_launched_rounds=0,
        _finish_or_drain=lambda session:None,_release_row=lambda row:released.append(row))
    pipeline=ReadyPipeline.__new__(ReadyPipeline)
    pipeline.closed=False;pipeline.engine=e;pipeline.controllers={32:controller};pipeline.batch=32
    pipeline.acoustic_batch=16;pipeline.mode=mode;pipeline.waves=[];pipeline.stream=NS(synchronize=lambda:None)
    def advance(_,logical_round):
        timeline.append(('ar',logical_round))
        r.ready[:min(count,16 if logical_round==0 else count)].fill_(True)
        return dict(ready=r.ready.int().tolist(),launched_rounds=2,deferred=False,status_wait_ms=0.)
    monkeypatch.setattr(module,'advance_native',advance)
    monkeypatch.setattr(torch.cuda,'Event',lambda:NS(record=lambda stream:None))
    def render(group,owners,dependencies):
        timeline.append(('render',len(group)))
        for row in group:owners[id(row)]['chunks'].append(dict(pcm=b'pcm'))
        return group
    pipeline.render=render
    events=pipeline.run(rows,owner,lambda event:timeline.append(('callback',event['request_id'])))
    assert len(events)==count and len(released)==count and len(set(map(id,released)))==count
    assert [g['rows'] for g in pipeline.waves[-1]['acoustic_groups']]==([16]*(count//16)+([count%16] if count%16 else []))
    assert all('_row' not in session for session in owner.values())
    if count>16 and mode!='barrier16':
        assert timeline.index(('callback','15'))<timeline.index(('ar',2))
    if count>16 and mode=='barrier16':
        assert timeline.index(('ar',2))<timeline.index(('callback','0'))
