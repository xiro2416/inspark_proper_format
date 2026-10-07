import json
from pathlib import Path
import pytest
import torch
from inspark_infer.runtime.asset_identity import tensor_digest,validate_component_roles
from inspark_infer.runtime.bundle_paths import read_json
from inspark_infer.runtime.deployment import validate

def test_tensor_identity_is_storage_container_independent():
    a={'weight':torch.arange(12,dtype=torch.float32).reshape(3,4)}
    assert tensor_digest(a)==tensor_digest({'weight':a['weight'].clone()})
    b={'weight':a['weight'].clone()};b['weight'][0,0]=1
    assert tensor_digest(a)!=tensor_digest(b)

def test_portable_bundle_moves_without_absolute_paths(tmp_path):
    root=tmp_path/'moved';root.mkdir();(root/'.bundle_root').write_text('current')
    p=root/'plan.json';p.write_text(json.dumps({'engine':'bundle://assets/engine.bin'}))
    assert read_json(p)['engine']==str(root/'assets/engine.bin')
    p.write_text(json.dumps({'engine':'bundle://../outside'}))
    with pytest.raises(ValueError,match='escapes'):read_json(p)

def test_reuse_requires_matching_component_recipe(tmp_path):
    source=tmp_path/'old.json';current=tmp_path/'current.json'
    old={'scheme':'fp8','role_specs':{'target.blocks.0':{'precision':'bf16'},'draft.blocks.1':{'input_scale':1}}}
    new={'scheme':'fp8','role_specs':{'target.blocks.0':{'precision':'bf16'},'draft.blocks.1':{'input_scale':2}}}
    source.write_text(json.dumps(old));current.write_text(json.dumps(new))
    plan={'calibration':str(current),'component_calibrations':{'target':str(source)}}
    validate_component_roles(plan)
    new['role_specs']['target.blocks.0']['precision']='fp8';current.write_text(json.dumps(new))
    with pytest.raises(ValueError,match='target'):validate_component_roles(plan)

def test_historical_optimization_route_is_removed():
    with pytest.raises(ValueError,match='Historical'):validate({'schema':8})

def test_models_can_link_to_an_immutable_archive(tmp_path,monkeypatch):
    base=tmp_path/'models';base.mkdir();archive=tmp_path/'archive';archive.mkdir()
    (archive/'weight.bin').write_bytes(b'weights');(base/'shared').symlink_to(archive,target_is_directory=True)
    monkeypatch.setenv('INSPARK_MODEL_ROOT',str(base))
    p=tmp_path/'plan.json';p.write_text(json.dumps({'source':'model://shared/weight.bin'}))
    assert Path(read_json(p)['source']).read_bytes()==b'weights'
    p.write_text(json.dumps({'source':'model://../archive/weight.bin'}))
    with pytest.raises(ValueError,match='escapes'):read_json(p)

def test_wrong_checkpoint_is_rejected(monkeypatch):
    from types import SimpleNamespace
    import inspark_infer.runtime.asset_identity as identity
    state={'weight':torch.tensor([1.])}
    monkeypatch.setattr(identity,'checkpoint_states',lambda cfg:{c:(lambda:state) for c in ['target','draft','cfm','vocoder']})
    expected={c:tensor_digest(state) for c in ['target','draft','cfm','vocoder']}
    identity.validate_model_identities(SimpleNamespace(config={}),{'model_tensor_hashes':expected})
    expected['draft']=tensor_digest({'weight':torch.tensor([2.])})
    with pytest.raises(ValueError,match='draft'):identity.validate_model_identities(SimpleNamespace(config={}),{'model_tensor_hashes':expected})
