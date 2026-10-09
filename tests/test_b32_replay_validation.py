import copy,json
from pathlib import Path

import pytest

from deployment.b32.replay_validation import digest,validate_plan


@pytest.fixture
def replay_files(tmp_path):
    binary=tmp_path/'model.engine';binary.write_bytes(b'local engine identity')
    onnx=tmp_path/'model.onnx';onnx.write_bytes(b'candidate representation')
    data=tmp_path/'model.onnx.data';data.write_bytes(b'candidate tensor bytes')
    plan=dict(component='vocoder',format=1,batch=32,frames=52,sm=89,gpu_name='NVIDIA GeForce RTX 4090',trt='11.3.0.99',kind='vocoder',precision='int8_smoothquant_static_qdq_fp32_interfaces',
        engine=binary.name,sha256=digest(binary),quantization_recipe=dict(scheme='int8_smoothquant',alpha=1.,role_specs_sha256='fixed recipe',role_manifest={'roles':['preserved']},calibration={'sha256':'fixed calibration'}),
        provenance=dict(model_sources={'checkpoint_sha256':'fixed weights'},onnx_binding=dict(onnx={'path':str(onnx),'sha256':digest(onnx)},external_data=[{'path':data.name,'sha256':digest(data)}])))
    source=copy.deepcopy(plan);source['provenance']['onnx_binding']={'onnx':{'path':'source representation','sha256':'source graph'}}
    path=tmp_path/'model.plan.json';path.write_text(json.dumps(plan))
    return {'plans':{'vocoder':source}},path,plan


def test_replay_allows_new_verified_representation_with_identical_recipe(replay_files):
    frozen,path,plan=replay_files
    assert validate_plan(frozen,path)==plan


@pytest.mark.parametrize('change',['recipe','weights','batch','engine','external'])
def test_replay_rejects_identity_or_recipe_mutation(replay_files,change):
    frozen,path,plan=replay_files
    if change=='recipe':plan['quantization_recipe']['role_specs_sha256']='different'
    elif change=='weights':plan['provenance']['model_sources']['checkpoint_sha256']='different'
    elif change=='batch':plan['batch']=16
    elif change=='engine':path.with_name('model.engine').write_bytes(b'tampered')
    else:path.with_name('model.onnx.data').write_bytes(b'tampered')
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):validate_plan(frozen,path)
