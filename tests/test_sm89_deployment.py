from types import SimpleNamespace

import pytest
import torch
from torch import nn

from inspark_infer.runtime.asset_identity import validate_model_identities
from inspark_infer.runtime.unified_deployment import validate
from inspark_infer.runtime.graph_policy import BATCHES
from deployment.export import TargetVerification
from inspark_infer.build.unified_acoustic_export import ExportWeightOp


@pytest.mark.parametrize('deconv', [False, True])
@pytest.mark.parametrize('representation', ['conv1d_as_2d','conv1d_as_gemm'])
def test_int8_convolution_rank_lift_preserves_exact_discrete_math(deconv,representation):
    module = (nn.ConvTranspose1d(4,8,4,stride=2,padding=1) if deconv else
              nn.Conv1d(4,8,3,padding=2,dilation=2))
    with torch.no_grad():
        module.weight.copy_((torch.arange(module.weight.numel()).reshape_as(module.weight)%7-3)*.125)
        module.bias.copy_(torch.arange(8)*.03125)
    spec=dict(precision='int8',smooth_scale=[1.]*4,input_scale=.25,
              weight_scale=[.125]*8,weight_axis=1 if deconv else 0)
    if deconv:spec['conv_transpose_rewrite']='zero_insert_conv'
    op=ExportWeightOp(module,spec)
    values=(torch.arange(80).reshape(1,4,20)%9-4)*.25
    original=op(values)
    setattr(op,representation,True)
    assert torch.equal(original,op(values))


def test_sm89_requires_explicit_hardware_and_cannot_accept_sm120():
    engine = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(
        get_device_name=lambda: 'NVIDIA GeForce RTX 4090',
        get_device_capability=lambda: (8, 9))))
    validate_model_identities(engine, {'hardware': {'gpu_name': 'NVIDIA GeForce RTX 4090', 'sm': 89}})
    with pytest.raises(ValueError, match='hardware identity'):
        validate_model_identities(engine, {})
    with pytest.raises(ValueError, match='hardware identity'):
        validate_model_identities(engine, {'hardware': {'gpu_name': 'NVIDIA GeForce RTX 4090', 'sm': 120}})


@pytest.mark.parametrize('changed_file', ['model.onnx', 'model.onnx.data'])
@pytest.mark.parametrize('relative_external', [False,True])
def test_resume_rejects_changed_build_inputs_even_when_engine_hash_matches(tmp_path,changed_file,relative_external):
    from deployment.multibatch.matrix import digest,save,verified_engine
    for name in ('model.onnx','model.onnx.data','model.engine'):
        (tmp_path/name).write_bytes(name.encode())
    record=lambda name:dict(path=str(tmp_path/name),sha256=digest(tmp_path/name))
    external=record('model.onnx.data')
    if relative_external:external['path']='model.onnx.data'
    save(tmp_path/'model.plan.json',dict(batch=16,sm=89,sha256=digest(tmp_path/'model.engine'),
        optimization_level=5,tiling_optimization_level='full',max_num_tactics=2147483646,
        provenance=dict(onnx_binding=dict(onnx=record('model.onnx'),external_data=[external]))))
    assert verified_engine(tmp_path,16)
    (tmp_path/changed_file).write_bytes(b'new build input')
    with pytest.raises(RuntimeError,match='ONNX binding mismatch'):
        verified_engine(tmp_path,16)


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16, 32, 64, 128])
def test_exact_local_batches_validate_without_relaxing_required_fields(batch):
    paths = dict.fromkeys(['calibration', 'target_plan', 'draft_plan', 'cfm_plan', 'vocoder_plan', 'official_sources'], 'asset.json')
    plan = dict(schema=9, status='local_sm89_candidate', precision='int8_smoothquant', batch=batch,
                runtime_backend='framework_dspark_adapter', graphs=True, **paths,
                hardware={'gpu_name': 'NVIDIA GeForce RTX 4090', 'sm': 89})
    assert batch in BATCHES
    assert validate(plan)['batch'] == batch
    del plan['draft_plan']
    with pytest.raises(ValueError, match='fields'):
        validate(plan)


def test_target_export_masks_uncommitted_history_and_keeps_appended_kv():
    torch.manual_seed(41)
    width, heads, past, query = 8, 2, 5, 3
    block = nn.Module()
    block.ln_1, block.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
    attention = nn.Module()
    attention.c_attn, attention.c_proj = nn.Linear(width, 3*width), nn.Linear(width, width)
    attention.split_size, attention.num_heads, attention.head_dim = width, heads, width//heads
    block.attn = attention
    block.mlp = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))
    transformer = nn.Module()
    transformer.h, transformer.ln_f = nn.ModuleList([block]), nn.LayerNorm(width)
    model = TargetVerification(SimpleNamespace(model=SimpleNamespace(transformer=transformer, lm_head=nn.Linear(width, 7)), target_layer_ids=[0])).eval()
    x = torch.randn(2, query, width)
    k, v = (torch.randn(2, heads, past, width//heads).bfloat16() for _ in range(2))
    history = torch.zeros(2, 1, query, past, dtype=torch.bool)
    history[..., :2] = True
    causal = torch.ones(query, query, dtype=torch.bool).tril()[None,None].expand(2,1,-1,-1)
    mask = torch.cat((history, causal), -1)
    original = model(x, mask, k, v)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:,:,2:] = 100; changed_v[:,:,2:] = -100
    changed = model(x, mask, changed_k, changed_v)
    assert all(torch.equal(a,b) for a,b in zip(original, changed))
    assert original[-1].dtype == torch.bfloat16 and original[-1].shape[-2] == query
    future_x = x.clone(); future_x[:, -1] += 10
    future = model(future_x, mask, k, v)
    assert torch.equal(original[0][:,:-1], future[0][:,:-1])
