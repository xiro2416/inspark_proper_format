"""Current protected-role policy with NVIDIA dynamic NVFP4 W4A4 quantizers."""
from collections import Counter
import json
from pathlib import Path
import torch
from torch import nn
from inspark_infer.quantization.unified import iter_roles
from inspark_infer.build.unified_acoustic_export import ExportWeightOp,fold_weight_norm_
from inspark_infer.build.nvfp4_graph import MatrixWeightOp

def install(engine,artifact):
    roles=list(iter_roles(engine,'fp8'))
    fold_weight_norm_(engine.student.model);fold_weight_norm_(engine.tts.bigvgan)
    manifest=[]
    for role in roles:
        spec=artifact['role_specs'][role.path]
        if role.precision=='bf16':
            if spec['precision']!='bf16':raise ValueError('Protected role was changed')
            module=role.module
            if type(module).__name__=='Conv1D':
                w=role.weight();linear=nn.Linear(w.shape[1],w.shape[0],bias=module.bias is not None,device=w.device)
                linear.weight.data.copy_(w)
                if module.bias is not None:linear.bias.data.copy_(module.bias.detach())
                module=linear
            protected_spec=dict(spec)
            if isinstance(module,nn.ConvTranspose1d):protected_spec['conv_transpose_rewrite']='zero_insert_conv'
            replacement=ExportWeightOp(module,protected_spec)
        elif spec['precision']=='fp8' and artifact['scheme']=='nvfp4_fp8':
            module=role.module
            if type(module).__name__=='Conv1D':
                w=role.weight();linear=nn.Linear(w.shape[1],w.shape[0],bias=module.bias is not None,device=w.device)
                linear.weight.data.copy_(w)
                if module.bias is not None:linear.bias.data.copy_(module.bias.detach())
                module=linear
            replacement=ExportWeightOp(module,spec)
        else:
            if spec['precision']!='nvfp4':raise ValueError('A required native NVFP4 role was changed')
            replacement=MatrixWeightOp(role.module,conv_layout=spec.get("conv_layout","channel_tap")).apply_nvfp4(spec['activation_amax'])
            replacement.nvfp4_layout=dict(replacement.spec)
            replacement.spec=dict(spec)
        role.parent.add_module(role.child_name,replacement)
        manifest.append({'path':role.path,'component':role.component,'precision':spec['precision'],'module':replacement})
    counts=Counter(row['precision'] for row in manifest)
    expected=Counter(s['precision'] for s in artifact['role_specs'].values())
    if counts!=expected or counts['bf16']!=90 or sum(counts.values())!=317:
        raise ValueError('Role policy changed: '+str(counts))
    if artifact['scheme']=='nvfp4' and counts!={'bf16':90,'nvfp4':227}:raise ValueError('Native NVFP4 role policy changed')
    from inspark_infer.build.unified_acoustic_export import CFMBF16SDPAAttention,StaticAliasFree
    from inspark_infer.models.indextts2.upstream.s2mel.modules.gpt_fast.model import Attention
    for parent in engine.student.model.modules():
        for name,child in list(parent.named_children()):
            if isinstance(child,Attention):parent.add_module(name,CFMBF16SDPAAttention(child))
    for parent in engine.tts.bigvgan.modules():
        for name,child in list(parent.named_children()):
            if type(child).__name__=='Activation1d':parent.add_module(name,StaticAliasFree(child,fir_polyphase=True))
    return manifest

def make_recipe(float_artifact,channel_amax):
    recipe=dict(float_artifact);recipe['scheme']='nvfp4';recipe['algorithm']='NVIDIA ModelOpt NVFP4_DEFAULT_CFG max'
    recipe['role_specs']={}
    for name,spec in float_artifact['role_specs'].items():
        if spec['precision']=='bf16':recipe['role_specs'][name]=dict(spec)
        else:recipe['role_specs'][name]={'precision':'nvfp4','activation_amax':float(channel_amax[name].max()),
            'block_size':16,'data_format':'e2m1','scale_format':'e4m3','weight_input_axis':spec['weight_input_axis'],
            'weight_axis':spec['weight_axis'],'smoothing_source':'none','source':'modelopt_official_max_statistics_replay',
            'activation_quantization':'dynamic_per_16_GEMM_K_with_calibrated_FP32_global_scale'}
    recipe['cfm_intervals']=[[0,.25],[.25,.5],[.5,.75],[.75,1]]
    return recipe

def packed_state(module):
    """Official ModelOpt E2M1 nibble packing and two-level scales."""
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    quantizer=module.nvfp4_linear.weight_quantizer
    weight=module.weight.detach()
    global_scale=NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
    block_scale,_=NVFP4QTensor.get_weights_scaling_factor_from_quantizer(quantizer,weight,global_scale)
    qt,block,global_scale=NVFP4QTensor.quantize(weight,16,block_scale,global_scale)
    return {'weight_packed':qt._quantized_data.cpu(),'weight_block_scale':block.cpu(),
        'weight_global_scale':global_scale.cpu(),'activation_amax':module.nvfp4_linear.input_quantizer.amax.detach().float().cpu()}
