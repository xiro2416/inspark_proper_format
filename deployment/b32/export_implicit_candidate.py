"""Preserve source Q/DQ scales and use compact INT8 input storage for convolution."""
import argparse
import ast,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=int,choices=[1,2,4,8,16,32,64,128],default=32)
    parser.add_argument("--history-dir",type=Path,default=ROOT/"deployment/b32/history")
    args=parser.parse_args();batch=args.batch;args.history_dir.mkdir(parents=True,exist_ok=True)
    from trt113_provenance import capture_onnx_artifact,file_record
    from inspark_infer.models.indextts2.upstream.s2mel.modules.bigvgan.bigvgan import BigVGAN,load_hparams_from_json
    source=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-tiled-fir/model.onnx'
    record=json.loads(source.with_suffix('.export.json').read_text());model=onnx.load(source,load_external_data=True)
    config=ROOT/'local_assets/runtime/models/index_tts2/hf_cache/bigvgan/config.json'
    geometry=BigVGAN(load_hparams_from_json(config))
    paths=record['quantization_recipe']['role_manifest']['conv1d_as_gemm']
    nodes=list(model.graph.node);initializers={v.name:v for v in model.graph.initializer};static=set(initializers)
    for n in nodes:
        if n.op_type=='Constant' or all(v in static for v in n.input if v):static.update(n.output)
    producer={v:i for i,n in enumerate(nodes) for v in n.output};consumers={}
    for i,n in enumerate(nodes):
        for v in n.input:consumers.setdefault(v,[]).append(i)
    aliases={p:[] for p in paths}
    for i,n in enumerate(nodes):
        props={v.key:v.value for v in n.metadata_props}
        for scope in ast.literal_eval(props.get('pkg.torch.onnx.name_scopes','[]')):
            if scope in aliases:aliases[scope].append(i)
    replacements={};rewrites=[]
    for number,path in enumerate(paths):
        module=geometry.get_submodule(path);indices=aliases[path];members=set(indices)
        if module.groups!=1:raise RuntimeError('Only original group1 calibrated convolutions supported')
        emitted={v for i in indices for v in nodes[i].output}
        exposed={v for v in emitted-static if any(i not in members for i in consumers.get(v,[])) or v in {x.name for x in model.graph.output}}
        if len(exposed)!=1:raise RuntimeError('Nonlocal weighted scope: '+path+' '+str(exposed))
        y=next(iter(exposed));mat=next(nodes[i] for i in indices if nodes[i].op_type=='MatMul')
        activation_dq=nodes[producer[mat.input[0]]]
        weight_transpose=nodes[producer[mat.input[1]]];weight_dq=nodes[producer[weight_transpose.input[0]]]
        if activation_dq.op_type!='DequantizeLinear' or weight_dq.op_type!='DequantizeLinear':raise RuntimeError('Source INT8 patterns changed')
        weight_q=nodes[producer[weight_dq.input[0]]]
        act_scale,act_zero=activation_dq.input[1:3];weight_scale=weight_dq.input[1]
        for zero in (act_zero,weight_q.input[2]):
            if zero not in initializers or np.any(numpy_helper.to_array(initializers[zero])):raise RuntimeError('Only unchanged zero-centered INT8 recipe supported')
        divs=[nodes[i] for i in indices if nodes[i].op_type=='Div' and nodes[i].output[0] not in static]
        if divs:
            normalized=divs[0].output[0]
        else:
            dynamic={v for i in indices for v in nodes[i].input if v and v not in emitted and v not in static}
            if len(dynamic)!=1:raise RuntimeError('No unique original convolution input: '+path)
            normalized=next(iter(dynamic))
        # Force actual compact NCF INT8 storage at the opaque convolution boundary.
        qname=f'b32_implicit_input_{number}'
        q=helper.make_node('QuantizeLinear',[normalized,act_scale,act_zero],[qname],name=f'b32_source_quantize_{number}')
        bias_name=path+'.bias'
        if bias_name not in initializers:
            bias_name=f'b32_implicit_zero_bias_{number}';model.graph.initializer.append(numpy_helper.from_array(np.zeros(module.out_channels,np.float32),bias_name))
        deconv=type(module).__name__=='ConvTranspose1d'
        expand=int(module.stride[0]) if deconv else 1
        stride=1 if deconv else int(module.stride[0])
        pad=int(module.dilation[0]*(module.kernel_size[0]-1)-module.padding[0]) if deconv else int(module.padding[0])
        dilation=int(module.dilation[0])
        # Source model static output shape is present in the exported value info.
        info=next(v for v in list(model.graph.value_info)+list(model.graph.output) if v.name==y)
        dims=[d.dim_value for d in info.type.tensor_type.shape.dim]
        if dims[:2]!=[batch,module.out_channels] or len(dims)!=3 or min(dims)<=0:raise RuntimeError('Invalid source output shape')
        plugin=helper.make_node('implicit_int8_conv_1d',[qname,weight_q.output[0],act_scale,weight_scale,bias_name],[y],
            name=f'b32_implicit_conv_{number}',domain='inspark_custom',plugin_namespace='inspark_custom',
            stride=stride,pad=pad,dilation=dilation,expand=expand,output_length=dims[2])
        replacements[producer[y]]=[q,plugin]
        rewrites.append(dict(path=path,input=normalized,output=y,output_shape=dims,
            stride=stride,pad=pad,dilation=dilation,expand=expand,kernel=module.kernel_size[0],
            original_weight_quantize_node=weight_q.name,activation_scale=act_scale,weight_scale=weight_scale))
    new=[]
    for i,n in enumerate(nodes):new.extend(replacements[i] if i in replacements else [n])
    needed={v.name for v in model.graph.output};kept=[]
    for n in reversed(new):
        if any(v in needed for v in n.output):kept.append(n);needed.update(n.input)
    new=list(reversed(kept));del model.graph.node[:];model.graph.node.extend(new)
    used={v for n in new for v in n.input};kept=[v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-implicit-int8/model.onnx';output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save_model(model,output,save_as_external_data=True,all_tensors_to_one_file=True,location='model.onnx.data',size_threshold=1024)
    onnx.checker.check_model(str(output));artifact=capture_onnx_artifact(output)
    record.update(onnx=str(output),onnx_sha256=artifact['sha256'],onnx_artifact=artifact,
        plugins=['inspark_custom::small_fir_activation_tiled','inspark_custom::implicit_int8_conv_1d'],
        graph=dict(nodes=len(new),custom_domains=['inspark_custom'],plugins=['inspark_custom::small_fir_activation_tiled','inspark_custom::implicit_int8_conv_1d']),
        custom_implicit_int8_conv=dict(rewrites=rewrites,compact_int8_storage=True,integer_accumulation='INT32',
            source=file_record(source,'validated_tiled_fir_source'),original_geometry_config=file_record(config,'original_bigvgan_config'),
            precision='unchanged signed INT8 operands; FP32 source scaling/bias/output; protected floating operators untouched',
            tile=[32,32,64],warps=4,stages=3,source_weight_quantize_nodes='preserved unchanged'))
    record['validation'].update(output_finite=None,numerical_audit='pending_candidate_real_input_replay')
    output.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
    (args.history_dir/'implicit-candidate-rewrites.json').write_text(json.dumps(dict(source_nodes=len(nodes),target_nodes=len(new),convolutions=len(rewrites),rewrites=rewrites),indent=2)+'\n')
    print(json.dumps(dict(convolutions=len(rewrites),nodes=len(new),onnx=str(output))),flush=True)


if __name__=='__main__':main()
