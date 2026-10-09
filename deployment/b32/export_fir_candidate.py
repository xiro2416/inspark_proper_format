"""Replace exact exported FIR/Snake/FIR scopes with the existing tiled AOT plugin."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import helper,numpy_helper

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=int,choices=[1,2,4,8,16,32,64,128],default=32)
    parser.add_argument("--history-dir",type=Path,default=ROOT/"deployment/b32/history")
    args=parser.parse_args();batch=args.batch;args.history_dir.mkdir(parents=True,exist_ok=True)
    from trt113_provenance import capture_onnx_artifact,file_record
    source=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-gemm/model.onnx'
    record=json.loads(source.with_suffix('.export.json').read_text())
    model=onnx.load(source,load_external_data=True)
    alias_paths=record['quantization_recipe']['role_manifest']['fir_polyphase_graphs']
    weights_path=ROOT/'local_assets/runtime/models/index_tts2/hf_cache/bigvgan/bigvgan_generator.pt'
    weights=torch.load(weights_path,map_location='cpu',weights_only=True)['generator']
    nodes=list(model.graph.node);initializer={value.name:value for value in model.graph.initializer}
    static=set(initializer)
    for node in nodes:
        if node.op_type=='Constant' or all(value in static for value in node.input if value):static.update(node.output)
    producer={value:index for index,node in enumerate(nodes) for value in node.output}
    consumers={}
    for index,node in enumerate(nodes):
        for value in node.input:consumers.setdefault(value,[]).append(index)
    aliases={name:[] for name in alias_paths}
    for index,node in enumerate(nodes):
        props={prop.key:prop.value for prop in node.metadata_props}
        scopes=ast.literal_eval(props.get('pkg.torch.onnx.name_scopes','[]'))
        for scope in scopes:
            if scope in aliases:aliases[scope].append(index)
    replacements={};removed=set();rewrites=[]
    for number,name in enumerate(alias_paths):
        indices=aliases[name];members=set(indices)
        if not indices:raise RuntimeError('Missing source FIR scope: '+name)
        down_nodes=[nodes[index] for index in indices if nodes[index].op_type=='Conv'
                    and nodes[index].input[1] in initializer
                    and list(initializer[nodes[index].input[1]].dims)[1:]==[1,12]]
        if len(down_nodes)!=1:raise RuntimeError('Expected exact final FIR convolution: '+name)
        down_node=down_nodes[0];y=down_node.output[0]
        down=numpy_helper.to_array(initializer[down_node.input[1]])
        channels=down.shape[0]
        # The exporter shares upsampling between parallel residual branches.
        # Find the actual shared producer instead of assuming module scopes own it.
        frontier=set(down_node.input);seen=set();up_node=None
        while frontier and up_node is None:
            next_frontier=set()
            for value in frontier-static-seen:
                seen.add(value)
                node=nodes[producer[value]]
                if (node.op_type=='Conv' and node.input[1] in initializer
                        and list(initializer[node.input[1]].dims)==[2*channels,1,6]):
                    up_node=node;break
                next_frontier.update(node.input)
            frontier=next_frontier
        if up_node is None:raise RuntimeError('No source polyphase producer: '+name)
        up_pad=nodes[producer[up_node.input[0]]]
        if up_pad.op_type!='Pad':raise RuntimeError('Source upsample padding is not explicit')
        x=up_pad.input[0]
        phase=numpy_helper.to_array(initializer[up_node.input[1]])
        channels=down.shape[0]
        if not np.array_equal(phase,np.tile(phase[:2],(channels,1,1))) or not np.array_equal(down,np.tile(down[:1],(channels,1,1))):raise RuntimeError('Source FIR taps differ per channel')
        up=np.empty(12,dtype=np.float32);up[::2]=phase[0,0,::-1];up[1::2]=phase[1,0,::-1]
        alpha=weights[name+'.act.alpha'].numpy();beta=weights[name+'.act.beta'].numpy()
        if alpha.shape!=(channels,) or beta.shape!=(channels,):raise RuntimeError('Source activation channel mismatch')
        parameter_names=[]
        for role,array in [('up',up),('down',down[0,0]),('alpha_log',alpha),('beta_log',beta)]:
            key=f'b32_fir_{number}_{role}';model.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array),key));parameter_names.append(key)
        plugin=helper.make_node('small_fir_activation_tiled',[x,*parameter_names],[y],name=f'b32_tiled_fir_{number}',domain='inspark_custom',plugin_namespace='inspark_custom')
        replacements[producer[y]]=plugin;removed.add(producer[y])
        rewrites.append(dict(path=name,nodes_removed=len(indices),input=x,output=y,channels=channels,parameters=parameter_names))
    new=[]
    for index,node in enumerate(nodes):
        if index in replacements:new.append(replacements[index])
        elif index not in removed:new.append(node)
    # Preserve shared producers until all activation endpoints are replaced,
    # then remove only nodes with no path to any required output.
    needed={value.name for value in model.graph.output};kept_nodes=[]
    for node in reversed(new):
        if any(value in needed for value in node.output):
            kept_nodes.append(node);needed.update(node.input)
    new=list(reversed(kept_nodes))
    del model.graph.node[:];model.graph.node.extend(new)
    model.opset_import.append(helper.make_opsetid('inspark_custom',1))
    # Prune obsolete initializer/filter constants; never prune computation outside
    # the proven single-input/single-output activation scopes.
    used={value for node in new for value in node.input}
    kept=[value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:];model.graph.initializer.extend(kept)
    output=ROOT/f'artifacts/sm89/int8_smoothquant/b{batch}/vocoder-tiled-fir/model.onnx';output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save_model(model,output,save_as_external_data=True,all_tensors_to_one_file=True,location='model.onnx.data',size_threshold=1024)
    onnx.checker.check_model(str(output))
    artifact=capture_onnx_artifact(output)
    record.update(onnx=str(output),onnx_sha256=artifact['sha256'],onnx_artifact=artifact,
        plugins=['inspark_custom::small_fir_activation_tiled'],
        graph=dict(nodes=len(new),custom_domains=['inspark_custom'],plugins=['inspark_custom::small_fir_activation_tiled']),
        custom_small_fir=dict(layout='all_tiled',batch=batch,rewrites=rewrites,
            source_onnx=file_record(source,'validated_migration_source_onnx'),
            original_weights=file_record(weights_path,'original_bigvgan_checkpoint'),
            mechanism='complete 12-tap upsample/SnakeBeta/downsample halos; unchanged FP32 activation math and original raw log parameters',
            scope='custom AOT/Triton activation inside unchanged INT8 weighted-op pipeline'))
    record['validation'].update(source_output_finite=True,output_finite=None,numerical_audit='pending_candidate_real_input_replay')
    output.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
    (args.history_dir/'fir-candidate-rewrites.json').write_text(json.dumps(dict(source_nodes=len(nodes),target_nodes=len(new),activation_count=len(rewrites),rewrites=rewrites),indent=2)+'\n')
    print(json.dumps(dict(activations=len(rewrites),source_nodes=len(nodes),target_nodes=len(new),onnx=str(output))),flush=True)


if __name__=='__main__':main()
