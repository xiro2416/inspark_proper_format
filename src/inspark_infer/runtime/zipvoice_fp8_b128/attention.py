"""Map modern ONNX attention boundaries to the existing floating fusion ABI."""
import importlib
import re


def boundaries(graph):
    by={o:n for n in graph.graph.node for o in n.output}
    groups={}
    for node in graph.graph.node:
        meta={x.key:x.value for x in node.metadata_props}
        ns=meta.get('namespace','')
        found=re.search(r'model\.fm_decoder\.encoders\.\d+(?:\.encoder)?\.layers\.\d+',ns)
        if found:groups.setdefault(found.group(0),[]).append(node)
    result=[]
    for scope,nodes in groups.items():
        def subsystem(node,name):return scope+'.'+name in next((x.value for x in node.metadata_props if x.key=='namespace'),'')
        weights=[n for n in nodes if subsystem(n,'self_attn_weights')]
        soft=next(n for n in weights if n.op_type=='Softmax')
        masked=by[soft.input[0]];assert masked.op_type=='Where'
        addition=by[masked.input[2]];assert addition.op_type=='Add'
        gather=next(by[k] for k in addition.input if by[k].op_type=='GatherElements')
        assert next(a.i for a in gather.attribute if a.name=='axis')==-1
        position=by[gather.input[0]];assert position.op_type=='MatMul'
        score=next(by[k] for k in addition.input if k!=gather.output[0]);assert score.op_type=='MatMul'
        branches=[]
        for name in ['nonlin_attention','self_attn1','self_attn2']:
            choices=[n for n in nodes if n.op_type=='MatMul' and subsystem(n,name)
                     and 'aten.matmul.default' in next((x.value for x in n.metadata_props if x.key=='pkg.torch.onnx.fx_node'),'')]
            if len(choices)!=1:raise ValueError('Ambiguous original AV boundary: '+scope+'.'+name)
            branches.append(choices[0])
        result.append(dict(scope=scope,q=score.input[0],k=score.input[1],pq=position.input[0],
                           embedding=position.input[1],mask=masked.input[0],branches=branches))
    if len(result)!=16:raise ValueError('Incomplete modern attention mapping')
    return result


def rewrite(network,graph,batch,trt,geometry=False):
    import tensorrt.plugin as trtp
    package=f'inspark_infer.ops.tensorrt.zipvoice_fp8_b128.b{batch}'+('.geo' if geometry else '')
    importlib.import_module(package+'.normal_tf32_plugin')
    importlib.import_module(package+'.online_nonlinear_rna_plugin')
    layers=[network.get_layer(i) for i in range(network.num_layers)]
    tensors={network.get_input(i).name:network.get_input(i) for i in range(network.num_inputs)}
    tensors.update({layer.get_output(i).name:layer.get_output(i) for layer in layers for i in range(layer.num_outputs)})
    changes=[]
    for contract in boundaries(graph):
        name=contract['scope'];common=[]
        for key,shape in [('q',(4,batch,-1,32)),('k',(4,batch,32,-1)),('pq',(4,batch,-1,4)),
                          ('embedding',(4,1,4,-1)),('mask',(batch,-1))]:
            value=tensors[contract[key]]
            if value.dtype!=(trt.bool if key=='mask' else trt.float32):raise ValueError('Attention ABI dtype mismatch')
            view=network.add_shuffle(value);view.reshape_dims=shape;view.name=name+'/fp8_common_'+key
            common.append(view.get_output(0))
        stats=None;outputs=[]
        for branch,node in enumerate(contract['branches']):
            value=tensors[node.input[1]];assert value.dtype==trt.float32
            view=network.add_shuffle(value);view.reshape_dims=(1,batch,-1,384) if branch==0 else (4,batch,-1,12)
            prefix=f'zipvoice_fp8_sm120_b{batch}'+('_geo' if geometry else '')
            namespace=getattr(trtp.op,prefix if branch==0 else prefix+'_normal_tf32')
            operation=getattr(namespace,'OnlineNonlinearRNAFloat32' if branch==0 else 'OnlineNormalWideStatsWriteFloat32' if branch==1 else 'OnlineNormalWideStatsReadFloat32')
            args=[*common,view.get_output(0)]
            if branch==2:args.append(stats)
            factory=operation(*args);plugin=network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT))
            plugin.name=name+f'/fp8_original_attention_{branch}'
            if branch==1:stats=plugin.get_output(1)
            old=tensors[node.output[0]];new=plugin.get_output(0)
            if len(old.shape)!=len(new.shape) or old.dtype!=new.dtype:raise ValueError('Attention output shape/dtype mismatch')
            count=0
            for consumer in layers:
                for slot in range(consumer.num_inputs):
                    existing=consumer.get_input(slot)
                    if existing is not None and existing.name==old.name:consumer.set_input(slot,new);count+=1
            if not count:raise ValueError('Missing attention consumer')
            outputs.append(dict(original=node.output[0],plugin=plugin.name,consumers=count))
        changes.append(dict(scope=name,outputs=outputs))
    return dict(package=package,source_mechanism='Original sequential nonlinear head0 and normal four-head online attention; original unrounded denominator',boundaries=changes)
