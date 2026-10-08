"""Equivalent lowering of Dynamo tensor-split sequences for TensorRT parsing."""


def lower_sequences(graph):
    import numpy as np
    import onnx
    from onnx import helper,numpy_helper
    constants={x.name:numpy_helper.to_array(x) for x in graph.graph.initializer if x.data_type in [6,7]}
    for node in graph.graph.node:
        if node.op_type=='Constant':
            for a in node.attribute:
                if a.name=='value' and a.t.data_type in [6,7]:constants[node.output[0]]=numpy_helper.to_array(a.t)
    splits={n.output[0]:n for n in graph.graph.node if n.op_type=='SplitToSequence'}
    consumers={name:[] for name in splits}
    for node in graph.graph.node:
        for name in node.input:
            if name in consumers:consumers[name].append(node)
    for name,nodes in consumers.items():
        if any(n.op_type!='SequenceAt' for n in nodes):raise ValueError('Non-index sequence consumer: '+name)
    nodes=[];count=0
    for node in graph.graph.node:
        if node.op_type=='SplitToSequence':continue
        if node.op_type!='SequenceAt' or node.input[0] not in splits:
            nodes.append(node);continue
        split=splits[node.input[0]]
        attrs={a.name:helper.get_attribute_value(a) for a in split.attribute}
        if attrs.get('keepdims',1)!=1 or len(split.input)!=2:raise ValueError('Unsupported tensor-split contract')
        if node.input[1] not in constants:raise ValueError('Sequence index is not constant')
        index=int(np.asarray(constants[node.input[1]]).item())
        if index<0:raise ValueError('Negative sequence index needs a distinct lowering')
        prefix=f'fp8_sequence_{count}';count+=1
        def const(value,suffix):
            name=prefix+'_'+suffix
            nodes.append(helper.make_node('Constant',[],[name],name=name,
                          value=numpy_helper.from_array(np.asarray(value,dtype=np.int64))))
            return name
        axis=const([attrs.get('axis',0)],'axis');steps=const([1],'step')
        if split.input[1] in constants:
            size=np.asarray(constants[split.input[1]])
            if size.ndim==0:
                start=index*int(size);end=(index+1)*int(size)
            else:
                if index>=size.size:raise ValueError('Sequence index out of range')
                start=int(size[:index].sum());end=int(size[:index+1].sum())
            starts=const([start],'starts');ends=const([end],'ends')
        else:
            # This model emits scalar symbolic split sizes. Multiplication
            # preserves the split bounds for every dynamic frame length.
            zero_axis=const([0],'unsqueeze_axis')
            starts_scalar=prefix+'_start_scalar';ends_scalar=prefix+'_end_scalar'
            nodes.append(helper.make_node('Mul',[split.input[1],const(index,'index')],[starts_scalar],name=starts_scalar))
            nodes.append(helper.make_node('Mul',[split.input[1],const(index+1,'next_index')],[ends_scalar],name=ends_scalar))
            starts=prefix+'_starts';ends=prefix+'_ends'
            nodes.append(helper.make_node('Unsqueeze',[starts_scalar,zero_axis],[starts],name=starts))
            nodes.append(helper.make_node('Unsqueeze',[ends_scalar,zero_axis],[ends],name=ends))
        replacement=helper.make_node('Slice',[split.input[0],starts,ends,axis,steps],node.output,name=node.name)
        replacement.metadata_props.extend(node.metadata_props);nodes.append(replacement)
    del graph.graph.node[:];graph.graph.node.extend(nodes)
    live={v for n in nodes for v in [*n.input,*n.output]}|{x.name for x in graph.graph.input}|{x.name for x in graph.graph.output}
    infos=[v for v in graph.graph.value_info if v.name in live and not v.type.HasField('sequence_type')]
    del graph.graph.value_info[:];graph.graph.value_info.extend(infos)
    return dict(split_sequences=len(splits),indexed_slices=count,quantizers=sum(n.op_type=='QuantizeLinear' for n in nodes))


def fold_shape_constants(graph):
    """Fold importer-required constants while preserving every FP8 Q/DQ site."""
    from onnxscript import optimizer,ir
    model=ir.from_proto(graph)
    optimizer.fold_constants(model,onnx_shape_inference=True,
        should_fold=lambda node:False if node.op_type in ('QuantizeLinear','DequantizeLinear') else None)
    optimizer.remove_unused_nodes(model)
    return ir.to_proto(model)
