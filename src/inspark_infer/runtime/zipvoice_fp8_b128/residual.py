"""Verified original residual boundaries and explicit FP8 projection fusion."""
import hashlib
import numpy as np
import onnx


def contracts(graph):
    by={o:n for n in graph.graph.node for o in n.output};initial={t.name:t for t in graph.graph.initializer}
    def array(name):
        if name in initial:return onnx.numpy_helper.to_array(initial[name])
        n=by[name];assert n.op_type=='Constant'
        return onnx.numpy_helper.to_array(next(a.t for a in n.attribute if a.type==onnx.AttributeProto.TENSOR))
    def unslice(name):
        n=by.get(name)
        if n and n.op_type=='Slice':
            starts,ends,axis,step=[array(x).tolist() for x in n.input[1:]]
            if starts==[0] and ends==[512] and axis==[-1] and step==[1]:return n.input[0]
        return name
    result=[];seen=set()
    for quant in graph.graph.node:
        if quant.op_type!='QuantizeLinear':continue
        addition=by.get(quant.input[0])
        if addition is None or addition.op_type!='Add':continue
        for value in addition.input:
            biasadd=by.get(unslice(value))
            if biasadd is None or biasadd.op_type!='Add':continue
            bias=next((x for x in biasadd.input if x in initial and x.endswith('.bias')),None)
            if bias is None:continue
            dot=by.get(next(x for x in biasadd.input if x!=bias))
            if dot is None or dot.op_type!='MatMul':continue
            activation=by.get(dot.input[0]);weight=by.get(dot.input[1])
            if weight and weight.op_type=='Transpose':weight=by.get(weight.input[0])
            if not(activation and weight and activation.op_type==weight.op_type=='DequantizeLinear' and weight.input[0] in initial and initial[weight.input[0]].data_type==17):continue
            if addition.name in seen:raise ValueError('Ambiguous multiple residual quantization boundaries')
            seen.add(addition.name)
            result.append(dict(name=addition.name,projection=dot.name,a=activation.input[0],w=weight.input[0],bias=bias,res=next(x for x in addition.input if x!=value),original=addition.output[0],q=quant.output[0],alpha=float(np.float32(float(array(activation.input[1]))*float(array(weight.input[1])))),inv=float(np.float32(1/float(array(quant.input[1])))),weight_sha256=hashlib.sha256(array(weight.input[0]).tobytes()).hexdigest(),k=int(initial[weight.input[0]].dims[1]),n=int(initial[weight.input[0]].dims[0])))
    if not result:raise ValueError('No eligible original residual boundaries')
    return result


_WEIGHTS=[]

def rewrite(network,graph,trt,ffn_only=False):
    import tensorrt.plugin as trtp
    from inspark_infer.ops.tensorrt.zipvoice_fp8_b128.b128 import residual_plugin
    old_layers=[network.get_layer(i) for i in range(network.num_layers)]
    tensors={network.get_input(i).name:network.get_input(i) for i in range(network.num_inputs)}
    tensors.update({l.get_output(i).name:l.get_output(i) for l in old_layers for i in range(l.num_outputs)})
    initial={t.name:t for t in graph.graph.initializer}
    changes=[]
    for c in contracts(graph):
        if ffn_only and '.feed_forward1.out_proj.bias' not in c['bias']:continue
        if c['w'] not in tensors:
            raw=np.ascontiguousarray(onnx.numpy_helper.to_array(initial[c['w']])).view(np.uint8)
            _WEIGHTS.append(raw)
            weights=trt.Weights(trt.fp8,raw.ctypes.data,raw.size)
            constant=network.add_constant(tuple(initial[c['w']].dims),weights)
            constant.name=c['name']+'/original_fp8_weight';tensors[c['w']]=constant.get_output(0)
        if c['bias'] not in tensors:
            bias=np.ascontiguousarray(onnx.numpy_helper.to_array(initial[c['bias']]))
            _WEIGHTS.append(bias)
            constant=network.add_constant(bias.shape,bias)
            constant.name=c['name']+'/original_fp32_bias';tensors[c['bias']]=constant.get_output(0)
        values=[tensors[c[x]] for x in ['a','w','bias','res']]
        assert values[0].dtype==values[1].dtype==trt.fp8 and values[2].dtype==values[3].dtype==trt.float32
        factory=trtp.op.zipvoice_fp8_sm120_b128_residual.FusedProjectionResidual(*values,alpha=c['alpha'],inv=c['inv'])
        layer=network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT));layer.name=c['name']+'/fp8_projection_residual_dual_output'
        counts=[]
        for key,slot in [('original',0),('q',1)]:
            old=tensors[c[key]];new=layer.get_output(slot);new.name=c[key]+'/residual_fused'
            assert old.dtype==new.dtype and len(old.shape)==len(new.shape)
            count=0
            for consumer in old_layers:
                for i in range(consumer.num_inputs):
                    t=consumer.get_input(i)
                    if t is not None and t.name==old.name:consumer.set_input(i,new);count+=1
            if old.is_network_output:network.unmark_output(old);network.mark_output(new)
            if not count:raise ValueError('Missing original residual consumer')
            counts.append(count)
            # Later original boundaries can depend on the previous replaced sum.
            tensors[c[key]]=new
        changes.append({**c,'consumers':counts})
    return dict(mechanism='Original FP8 projection plus FP32 bias/residual and next unchanged per-tensor quantization',boundaries=changes)
