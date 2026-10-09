"""Selective source floating fusion with original weights/bias/SwooshL."""
import numpy as np
import onnx
_KEEP=[]

def rewrite(network,graph,trt):
    import tensorrt.plugin as p
    from inspark_infer.ops.tensorrt.zipvoice_fp8_b128.b128 import protected_ffn_plugin
    initial={t.name:t for t in graph.graph.initializer}
    layers=[network.get_layer(i) for i in range(network.num_layers)]
    tensors={network.get_input(i).name:network.get_input(i) for i in range(network.num_inputs)}
    tensors.update({l.get_output(i).name:l.get_output(i) for l in layers for i in range(l.num_outputs)})
    targets=[(f'model.fm_decoder.encoders.0.layers.{i}.feed_forward1',64) for i in (0,1)]+[(f'model.fm_decoder.encoders.1.encoder.layers.{i}.feed_forward3',128) for i in (0,1)]
    result=[]
    for scope,bm in targets:
        nodes=[n for n in graph.graph.node if scope in next((a.value for a in n.metadata_props if a.key=='namespace'),'')]
        mm=next(n for n in nodes if n.op_type=='MatMul' and scope+'.in_proj' in next((a.value for a in n.metadata_props if a.key=='namespace'),''))
        out=next(n for n in nodes if n.op_type=='MatMul' and scope+'.out_proj' in next((a.value for a in n.metadata_props if a.key=='namespace'),''))
        w=np.ascontiguousarray(onnx.numpy_helper.to_array(initial[mm.input[1]]))
        bias=np.ascontiguousarray(onnx.numpy_helper.to_array(initial[scope+'.in_proj.bias']))
        assert w.dtype==bias.dtype==np.float32 and w.shape[0]==512
        _KEEP.extend([w,bias]);wc=network.add_constant(w.shape,w).get_output(0);bc=network.add_constant(bias.shape,bias).get_output(0)
        factory=p.op.zipvoice_fp8_sm120_b128_protected.F32SwooshL(tensors[mm.input[0]],wc,bc,width=w.shape[1],bm=bm)
        layer=network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT));layer.name=scope+'/original_floating_projection_swoosh'
        old=tensors[out.input[0]];new=layer.get_output(0);new.name=old.name+'/protected_fused';assert old.dtype==new.dtype
        count=0
        for consumer in layers:
            for slot in range(consumer.num_inputs):
                t=consumer.get_input(slot)
                if t is not None and t.name==old.name:consumer.set_input(slot,new);count+=1
        if not count:raise ValueError('Missing protected activation consumer')
        result.append(dict(scope=scope,bm=bm,width=w.shape[1],consumers=count,weights_unchanged=True,original_activation='SwooshL4/.08/.035 FP32; original TF32RNA projection'))
    return dict(boundaries=result)
