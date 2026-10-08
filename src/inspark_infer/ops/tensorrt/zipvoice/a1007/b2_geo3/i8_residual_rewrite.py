"""24 original INT8 attention output projections plus their original residual Add."""
import copy
import hashlib
import re
from pathlib import Path
import numpy as np
import onnx

def rewrite(network, source, trt):
    import tensorrt.plugin as trtp
    from . import i8_residual_plugin
    model = onnx.load(source, load_external_data=False)
    named = {n.name: n for n in model.graph.node}
    prod = {o: n for n in model.graph.node for o in n.output}
    initial = {t.name: t for t in model.graph.initializer}

    def value(name):
        if name in initial:
            t = copy.deepcopy(initial[name])
            if t.external_data:
                onnx.external_data_helper.load_external_data_for_tensor(t, str(Path(source).parent))
            return onnx.numpy_helper.to_array(t).copy()
        n = prod[name]
        if n.op_type == 'Identity':
            return value(n.input[0])
        if n.op_type == 'Cast':
            return value(n.input[0]).astype(onnx.helper.tensor_dtype_to_np_dtype(next((a.i for a in n.attribute if a.name == 'to'))))
        assert n.op_type == 'Constant'
        return onnx.numpy_helper.to_array(next((a.t for a in n.attribute if a.name == 'value')))
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    tensors = {l.get_output(i).name: l.get_output(i) for l in layers for i in range(l.num_outputs)}
    changes = []
    for mm in model.graph.node:
        match = re.match('^(.*)/self_attn([12])/out_proj/MatMul$', mm.name)
        if not match:
            continue
        parent, which = match.groups()
        p = mm.name.removesuffix('MatMul')
        if p + 'QuantizeLinear' not in named:
            continue
        quant = named[p + 'QuantizeLinear']
        dq = named[p + 'DequantizeLinear']
        wdq = named[p + 'DequantizeLinear_1']
        biasadd = named[p + 'Add']
        resadd = named[parent + ('/Add_3' if which == '1' else '/Add_7')]
        assert dq.input[0] == quant.output[0] and mm.input[0] == dq.output[0]
        transpose = prod[mm.input[1]]
        assert transpose.op_type == 'Transpose' and transpose.input[0] == wdq.output[0] and (list(next((a.ints for a in transpose.attribute if a.name == 'perm'))) == [1, 0])
        assert np.array_equal(value(quant.input[1]), value(dq.input[1])) and all((np.all(value(n.input[2]) == 0) for n in (quant, dq, wdq)))
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in wdq.attribute}
        assert attrs.get('axis', 1) == 0
        assert biasadd.op_type == resadd.op_type == 'Add' and mm.output[0] in biasadd.input and (biasadd.output[0] in resadd.input)
        residual = next((x for x in resadd.input if x != biasadd.output[0]))
        w = value(wdq.input[0])
        as_ = value(dq.input[1])
        ws = value(wdq.input[1])
        bias = value(next((x for x in biasadd.input if x != mm.output[0])))
        assert w.dtype == np.int8 and w.shape == (512, 48) and (as_.size == 1) and (float(as_) > 0) and (as_.dtype == ws.dtype == bias.dtype == np.float32) and (ws.shape == bias.shape == (512,))
        q = tensors[quant.output[0]]
        r = tensors[residual]
        assert q.dtype == trt.int8 and r.dtype == trt.float32 and (len(q.shape) == len(r.shape) == 3) and (q.shape[1] in (-1, 2)) and (q.shape[2] == 48) and (r.shape[1] in (-1, 2)) and (r.shape[2] == 512), (p, list(q.shape), list(r.shape))
        parsed = {'q8': list(q.shape), 'residual': list(r.shape)}
        qview = network.add_shuffle(q)
        qview.name = p + 'original_Q8_B2_view'
        qview.reshape_dims = (-1, 2, 48)
        q = qview.get_output(0)
        rview = network.add_shuffle(r)
        rview.name = p + 'original_residual_B2_view'
        rview.reshape_dims = (-1, 2, 512)
        r = rview.get_output(0)
        args = [q]
        hashes = {}
        for label, array in [('weight', w), ('input_scale', as_.reshape(1)), ('weight_scale', ws), ('bias', bias)]:
            l = network.add_constant(array.shape, array)
            l.name = p + 'original_' + label
            args.append(l.get_output(0))
            hashes[label] = hashlib.sha256(array.tobytes()).hexdigest()
        factory = trtp.op.zipvoice_int8_b2_geo3.OriginalInt8Residual32x64(*args, r)
        plugin = network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT))
        plugin.name = p + 'OriginalInt8ResidualPlugin'
        old = tensors[resadd.output[0]]
        new = plugin.get_output(0)
        new.name = parent + ('/original_integer_residual1' if which == '1' else '/original_integer_residual2')
        assert old.dtype == new.dtype == trt.float32 and len(old.shape) == len(new.shape) and all((a == -1 or b == -1 or a == b for a, b in zip(old.shape, new.shape)))
        consumers = []
        for l in layers:
            for i in range(l.num_inputs):
                t = l.get_input(i)
                if t is not None and t.name == old.name:
                    l.set_input(i, new)
                    consumers.append({'layer': l.name, 'slot': i})
        assert consumers
        changes.append({'module': p, 'original_matmul': mm.name, 'original_residual_add': resadd.name, 'original_residual_tensor': residual, 'quantized_input': quant.output[0], 'parsed_shapes': parsed, 'canonical_Q8_shape': [-1, 2, 48], 'canonical_residual_shape': [-1, 2, 512], 'plugin': plugin.name, 'parameter_hashes': hashes, 'output': {'original_tensor': old.name, 'new_tensor': new.name, 'shape': list(new.shape), 'consumers': consumers}, 'formula': 'int32(q8@originalWi8.T).F32*(originalAS*WS)+originalBias+originalResidual'})
    assert len(changes) == 24
    return changes
