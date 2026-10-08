"""Original last12 signed-INT8 projection and tanh/value/gate boundary replacement."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import onnx

def rewrite(network, source, trt):
    import tensorrt.plugin as trtp
    from . import int8_nonlinear_value_plugin
    model = onnx.load(source, load_external_data=False)
    named = {n.name: n for n in model.graph.node}
    producers = {o: n for n in model.graph.node for o in n.output}
    initial = {t.name: t for t in model.graph.initializer}

    def value(name):
        if name in initial:
            t = copy.deepcopy(initial[name])
            if t.external_data:
                onnx.external_data_helper.load_external_data_for_tensor(t, str(Path(source).parent))
            return onnx.numpy_helper.to_array(t).copy()
        n = producers[name]
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
        if not mm.name.endswith('/nonlin_attention/in_proj/MatMul'):
            continue
        p = mm.name.removesuffix('in_proj/MatMul')
        qp = p + 'in_proj/QuantizeLinear'
        if qp not in named:
            continue
        quant = named[qp]
        dq = named[p + 'in_proj/DequantizeLinear']
        wdq = named[p + 'in_proj/DequantizeLinear_1']
        add = named[p + 'in_proj/Add']
        assert dq.input[0] == quant.output[0] and np.array_equal(value(quant.input[1]), value(dq.input[1]))
        assert np.all(value(quant.input[2]) == 0) and np.all(value(dq.input[2]) == 0) and np.all(value(wdq.input[2]) == 0)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in wdq.attribute}
        assert attrs.get('axis', 1) == 0
        w = value(wdq.input[0])
        as_ = value(dq.input[1])
        ws = value(wdq.input[1])
        bias = value(next((s for s in add.input if s != mm.output[0])))
        assert w.dtype == np.int8 and w.shape == (1152, 512) and (as_.size == 1) and (float(as_) > 0)
        assert as_.dtype == ws.dtype == bias.dtype == np.float32 and ws.shape == bias.shape == (1152,)
        assert named[p + 'tanh/Tanh'].input[0] == named[p + 'Slice'].output[0] and named[p + 'Mul_3'].input[0] == named[p + 'Slice_1'].output[0]
        assert list(next((a.ints for a in named[p + 'Transpose'].attribute if a.name == 'perm'))) == [2, 1, 0, 3]
        for part, label in enumerate(('Slice', 'Slice_1', 'Slice_2')):
            n = named[p + label]
            assert n.input[0] == add.output[0] and value(n.input[3]).tolist() == [2]
        q = tensors[quant.output[0]]
        assert q.dtype == trt.int8 and len(q.shape) == 3 and (q.shape[1] in (-1, 64)) and (q.shape[2] == 512), (p, str(q.dtype), list(q.shape))
        parsed_q_shape = list(q.shape)
        view = network.add_shuffle(q)
        view.name = p + 'original_Q8_B64_view'
        view.reshape_dims = (-1, 64, 512)
        q = view.get_output(0)
        args = [q]
        hashes = {}
        for label, array in [('weight', w), ('input_scale', as_.reshape(1)), ('weight_scale', ws), ('bias', bias)]:
            c = network.add_constant(array.shape, array)
            c.name = mm.name + '/original_' + label
            args.append(c.get_output(0))
            hashes[label] = hashlib.sha256(array.tobytes()).hexdigest()
        factory = trtp.op.zipvoice_int8_b64_normtf32k16.OriginalInt8NonlinearValue128x32(*args)
        plugin = network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT))
        plugin.name = p + 'in_proj/OriginalInt8NonlinearValuePlugin'
        outs = []
        for slot, name in enumerate((named[p + 'Transpose'].output[0], named[p + 'Slice_2'].output[0])):
            old = tensors[name]
            new = plugin.get_output(slot)
            new.name = p + ('original_integer_value' if slot == 0 else 'original_integer_gate')
            assert old.dtype == new.dtype == trt.float32 and len(old.shape) == len(new.shape) and all((a == -1 or b == -1 or a == b for a, b in zip(old.shape, new.shape))), (p, slot, list(old.shape), list(new.shape))
            consumers = []
            for l in layers:
                for i in range(l.num_inputs):
                    t = l.get_input(i)
                    if t is not None and t.name == old.name:
                        l.set_input(i, new)
                        consumers.append({'layer': l.name, 'slot': i})
            assert consumers
            outs.append({'original_tensor': name, 'new_tensor': new.name, 'shape': list(new.shape), 'consumers': consumers})
        changes.append({'module': p + 'in_proj/', 'original_matmul': mm.name, 'quantized_input': quant.output[0], 'parsed_Q8_shape': parsed_q_shape, 'canonical_Q8_shape': [-1, 64, 512], 'plugin': plugin.name, 'parameter_hashes': hashes, 'outputs': outs, 'formula': 'value=tanh(z0)*z1 in originalBT384; gate=z2 inoriginalTB384; originalinteger projection scales/bias'})
    assert len(changes) == 12
    return changes
