import hashlib
from pathlib import Path
import onnx
import numpy as np

def rewrite(network, source, trt, micro=False, ff=2):
    import tensorrt.plugin as trtp
    from . import f32_tf32_rna_plugin
    model = onnx.load(source, load_external_data=False)
    named = {n.name: n for n in model.graph.node}
    prod = {o: n for n in model.graph.node for o in n.output}
    initial = {t.name: t for t in model.graph.initializer}

    def array(name):
        if name in initial:
            t = initial[name]
            if t.external_data:
                onnx.external_data_helper.load_external_data_for_tensor(t, str(Path(source).parent))
            return onnx.numpy_helper.to_array(t).copy()
        n = prod[name]
        if n.op_type == 'Constant':
            return onnx.numpy_helper.to_array(onnx.helper.get_attribute_value(n.attribute[0]))
        assert n.op_type == 'Identity'
        return array(n.input[0])
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    tensors = {l.get_output(i).name: l.get_output(i) for l in layers for i in range(l.num_outputs)}
    tensors.update({network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)})
    changes = []
    prefixes = [f'/fm_decoder/encoders.0/layers.0/feed_forward{ff}/'] if micro else [f'/fm_decoder/{encoder}/layers.{layer}/feed_forward{f}/' for encoder in ('encoders.0', 'encoders.1/encoder') for layer in (0, 1) for f in (1, 2, 3)]
    for prefix in prefixes:
        family = int(prefix.split('feed_forward')[1].split('/')[0])
        width = {1: 1152, 2: 1536, 3: 1920}[family]
        mm = named[prefix + 'in_proj/MatMul']
        add = named[prefix + 'in_proj/Add']
        end = named[prefix + 'out_proj/Sub_3']
        w = array(mm.input[1])
        bias = array(next((v for v in add.input if v != mm.output[0])))
        assert w.shape == (512, width) and bias.shape == (width,) and (w.dtype == bias.dtype == np.float32)
        constants = [float(array(named[prefix + 'out_proj/' + n].output[0])) for n in ('Constant', 'Constant_1', 'Constant_2', 'Constant_3', 'Constant_4', 'Constant_5')]
        assert constants == [4.0, 0.0, 0.0, 1.0, float(np.float32(0.08)), float(np.float32(0.035))]
        x = tensors[mm.input[0]]
        old = tensors[end.output[0]]
        assert x.dtype == old.dtype == trt.float32 and len(x.shape) == 3 and (x.shape[-1] == 512)
        args = [x]
        for label, value in [('weight', w), ('bias', bias)]:
            c = network.add_constant(value.shape, value)
            c.name = prefix + 'original_' + label
            args.append(c.get_output(0))
        op = getattr(trtp.op.zipvoice_int8_b8_geo1, 'F32TF32RNAFF' + str(family))
        f = op(*args)
        plugin = network.add_plugin_v3(*f(trt.QuickPluginCreationRequest.STRICT_AOT))
        plugin.name = prefix + 'TF32RNAOriginalActivation'
        new = plugin.get_output(0)
        new.name = prefix + 'fused_original_activation'
        consumers = []
        for l in layers:
            for slot in range(l.num_inputs):
                t = l.get_input(slot)
                if t is not None and t.name == old.name:
                    l.set_input(slot, new)
                    consumers.append(l.name)
        if old.is_network_output:
            network.unmark_output(old)
            network.mark_output(new)
        else:
            assert consumers
        changes.append({'module': prefix, 'input': mm.input[0], 'output': end.output[0], 'consumers': consumers, 'weight_sha256': hashlib.sha256(w.tobytes()).hexdigest(), 'bias_sha256': hashlib.sha256(bias.tobytes()).hexdigest(), 'math': 'OriginalFloat32 payloads/accumulation; explicit TF32 nearest rounding matches nativeTF32; original bias/activation and INT8 recipe unchanged'})
    return changes
