"""Replace original three AV outputs with sequential independent nodes, preserving all upstream projections and shape consumers."""
import hashlib
from pathlib import Path
import onnx
from . import position_fm_rewrite

def rewrite(network, source, trt):
    import tensorrt.plugin as trtp
    from . import normal_tf32_plugin
    from . import online_nonlinear_rna_plugin
    model = onnx.load(source, load_external_data=False)
    named = {n.name: n for n in model.graph.node}
    contract = position_fm_rewrite.audit(source)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    tensors = {network.get_input(i).name: network.get_input(i) for i in range(network.num_inputs)}
    tensors.update({l.get_output(i).name: l.get_output(i) for l in layers for i in range(l.num_outputs)})
    changes = []
    for b in contract['boundaries']:
        domain = b['domain'].removesuffix('self_attn_weights/')
        score = named[b['domain'] + 'MatMul']
        branches = [named[domain + name + '/MatMul'] for name in ('nonlin_attention', 'self_attn1', 'self_attn2')]
        names = [score.input[0], score.input[1], b['query'], b['embedding'], b['mask']] + [n.input[1] for n in branches]
        expected = [(4, 1, -1, 32), (4, 1, 32, -1), (4, 1, -1, 4), (4, 1, 4, -1), (1, -1)]
        common = []
        for i, (name, shape) in enumerate(zip(names[:5], expected)):
            value = tensors[name]
            assert value.dtype == (trt.bool if i == 4 else trt.float32)
            view = network.add_shuffle(value)
            view.name = domain + 'online_common_view_' + str(i)
            view.reshape_dims = shape
            common.append(view.get_output(0))
        outputs = []
        stats = None
        for i, node in enumerate(branches):
            value = tensors[node.input[1]]
            assert value.dtype == trt.float32
            view = network.add_shuffle(value)
            view.name = domain + 'online_value_view_' + str(i)
            view.reshape_dims = (1, 1, -1, 384) if i == 0 else (4, 1, -1, 12)
            op = trtp.op.zipvoice_int8_b1.OnlineNonlinearRNAFloat32 if i == 0 else trtp.op.zipvoice_int8_b1_normal_tf32.OnlineNormalWideStatsWriteFloat32 if i == 1 else trtp.op.zipvoice_int8_b1_normal_tf32.OnlineNormalWideStatsReadFloat32
            ins = [*common, view.get_output(0)]
            if i == 2:
                ins.append(stats)
            factory = op(*ins)
            plugin = network.add_plugin_v3(*factory(trt.QuickPluginCreationRequest.STRICT_AOT))
            plugin.name = domain + 'OnlineBranchFloat32_' + str(i)
            if i == 1:
                stats = plugin.get_output(1)
                stats.name = domain + 'original_probability_row_max_den'
            old = tensors[node.output[0]]
            new = plugin.get_output(0)
            new.name = domain + 'original_online_AV_' + str(i)
            old_shape = list(old.shape)
            new_shape = list(new.shape)
            assert old.dtype == new.dtype == trt.float32 and len(old_shape) == len(new_shape), (node.name, old_shape, new_shape)
            consumers = []
            for layer in layers:
                for slot in range(layer.num_inputs):
                    v = layer.get_input(slot)
                    if v is not None and v.name == old.name:
                        layer.set_input(slot, new)
                        consumers.append({'layer': layer.name, 'slot': slot})
            assert consumers, (node.name, 'missingoriginalAVconsumer')
            outputs.append({'source_node': node.name, 'original_output': node.output[0], 'consumers': consumers, 'old_shape': old_shape, 'new_shape': new_shape, 'plugin': plugin.name, 'row_stats_mode': i if i else 0})
        changes.append({'domain': domain, 'inputs': names, 'factor': b['factor'], 'outputs': outputs, 'nonlinear_original_head0_only': True, 'all4normal_heads_preserved': True})
    assert len(changes) == 16 and sum((len(c['outputs']) for c in changes)) == 48
    return changes
