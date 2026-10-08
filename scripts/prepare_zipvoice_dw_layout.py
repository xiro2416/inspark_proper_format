"""Lift source depthwise Conv1D to NC1T without changing weights or Q/DQ."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    if args.output.exists():raise FileExistsError('Keep source artifacts; choose a new output graph')
    original = onnx.load(source, load_external_data=False)
    assert not original.graph.value_info
    model = copy.deepcopy(original)
    producers = {o: n for n in model.graph.node for o in n.output}
    initial = {t.name: t for t in model.graph.initializer}
    uses = {}
    for n in model.graph.node:
        for value in n.input:
            uses.setdefault(value, []).append(n.name)
    axis = 'int8_b64_dw_layout_axes2'
    assert axis not in initial
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([2], np.int64), axis))
    additions = []
    modified = set()
    original_dims = {}
    mappings = []
    rebuilt = []
    for node in model.graph.node:
        if node.op_type != 'Conv' or '/depthwise_conv/' not in node.name:
            rebuilt.append(node)
            continue
        prefix = node.name.removesuffix('Conv')
        quantized = producers.get(node.input[1]) is not None
        if quantized:
            weight_dq = producers[node.input[1]]
            act_dq = producers[node.input[0]]
            assert weight_dq.op_type == act_dq.op_type == 'DequantizeLinear'
            quant = producers[act_dq.input[0]]
            assert quant.op_type == 'QuantizeLinear'
            assert uses[quant.output[0]] == [act_dq.name] and uses[act_dq.output[0]] == [node.name]
            assert uses[weight_dq.output[0]] == [node.name]
            assert next(a.i for a in weight_dq.attribute if a.name == 'axis') == 0
            weight = initial[weight_dq.input[0]]
            assert weight.data_type == onnx.TensorProto.INT8
            # Insert before the original Q, retaining all original Q/DQ names,
            # scales, zero points and attributes. Its input is a pure view.
            lifted_input = prefix + 'layout_activation4d'
            extra = onnx.helper.make_node('Unsqueeze', [quant.input[0], axis], [lifted_input], name=prefix + 'layout_unsqueeze')
            where = next(i for i, n in enumerate(rebuilt) if n.name == quant.name)
            rebuilt.insert(where, extra)
            additions.append(extra.name)
            quant.input[0] = lifted_input
            modified.add(quant.name)
        else:
            weight = initial[node.input[1]]
            assert weight.data_type == onnx.TensorProto.FLOAT
            lifted_input = prefix + 'layout_activation4d'
            extra = onnx.helper.make_node('Unsqueeze', [node.input[0], axis], [lifted_input], name=prefix + 'layout_unsqueeze')
            rebuilt.append(extra)
            additions.append(extra.name)
            node.input[0] = lifted_input
        assert len(weight.dims) == 3 and weight.dims[1] == 1
        original_dims[weight.name] = list(weight.dims)
        weight.dims[:] = [weight.dims[0], 1, 1, weight.dims[2]]
        for attribute in node.attribute:
            if attribute.name in ('kernel_shape', 'strides', 'dilations'):
                assert len(attribute.ints) == 1
                attribute.ints[:] = [1, attribute.ints[0]]
            elif attribute.name == 'pads':
                assert len(attribute.ints) == 2
                attribute.ints[:] = [0, attribute.ints[0], 0, attribute.ints[1]]
        output = node.output[0]
        node.output[0] = prefix + 'layout_conv4d'
        rebuilt.append(node)
        extra = onnx.helper.make_node('Squeeze', [node.output[0], axis], [output], name=prefix + 'layout_squeeze')
        rebuilt.append(extra)
        additions.append(extra.name)
        modified.add(node.name)
        mappings.append({'module': prefix, 'quantized': quantized, 'weight': weight.name,
                         'original_weight_dims': original_dims[weight.name], 'lifted_weight_dims': list(weight.dims)})
    assert len(mappings) == 32 and sum(x['quantized'] for x in mappings) == 24
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    restored = copy.deepcopy(model)
    original_nodes = {n.name: n for n in original.graph.node}
    restored_nodes = [copy.deepcopy(original_nodes[n.name] if n.name in modified else n)
                      for n in restored.graph.node if n.name not in additions]
    del restored.graph.node[:]
    restored.graph.node.extend(restored_nodes)
    restored_initial = [t for t in restored.graph.initializer if t.name != axis]
    for tensor in restored_initial:
        if tensor.name in original_dims:
            tensor.dims[:] = original_dims[tensor.name]
    del restored.graph.initializer[:]
    restored.graph.initializer.extend(restored_initial)
    assert restored.SerializeToString() == original.SerializeToString()
    destination = args.output.resolve().parent
    destination.mkdir(parents=True,exist_ok=True)
    for location in {entry.value for t in model.graph.initializer for entry in t.external_data if entry.key == 'location'}:
        if source.parent != destination:
            shutil.copyfile(source.parent / location, destination / location)
        assert sha(source.parent / location) == sha(destination / location)
    path = args.output.resolve()
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    proof = {'status': 'source_layout_inverse_protobuf_exact', 'source': str(path),
             'source_sha256': sha(path), 'original_source_sha256': sha(source),
             'source_restore_exact': True, 'weights_values_bytes_unchanged': True,
             'qparams_unchanged': True, 'original_position_nodes_unchanged': True,
             'protected_precision_unchanged': True, 'mappings': mappings,
             'performance_acceptance': False,
             'scope': 'Only singleton spatial-axis views, Conv attributes and weight dimensions change. Entire original protobuf restores exactly; external weight payload bytes identical. Full-model execution remains pending.'}
    path.with_suffix('.layout.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({'status': proof['status'], 'depthwise_modules': len(mappings), 'int8': 24, 'float_protected': 8}))


if __name__ == '__main__':
    main()
