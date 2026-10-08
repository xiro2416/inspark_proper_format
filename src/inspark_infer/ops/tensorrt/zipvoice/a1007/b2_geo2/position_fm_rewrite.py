"""Audit original FM paths and replace only Float32 position/softmax regions.

The CPU audit can run while GPU validation is active. Network rewriting is
experimental: it requires a successful fragment check before any full build.
"""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'optimization/int8_sm89_b64'

def audit(source):
    import onnx
    model = onnx.load(source, load_external_data=False)
    producers = {o: n for n in model.graph.node for o in n.output}
    named = {n.name: n for n in model.graph.node}
    initializers = {t.name: t for t in model.graph.initializer}

    def constant(name):
        if name in initializers:
            return onnx.numpy_helper.to_array(initializers[name]).tolist()
        node = producers[name]
        if node.op_type == 'Identity':
            return constant(node.input[0])
        assert node.op_type == 'Constant'
        return onnx.numpy_helper.to_array(next((a.t for a in node.attribute if a.name == 'value'))).tolist()
    boundaries = []
    for node in model.graph.node:
        if not node.name.endswith('/self_attn_weights/Softmax'):
            continue
        prefix = node.name[:-len('Softmax')]
        domain = prefix.removesuffix('self_attn_weights/')
        scores = named[prefix + 'MatMul']
        position = named[prefix + 'MatMul_1']
        where = producers[node.input[0]]
        assert where.op_type == 'Where' and constant(where.input[1]) == -1000.0
        add = producers[where.input[2]]
        assert add.op_type == 'Add' and scores.output[0] in add.input
        gather = producers[next((x for x in add.input if x != scores.output[0]))]
        assert gather.op_type == 'GatherElements' and gather.input[0] == position.output[0]
        assert next((a.i for a in gather.attribute if a.name == 'axis')) in (-1, 3)
        embedding_transpose = producers[position.input[1]]
        assert embedding_transpose.op_type == 'Transpose'
        assert list(next((a.ints for a in embedding_transpose.attribute if a.name == 'perm'))) == [2, 0, 3, 1]
        embedding_reshape = producers[embedding_transpose.input[0]]
        assert embedding_reshape.op_type == 'Reshape'
        embedding_dims = producers[embedding_reshape.input[1]]
        assert embedding_dims.op_type == 'Concat'
        assert [constant(embedding_dims.input[i]) for i in (0, 2, 3)] == [[-1], [4], [4]]
        projection = producers[embedding_reshape.input[0]]
        assert projection.op_type == 'MatMul'
        position_input = projection.input[0]
        while producers[position_input].op_type in ('DequantizeLinear', 'QuantizeLinear', 'Mul', 'Identity'):
            position_input = producers[position_input].input[0]
        position_unsqueeze = producers[position_input]
        assert position_unsqueeze.op_type == 'Unsqueeze'
        assert constant(position_unsqueeze.input[1]) == [0]
        cast = producers[where.input[0]]
        assert cast.op_type == 'Cast' and next((a.i for a in cast.attribute if a.name == 'to')) == 9
        unsqueeze = producers[cast.input[0]]
        assert unsqueeze.op_type == 'Unsqueeze' and constant(unsqueeze.input[1]) == [1]
        mask = unsqueeze.input[0]
        if mask == 'padding_mask':
            factor = 1
        else:
            sliced = producers[mask]
            assert sliced.op_type == 'Slice' and sliced.input[0] == 'padding_mask'
            starts, ends, axes, steps = [constant(x) for x in sliced.input[1:]]
            assert starts == [0] and ends == [2 ** 63 - 1] and (axes == [1]) and (steps in ([2], [4]))
            factor = steps[0]
        head0 = named[domain + 'Slice']
        assert head0.input[0] == node.output[0]
        assert [constant(x) for x in head0.input[1:]] == [[0], [1], [0], [1]]
        for branch in ('self_attn1', 'self_attn2'):
            assert named[domain + branch + '/MatMul'].input[0] == node.output[0]
        assert named[domain + 'nonlin_attention/MatMul'].input[0] == head0.output[0]
        consumers = [n.name for n in model.graph.node if node.output[0] in n.input]
        assert set(consumers) == {domain + 'Slice', domain + 'self_attn1/Shape_2', domain + 'self_attn1/MatMul', domain + 'self_attn2/MatMul'}
        boundaries.append(dict(domain=prefix, query=position.input[0], embedding=position.input[1], scores=scores.output[0], mask=mask, probability_output=node.output[0], factor=factor, frame_capacity=(1750 + factor - 1) // factor, consumers=consumers, all_three_data_branches_preserved=True, source_position_singleton_axis_proven=True))
    assert len(boundaries) == 16
    return {'status': 'source_paths_audited_full_engine_pending', 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(), 'boundaries': boundaries, 'quantized_modules_changed': 0, 'performance_acceptance': False}
