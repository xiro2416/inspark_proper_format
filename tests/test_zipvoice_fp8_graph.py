"""Sequence lowering preserves constant/dynamic split boundaries and FP8 Q/DQ."""
import numpy as np
import onnx
from onnx import helper,numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest

from inspark_infer.runtime.zipvoice_fp8.graph import lower_sequences,fold_shape_constants


@pytest.mark.parametrize('split,index,shape',[(3,0,(2,8)),(3,2,(2,8)),([2,3,3],1,(2,8)),('dynamic',0,(2,8)),('dynamic',1,(2,8))])
def test_slice_matches_sequence(split,index,shape):
    nodes=[];initializers=[numpy_helper.from_array(np.array(index,np.int64),'idx')]
    if split=='dynamic':
        nodes.append(helper.make_node('Constant',[],['size'],value=numpy_helper.from_array(np.array(3,np.int64))))
        # A runtime input exercises the non-constant split-size path.
        split_name='chunk'
    else:
        split_name='chunk';initializers.append(numpy_helper.from_array(np.array(split,np.int64),split_name))
    nodes += [helper.make_node('SplitToSequence',['x',split_name],['parts'],axis=-1,keepdims=1),
              helper.make_node('SequenceAt',['parts','idx'],['y'])]
    inputs=[helper.make_tensor_value_info('x',onnx.TensorProto.FLOAT,list(shape))]
    if split=='dynamic':inputs.append(helper.make_tensor_value_info('chunk',onnx.TensorProto.INT64,[]))
    g=helper.make_graph(nodes,'split',inputs,[helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[2,None])],initializers)
    m=helper.make_model(g,opset_imports=[helper.make_opsetid('',21)]);m.ir_version=10
    values={'x':np.arange(np.prod(shape),dtype=np.float32).reshape(shape)}
    if split=='dynamic':values['chunk']=np.array(3,np.int64)
    expected=ReferenceEvaluator(m).run(None,values)[0]
    lower_sequences(m);onnx.checker.check_model(m)
    actual=ReferenceEvaluator(m).run(None,values)[0]
    assert np.array_equal(expected,actual)
    assert not any(n.op_type in ('SequenceAt','SplitToSequence') for n in m.graph.node)
