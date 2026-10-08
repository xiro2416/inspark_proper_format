"""Prove the rewrite keeps channel/tap ordering and padded edge windows."""
import sys
from pathlib import Path
import numpy as np
import pytest
import onnx_graphsurgeon as gs
from onnx.reference import ReferenceEvaluator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from canonicalize_nvfp4_windows import rewrite

@pytest.mark.parametrize('taps,dilation,stride',[(3,1,1),(7,3,1),(12,1,2)])
def test_single_window_gather(taps,dilation,stride):
    length=43;positions=np.arange((length-dilation*(taps-1)-1)//stride+1)*stride
    x=gs.Variable('x',dtype=np.float32,shape=[2,3,length]);windows=[];nodes=[]
    for i in range(taps):
        v=gs.Variable('g'+str(i),dtype=np.float32);u=gs.Variable('u'+str(i),dtype=np.float32)
        nodes.extend([gs.Node(op='Gather',name='g'+str(i),attrs={'axis':-1},inputs=[x,gs.Constant('idx'+str(i),positions+i*dilation)],outputs=[v]),
                      gs.Node(op='Unsqueeze',name='u'+str(i),inputs=[v,gs.Constant('axis'+str(i),np.array([3],np.int64))],outputs=[u])]);windows.append(u)
    out=gs.Variable('out',dtype=np.float32,shape=[2,3,len(positions),taps]);nodes.append(gs.Node(op='Concat',name='windows',attrs={'axis':-1},inputs=windows,outputs=[out]))
    graph=gs.Graph(nodes=nodes,inputs=[x],outputs=[out],opset=23)
    data=np.arange(2*3*length,dtype=np.float32).reshape(2,3,length)
    before=ReferenceEvaluator(gs.export_onnx(graph)).run(None,{'x':data})[0]
    assert len(rewrite(graph))==1
    after=ReferenceEvaluator(gs.export_onnx(graph)).run(None,{'x':data})[0]
    np.testing.assert_array_equal(before,after)
