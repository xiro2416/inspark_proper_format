import sys
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from canonicalize_nvfp4_attention import rewrite

def graph(fill):
    q,k,v=[gs.Variable(x,dtype=onnx.TensorProto.BFLOAT16) for x in ['q','k','v']]
    mask=gs.Variable('mask',dtype=np.bool_);nodes=[]
    def op(kind,name,inputs,attrs=None):
        y=gs.Variable(name);nodes.append(gs.Node(op=kind,name=name,inputs=inputs,outputs=[y],attrs=attrs or {}));return y
    qf=op('Cast','qf',[q],{'to':1});kf=op('Cast','kf',[k],{'to':1});vf=op('Cast','vf',[v],{'to':1})
    kt=op('Transpose','kt',[kf],{'perm':[0,1,3,2]});scores=op('MatMul','scores',[qf,kt]);inverse=op('Not','inverse',[mask])
    masked=op('Where','masked',[inverse,gs.Constant('fill',np.array(fill,np.float32)),scores]);p=op('Softmax','p',[masked],{'axis':-1});y=op('MatMul','y',[p,vf])
    return gs.Graph(nodes=nodes,inputs=[q,k,v,mask],outputs=[y],opset=23)

def test_prescaled_query_and_original_verification_mask_are_preserved():
    g=graph(float('-inf'));original=list(g.inputs);assert len(rewrite(g))==1
    attention=next(n for n in g.nodes if n.op=='Attention')
    assert list(attention.inputs)==original
    assert attention.attrs['scale']==1 and attention.attrs['is_causal']==0
    assert g.outputs[0].inputs[0].op=='Cast'
    with pytest.raises(ValueError,match='mask fill'):rewrite(graph(-100))
