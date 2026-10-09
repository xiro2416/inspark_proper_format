"""Compare mathematically equivalent Q/DQ Conv1d and Conv2d lowerings on GPU1."""
import hashlib
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--strongly-typed',action='store_true');p.add_argument('--activation-axis',type=int,choices=[0,1],default=1);a=p.parse_args()
    torch.set_num_threads(8)
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
    tag=('strong' if a.strongly_typed else 'default')+f'-axis{a.activation_axis}'
    directory=ROOT/'.work/vocoder-conv-probe'/tag;directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    weight=rng.normal(0,.02,(192,384,4)).astype('float32')
    bias=rng.normal(0,.02,(192,)).astype('float32')
    scale=np.max(np.abs(weight),axis=(1,2))/127
    x=rng.normal(0,.1,(1,384,1670)).astype('float32')
    reference=F.conv1d(torch.from_numpy(np.clip(np.round(x/.003),-128,127)*.003),
                      torch.from_numpy(np.clip(np.round(weight/scale[:,None,None]),-128,127)*scale[:,None,None]),
                      torch.from_numpy(bias)).numpy()
    report=[]
    with GPULease(1):
        select_gpu(1)
        trt=_import_trt113()
        for lifted in (False,True):
            label='conv2d' if lifted else 'conv1d'
            nodes=[]
            values=dict(weight=weight[:,:,None,:] if lifted else weight,bias=bias,
                        xs=np.array(.003,dtype='float32'),xz=np.array(0,dtype='int8'),
                        ws=scale,wz=np.zeros(192,dtype='int8'),axes=np.array([2],dtype='int64'))
            input_name='x'
            if lifted:
                nodes.append(h.make_node('Unsqueeze',['x','axes'],['lifted_x']))
                input_name='lifted_x'
            nodes.extend([h.make_node('QuantizeLinear',[input_name,'xs','xz'],['xq'],axis=a.activation_axis),
                          h.make_node('DequantizeLinear',['xq','xs','xz'],['xd'],axis=a.activation_axis),
                          h.make_node('QuantizeLinear',['weight','ws','wz'],['wq'],axis=0),
                          h.make_node('DequantizeLinear',['wq','ws','wz'],['wd'],axis=0),
                          h.make_node('Conv',['xd','wd','bias'],['conv'],kernel_shape=[1,4] if lifted else [4],
                                      pads=[0,0,0,0] if lifted else [0,0],
                                      strides=[1,1] if lifted else [1],dilations=[1,1] if lifted else [1],group=1)])
            nodes.append(h.make_node('Squeeze',['conv','axes'],['output']) if lifted else
                         h.make_node('Identity',['conv'],['output']))
            graph=h.make_graph(nodes,label,[h.make_tensor_value_info('x',T.FLOAT,[1,384,1670])],
                               [h.make_tensor_value_info('output',T.FLOAT,[1,192,1667])],
                               [nh.from_array(v,k) for k,v in values.items()])
            model=h.make_model(graph,opset_imports=[h.make_opsetid('',20)],ir_version=10)
            onnx.checker.check_model(model);path=directory/(label+'.onnx');onnx.save(model,path)
            logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
            network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if a.strongly_typed else 0);parser=trt.OnnxParser(network,logger)
            if not parser.parse(path.read_bytes()):raise RuntimeError(str([parser.get_error(i) for i in range(parser.num_errors)]))
            config=builder.create_builder_config();config.builder_optimization_level=5
            config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
            config.tiling_optimization_level=trt.TilingOptimizationLevel.FULL
            config.max_num_tactics=2147483646;config.max_aux_streams=0
            config.clear_flag(trt.BuilderFlag.TF32)
            started=time.monotonic();blob=builder.build_serialized_network(network,config)
            if blob is None:raise RuntimeError('Probe build failed')
            engine_path=directory/(label+'.engine');engine_path.write_bytes(bytes(blob))
            plan=dict(engine=str(engine_path),sha256=hashlib.sha256(bytes(blob)).hexdigest(),batch=1,
                      gpu_name=torch.cuda.get_device_name(),sm=89,trt=trt.__version__)
            plan_path=directory/(label+'.plan.json');plan_path.write_text(json.dumps(plan))
            engine=StaticEngine(plan_path,1);actual=engine({'x':torch.from_numpy(x).cuda()})['output'].cpu().numpy()
            inspector=json.loads(engine.engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
            (directory/(label+'.inspector.json')).write_text(json.dumps(inspector,indent=2))
            tactics=[l.get('TacticName','') for l in inspector['Layers']]
            row=dict(kind=label,strongly_typed=a.strongly_typed,activation_axis=a.activation_axis,build_seconds=time.monotonic()-started,max_abs_error=float(np.max(np.abs(actual-reference))),
                     int8_tactics=[t for t in tactics if '_i8' in t],tactics=tactics)
            report.append(row);print(json.dumps(row),flush=True)
            del engine,blob,config,parser,network,builder
        (ROOT/'deployment/history'/f'vocoder-conv-lowering-probe-{tag}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
