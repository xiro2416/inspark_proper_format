"""Matched strong-build synthetic shape probe; no full application claims."""
import json,statistics
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    from inspark_infer.build.trt113_policy import prepare,save
    rows=[];outputs=[]
    with GPULease(1):
        select_gpu(1);trt=_import_trt113();torch.manual_seed(1032)
        values=torch.randn(32,24,13312,device='cuda');stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for label in ('floating','integer'):
            directory=ROOT/'.cache/b32-columns-probe'/label
            logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
            network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser=trt.OnnxParser(network,logger)
            if not parser.parse_from_file(str(directory/'model.onnx')):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32)
            config.builder_optimization_level=5;config.max_num_tactics=2147483646
            config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
            cache,_=prepare(config,trt,component='b32_columns_probe',tiling='full',workspace_bytes=0,max_aux_streams=0,l2_limit_for_tiling=-1)
            blob=builder.build_serialized_network(network,config)
            if blob is None:raise RuntimeError('Probe build failed: '+label)
            save(config,cache);(directory/'model.engine').write_bytes(bytes(blob))
            runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(blob);context=engine.create_execution_context()
            result=torch.empty(tuple(engine.get_tensor_shape('y')),device='cuda')
            context.set_tensor_address('x',values.data_ptr());context.set_tensor_address('y',result.data_ptr())
            with torch.cuda.stream(stream):
                for _ in range(3):assert context.execute_async_v3(stream.cuda_stream)
                stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):assert context.execute_async_v3(stream.cuda_stream)
                times=[]
                for _ in range(30):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record(stream);graph.replay();end.record(stream);end.synchronize();times.append(start.elapsed_time(end))
                assert torch.isfinite(result).all();outputs.append(result.clone())
            inspector=engine.create_engine_inspector();inspector.execution_context=context
            text=inspector.get_engine_information(trt.LayerInformationFormat.JSON);(directory/'model.inspector.json').write_text(text)
            layers=json.loads(text)['Layers'];integer=sum('i8i32' in l.get('TacticName','') for l in layers)
            if not integer:raise RuntimeError('Probe lost integer GEMM coverage')
            rows.append(dict(label=label,p50_ms=statistics.median(times),mean_ms=statistics.mean(times),samples=times,int8_tactic_layers=integer,layers=len(layers)))
            print(json.dumps(rows[-1]),flush=True)
            del graph,context,engine,runtime
        torch.cuda.synchronize()
        error=(outputs[1]-outputs[0]).float()
        report=dict(shape=[32,24,13312],kernel=11,out_channels=24,scope='synthetic isolated convolution Graph, same weights/scales/input; not E2E',
            rows=rows,output_difference=dict(max_abs=float(error.abs().max()),relative_l2=float(error.norm()/outputs[0].norm())),
            build=dict(optimization_level=5,tiling='full',max_num_tactics=2147483646,aux_streams=0),
            gain_pct=100*(rows[0]['p50_ms']-rows[1]['p50_ms'])/rows[0]['p50_ms'])
        (ROOT/'deployment/b32/history/integer-columns-probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
