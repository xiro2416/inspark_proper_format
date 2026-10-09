"""Real checkpoint FIR/Snake parameters, matched standalone TRT and Triton Graphs."""
import json,statistics
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]


def graph_measure(fn,stream):
    with torch.cuda.stream(stream):
        for _ in range(3):result=fn()
        stream.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result=fn()
        times=[]
        for _ in range(30):
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record(stream);graph.replay();e.record(stream);e.synchronize();times.append(s.elapsed_time(e))
        copy=result.clone();stream.synchronize()
    return dict(p50_ms=statistics.median(times),mean_ms=statistics.mean(times),samples=times),copy


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    from inspark_infer.build.unified_acoustic_export import StaticAliasFree
    from inspark_infer.models.indextts2.upstream.s2mel.modules.bigvgan.activations import SnakeBeta
    from inspark_infer.models.indextts2.upstream.s2mel.modules.bigvgan.alias_free_activation.torch.act import Activation1d
    from inspark_infer.ops.triton.vocoder_small_fir import SmallFIR
    from inspark_infer.ops.triton.vocoder_tiled_fir import TiledFIR
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    from inspark_infer.build.trt113_policy import prepare,save
    weights=torch.load(ROOT/'local_assets/runtime/models/index_tts2/hf_cache/bigvgan/bigvgan_generator.pt',map_location='cpu',weights_only=True)['generator']
    rows=[]
    with GPULease(1):
        select_gpu(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        trt=_import_trt113();logger=trt.Logger(trt.Logger.WARNING);torch.manual_seed(8032)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for stage,(channels,frames) in enumerate([(768,208),(384,832),(192,1664),(96,3328),(48,6656),(24,13312)]):
            path=ROOT/f'.cache/b32-fir-probe/stage{stage}';path.mkdir(parents=True,exist_ok=True)
            original=Activation1d(SnakeBeta(channels,alpha_logscale=True)).cuda().eval()
            key=f'resblocks.{stage*3}.activations.0.act'
            original.act.alpha.data.copy_(weights[key+'.alpha']);original.act.beta.data.copy_(weights[key+'.beta'])
            reference=StaticAliasFree(original,fir_polyphase=True).requires_grad_(False)
            x=torch.randn(32,channels,frames,device='cuda')*.25
            torch.onnx.export(reference,(x,),str(path/'model.onnx'),opset_version=20,dynamo=True,external_data=True,input_names=['x'],output_names=['y'],optimize=True)
            builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED));parser=trt.OnnxParser(network,logger)
            if not parser.parse_from_file(str(path/'model.onnx')):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config=builder.create_builder_config();config.clear_flag(trt.BuilderFlag.TF32);config.builder_optimization_level=5;config.max_num_tactics=2147483646
            config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
            cache,_=prepare(config,trt,component='b32_fir_probe',tiling='full',workspace_bytes=0,max_aux_streams=0,l2_limit_for_tiling=-1)
            blob=builder.build_serialized_network(network,config)
            if blob is None:raise RuntimeError('FIR probe build failed')
            save(config,cache);(path/'model.engine').write_bytes(bytes(blob));runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(blob);context=engine.create_execution_context()
            output=torch.empty_like(x);context.set_tensor_address('x',x.data_ptr());context.set_tensor_address('y',output.data_ptr())
            def control():
                if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('enqueue failed')
                return output
            base,y=graph_measure(control,stream)
            params=(original.upsample.filter.flatten(),original.downsample.lowpass.filter.flatten(),original.act.alpha,original.act.beta)
            candidates={'tiled':TiledFIR(*params)}
            if frames<=2048:candidates['whole_row']=SmallFIR(*params)
            row=dict(stage=stage,shape=list(x.shape),parameter_source=key,baseline_trt=base,candidates={})
            with torch.cuda.stream(stream),torch.inference_mode():
                ref=reference(x);stream.synchronize()
                row['trt_reference_max_abs']=float((y-ref).abs().max())
                for name,op in candidates.items():
                    measured,z=graph_measure(lambda:op(x),stream);diff=z-y
                    if not torch.isfinite(z).all():raise RuntimeError('Nonfinite candidate')
                    row['candidates'][name]=dict(measured,max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/y.norm()),gain_pct=100*(base['p50_ms']-measured['p50_ms'])/base['p50_ms'])
            rows.append(row)
            (ROOT/'deployment/b32/history/fir-probe.json').write_text(json.dumps(dict(scope='standalone exact FIR/Snake/FIR, real parameters, random inputs, complete Graph timings; no application E2E claim',rows=rows),indent=2)+'\n')
            print(json.dumps(dict(stage=stage,baseline_ms=base['p50_ms'],candidates={n:dict(ms=v['p50_ms'],gain=v['gain_pct'],relative_l2=v['relative_l2']) for n,v in row['candidates'].items()})),flush=True)
            del context,engine,runtime


if __name__=='__main__':main()
