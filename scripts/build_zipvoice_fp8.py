"""Serial SM120 TensorRT11.3 builds with exact batches and dynamic T/L profiles."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES, profiles, sha, write


def prepare_graph(batch, kind):
    import onnx
    source = ROOT/'models/zipvoice'/('fp8/fm.onnx' if kind=='fm' else f'onnx/{kind}.onnx')
    graph = onnx.load(source)
    for value in [*graph.graph.input,*graph.graph.output]:
        dims=value.type.tensor_type.shape.dim
        if not dims or value.name=='frame_positions':continue
        dims[0].dim_value=batch
    target=ROOT/f'.work/fp8/b{batch}/{kind}.onnx'
    target.parent.mkdir(parents=True,exist_ok=True)
    external=target.with_suffix('.data')
    if external.exists():external.unlink()
    onnx.save_model(graph,str(target),save_as_external_data=True,
                   all_tensors_to_one_file=True,location=external.name,size_threshold=1024)
    onnx.checker.check_model(str(target))
    return target


def build(batch,kind,variant='native'):
    import tensorrt as trt
    output=ROOT/f'artifacts/zipvoice/sm120/fp8/b{batch}/{kind if variant=="native" else kind+"-"+variant}'
    graph=prepare_graph(batch,kind)
    recipe=ROOT/'models/zipvoice/fp8/quantization.json'
    report_path=output/'build.json'
    expected=dict(source_sha256=sha(graph),recipe_sha256=sha(recipe) if kind=='fm' else None,batch=batch,
                  component=kind,tensorrt=trt.__version__)
    if report_path.is_file():
        previous=json.loads(report_path.read_text())
        if all(previous.get(k)==v for k,v in expected.items()) and previous.get('status')=='built_unvalidated':
            if sha(output/'engine.plan')==previous['engine_sha256']:
                print(json.dumps({'event':'reuse_verified_build','batch':batch,'component':kind}),flush=True)
                return
        raise RuntimeError('Existing build differs; preserve it before rebuilding: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    report=dict(expected,status='building',physical_gpu=3,hardware_sm=120,
                arithmetic='FM eligible last12 W8A8 E4M3; remaining operators FLOAT32' if kind=='fm' else 'original FLOAT32 component',
                correctness_acceptance=False,performance_acceptance=False)
    write(report_path,report)
    logger=trt.Logger(trt.Logger.INFO)
    start=time.monotonic()
    try:
        builder=trt.Builder(logger)
        network=builder.create_network(0)
        parser=trt.OnnxParser(network,logger)
        if not parser.parse_from_file(str(graph)):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        if variant in ['attention','attentiongeo']:
            import onnx
            from inspark_infer.runtime.zipvoice_fp8.attention import rewrite
            report['attention_rewrite']=rewrite(network,onnx.load(graph,load_external_data=False),batch,trt,variant=='attentiongeo')
            package=ROOT/'src/inspark_infer/ops/tensorrt/zipvoice_fp8'/f'b{batch}'/('geo' if variant=='attentiongeo' else '')
            report['plugin_sources']={str(p.relative_to(ROOT)):sha(p) for p in sorted(package.glob('*.py'))}
        config=builder.create_builder_config()
        config.builder_optimization_level=5
        config.tiling_optimization_level=trt.TilingOptimizationLevel.FULL
        config.max_num_tactics=2**31-2
        config.max_aux_streams=0
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        p=builder.create_optimization_profile()
        actual={}
        desired=profiles(batch)[kind]
        for i in range(network.num_inputs):
            v=network.get_input(i)
            if -1 not in v.shape:continue
            bounds=desired[v.name]
            p.set_shape(v.name,*[bounds[k] for k in ('min','opt','max')])
            actual[v.name]=bounds
        if actual and (not p or config.add_optimization_profile(p)!=0):
            raise RuntimeError('Profile registration failed')
        cache_path=ROOT/f'.cache/fp8/timing/{kind}.cache'
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
        if not config.set_timing_cache(cache,ignore_mismatch=False):
            raise RuntimeError('SM120 cache incompatibility')
        report.update(shape_profile=desired,network_inputs=[
            dict(name=network.get_input(i).name,shape=list(network.get_input(i).shape),
                 dtype=str(network.get_input(i).dtype)) for i in range(network.num_inputs)],
            builder=dict(level=5,tiling='FULL',max_num_tactics=config.max_num_tactics,
                         max_aux_streams=0,workspace=config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
            network_layers=network.num_layers)
        write(report_path,report)
        (output/'build-routes.json').write_text(config.all_build_routes or '{}')
        print(json.dumps({'event':'build_begin','batch':batch,'component':kind}),flush=True)
        plan=builder.build_serialized_network(network,config)
        if plan is None:raise RuntimeError('TensorRT failed to build')
        path=output/'engine.plan';path.write_bytes(bytes(plan))
        cached=bytes(config.get_timing_cache().serialize());cache_path.write_bytes(cached)
        (output/'timing.cache').write_bytes(cached)
        runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(plan)
        if engine is None:raise RuntimeError('SM120 plan cannot deserialize')
        inspector=engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        (output/'inspector.json').write_text(inspector)
        report.update(status='built_unvalidated',engine_sha256=sha(path),engine_layers=engine.num_layers,
                      device_memory_size=engine.device_memory_size_v2,aux_streams=engine.num_aux_streams,
                      inspector_sha256=sha(output/'inspector.json'))
        for name,bounds in actual.items():
            if [list(s) for s in engine.get_tensor_profile_shape(name,0)]!=[bounds[k] for k in ('min','opt','max')]:
                raise RuntimeError('Effective engine profile differs: '+name)
        if kind=='fm':
            parsed=json.loads(inspector)
            fp8_layers=[x for x in parsed.get('Layers',[]) if 'fp8' in json.dumps(x).lower()
                        or 'e4m3' in json.dumps(x).lower()]
            if not fp8_layers:raise RuntimeError('No actual FP8 layer/tactic evidence in engine inspector')
            report['fp8_inspector_layers']=len(fp8_layers)
    except Exception as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.monotonic()-start
        write(report_path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=[16,1,2,4,8,32,64])
    p.add_argument('--components',nargs='+',choices=['fm','text','vocos'],default=['fm','text','vocos'])
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--variant',choices=['native','attention','attentiongeo'],default='native')
    args=p.parse_args()
    if args.gpu!=3 or os.getenv('CUDA_VISIBLE_DEVICES')!='3':
        raise RuntimeError('This build task is authorized only on physical GPU3')
    from inspark_infer.runtime.device import GPULease
    with GPULease(args.gpu):
        for batch in args.batches:
            for kind in args.components:build(batch,kind,args.variant)


if __name__=='__main__':main()
