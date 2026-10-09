#!/usr/bin/env python3
"""Build and inspect a static standard-ONNX candidate without project plugins."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', type=int, default=7)
    ap.add_argument('--onnx', type=Path, required=True)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--optimization-level', type=int, default=5, choices=range(6))
    ap.add_argument('--tiling', choices=('none', 'fast', 'moderate', 'full'), default='moderate')
    ap.add_argument('--aux-streams', type=int, default=0, choices=(0, 1, 2))
    ap.add_argument('--l2-limit', type=int, default=-1)
    ap.add_argument('--merge-timing-cache',type=Path,
                    help='Merge a compatible measured cache, retaining source hash and compatibility outcome')
    ap.add_argument('--no-timing-cache-seed', action='store_true',
                    help='Start a new parameter-specific timing cache without seeding baseline measurements')
    ap.add_argument('--official-bigvgan-plugin',action='store_true',
                    help='Explicitly permit the unchanged NVIDIA activation adapter inventory')
    ap.add_argument('--cfm-gated-up-plugin',action='store_true',help='Permit the explicit custom FP8 paired GEMM plugin')
    ap.add_argument('--cfm-gated-up-layout',choices=('paired','interleaved'),default='paired')
    ap.add_argument('--small-fir-plugin',action='store_true',help='Permit the explicit custom short FIR activation')
    ap.add_argument('--small-fir-layout',choices=('whole','tiled_mix','all_tiled'),default='whole')
    ap.add_argument('--implicit-int8-conv',action='store_true',help='Explicit B32 source-recipe custom INT8 convolution inventory')
    ap.add_argument('--tuned-implicit-int8-conv',action='store_true',help='Measured B32/C192/K11 custom convolution schedule')
    ap.add_argument('--migrated-implicit-int8-conv',action='store_true',help='B32 schedule migrated to authorized target batches')
    ap.add_argument('--target-implicit-int8-schedule',action='store_true',help='Target measured source-recipe INT8 schedules')
    ap.add_argument('--fir-int8-quant-plugin',action='store_true',help='Fuse unchanged FP32 FIR math with original terminal signed INT8 quantization')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    from inspark_infer.build.trt113_policy import prepare, save, timing_cache_metadata
    from trt113_provenance import load_onnx_export, source_identity
    export, provenance = load_onnx_export(args.onnx.resolve())
    plugins=['nvidia_bigvgan::alias_free'] if args.official_bigvgan_plugin else []
    if args.cfm_gated_up_plugin:
        if plugins:raise ValueError('Select one explicit plugin inventory')
        plugins=['inspark_custom::fp8_quantize','inspark_custom::fp8_gated_up'+('_interleaved' if args.cfm_gated_up_layout=='interleaved' else '')]
    if args.small_fir_plugin:
        if plugins:raise ValueError('Select one explicit plugin inventory')
        plugins=['inspark_custom::small_fir_activation']
        if args.small_fir_layout=='tiled_mix':plugins.append('inspark_custom::small_fir_activation_tiled')
        if args.small_fir_layout=='all_tiled':plugins=['inspark_custom::small_fir_activation_tiled']
    if args.implicit_int8_conv:
        if not args.small_fir_plugin or export['component']!='vocoder' or export['quantization_recipe']['scheme']!='int8_smoothquant' or export['batch'] not in (1,2,4,8,16,32,64,128):
            raise ValueError('Implicit INT8 convolution requires the explicit B32 INT8 Vocoder candidate')
        plugins.append('inspark_custom::implicit_int8_conv_1d')
    if args.tuned_implicit_int8_conv:
        if not args.implicit_int8_conv:raise ValueError('Tuned schedule requires explicit implicit INT8 candidate')
        plugins.append('inspark_custom::implicit_int8_conv_1d_tuned')
    if args.migrated_implicit_int8_conv:
        if not args.implicit_int8_conv or args.tuned_implicit_int8_conv or export['batch'] not in (1,2,4,8,16,64,128):raise ValueError('Invalid migrated INT8 schedule inventory')
        plugins.append('inspark_custom::implicit_int8_conv_1d_migrated')
    if args.target_implicit_int8_schedule:
        if not args.implicit_int8_conv or args.tuned_implicit_int8_conv or args.migrated_implicit_int8_conv or export['batch'] not in (1,2,4,8,16,64,128):raise ValueError('Invalid target schedule inventory')
        plugins.append('inspark_custom::implicit_int8_conv_1d_schedule')
    if args.fir_int8_quant_plugin:
        if not args.implicit_int8_conv or not args.small_fir_plugin or export['batch'] not in (1,2,4,8,16,64,128):raise ValueError('FIR quant fusion requires explicit source-recipe INT8 Vocoder')
        plugins.append('inspark_custom::small_fir_activation_quantized')
    if not export or export.get('plugins') != plugins or not export.get('quantization_recipe'):
        raise ValueError('Expected calibrated export with the explicitly selected plugin inventory')
    with GPULease(args.gpu):
        trt = _import_trt113()
        if args.official_bigvgan_plugin:
            from inspark_infer.ops.tensorrt.official_bigvgan_plugin import register
            register()
        if args.cfm_gated_up_plugin:
            from inspark_infer.ops.tensorrt.cfm_gated_up_plugin import register
            register()
        if args.small_fir_plugin:
            from inspark_infer.ops.tensorrt.vocoder_small_fir_plugin import register
            register()
        if args.implicit_int8_conv:
            from deployment.b32.implicit_int8_plugin import register
            register()
            if args.tuned_implicit_int8_conv:register(tuned=True)
            if args.migrated_implicit_int8_conv:register(migrated=True)
        if args.target_implicit_int8_schedule:
            from deployment.multibatch.schedule_plugin import register
            register()
        if args.fir_int8_quant_plugin:
            from deployment.multibatch.fir_quant_plugin import register
            register()
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(args.onnx.resolve())):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        for i in range(network.num_inputs):
            if min(network.get_input(i).shape) <= 0:
                raise ValueError('Only static shapes are allowed for this experiment')
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = args.optimization_level
        config.max_num_tactics = 2147483646
        if config.max_num_tactics != 2147483646:
            raise RuntimeError('TensorRT did not accept the supported maximum tactic search count')
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        cache, _ = prepare(config, trt, component='unified_' + export['component'],
                           tiling=args.tiling, workspace_bytes=0,
                           max_aux_streams=args.aux_streams, l2_limit_for_tiling=args.l2_limit,
                           seed_timing_cache=not args.no_timing_cache_seed)
        cache_preparation=timing_cache_metadata(cache)
        if args.merge_timing_cache:
            payload=args.merge_timing_cache.read_bytes()
            current=config.get_timing_cache();other=config.create_timing_cache(payload)
            if not current.combine(other,False):raise RuntimeError('Additional timing cache is incompatible')
            if not config.set_timing_cache(current,False):raise RuntimeError('Cannot bind merged timing cache')
            cache_preparation['merged_source']=dict(path=str(args.merge_timing_cache.resolve()),
                sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),compatible=True)
        cache_before_sha256 = hashlib.sha256(Path(cache).read_bytes()).hexdigest() if Path(cache).is_file() else None
        if args.no_timing_cache_seed and cache_preparation.get('existing_bytes', 0):
            raise ValueError('Fresh tactic search requested but parameter-specific timing cache already exists')
        print(json.dumps(dict(event='timing_cache_prepare',**cache_preparation)),flush=True)
        started = time.perf_counter()
        print(json.dumps(dict(event='build_start', component=export['component'],
                              batch=export['batch'], scheme=export['quantization_recipe']['scheme'])), flush=True)
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError('TensorRT returned no engine')
        save(config, cache)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError('Deserialization failed')
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError('Context creation failed')
        args.engine.parent.mkdir(parents=True, exist_ok=True)
        args.engine.write_bytes(bytes(blob))
        inspector = engine.create_engine_inspector()
        inspector.execution_context = context
        inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        args.engine.with_suffix('.inspector.json').write_text(inspection)
        tensors = [dict(name=engine.get_tensor_name(i),
                        shape=list(engine.get_tensor_shape(engine.get_tensor_name(i))),
                        dtype=str(engine.get_tensor_dtype(engine.get_tensor_name(i))),
                        mode=str(engine.get_tensor_mode(engine.get_tensor_name(i))))
                   for i in range(engine.num_io_tensors)]
        report = dict(format=1, backend=(f'TensorRT {trt.__version__} custom AOT plugins' if args.cfm_gated_up_plugin or args.small_fir_plugin
                                       else f'TensorRT {trt.__version__} with unchanged NVIDIA activation' if plugins
                                       else f'TensorRT {trt.__version__} standard ONNX'),
                      component=export['component'], kind=export['kind'],
                      batch=export['batch'], frames=export['frames'],
                      engine=args.engine.name, sha256=hashlib.sha256(bytes(blob)).hexdigest(),
                      bytes=len(bytes(blob)), trt=trt.__version__, torch=torch.__version__,
                      cuda=torch.version.cuda, gpu_name=torch.cuda.get_device_name(),
                      sm=int(''.join(map(str, torch.cuda.get_device_capability()))),
                      build_seconds=time.perf_counter()-started, optimization_level=args.optimization_level,
                      tiling_optimization_level=args.tiling, max_aux_streams=args.aux_streams,
                      l2_limit_for_tiling=args.l2_limit, workspace_bytes=0, timing_cache=str(cache),
                      timing_cache_preparation=cache_preparation,
                      timing_cache_seed_disabled=args.no_timing_cache_seed,
                      timing_cache_sha256_before=cache_before_sha256,
                      timing_cache_sha256_after=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
                      effective_l2_limit_for_tiling=int(config.l2_limit_for_tiling),
                      avg_timing_iterations=int(config.avg_timing_iterations),
                      tactic_sources=int(config.get_tactic_sources()),
                      available_tactic_sources=[name for name in dir(trt.TacticSource) if name.isupper()],
                      max_num_tactics=int(config.max_num_tactics),
                      actual_aux_streams=int(engine.num_aux_streams),
                      workspace_limit_bytes=int(config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)),
                      tactic_shared_memory_limit_bytes=int(config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY)),
                      strongly_typed=True, tf32=False, plugins=plugins, tensors=tensors,
                      onnx=str(args.onnx.resolve()), onnx_sha256=export['onnx_sha256'],
                      provenance=provenance, provenance_status=provenance['status'],
                      build_source=source_identity(), quantization_recipe=export['quantization_recipe'],
                      precision=export['export_settings']['precision'],
                      validation=dict(deserialized=True, context_created=True, executed=False,
                                      numerical_audit='pending_final_audit'))
        if args.official_bigvgan_plugin:
            vendor=json.loads(Path('artifacts/sm120_0924/unified/gpu_inference_20261001/official_bigvgan_kernel.json').read_text())
            vendor['module_sha256']=hashlib.sha256(Path(vendor['module']).read_bytes()).hexdigest()
            report['vendor_plugin_provenance']=vendor
            report['official_activation_rewrite']=export['official_activation_rewrite']
        if args.cfm_gated_up_plugin:
            report['custom_gated_up']=export['custom_gated_up']
            if 'interleaved' in export: report['interleaved']=export['interleaved']
        if args.small_fir_plugin:
            report['custom_small_fir']=export['custom_small_fir']
        if 'prompt_frames' in export:
            report['prompt_frames'] = export['prompt_frames']
        if 'kv_limit' in export:
            report['kv_limit'] = export['kv_limit']
        args.engine.with_suffix('.plan.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: report[key] for key in ('component', 'batch', 'precision', 'build_seconds', 'bytes')}), flush=True)


if __name__ == '__main__':
    main()
