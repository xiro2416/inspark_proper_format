"""Build a native or inherited A_1007 engine with maximum supported TRT search."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
import traceback
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, choices=(1,2,4,8,16,32,64), required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--timing-cache', type=Path)
    parser.add_argument('--inherited', action='store_true', help='Experimental B32 source mechanisms; target validation required')
    parser.add_argument('--plugin-code', type=Path, help='Isolated prepared plugin source directory')
    parser.add_argument('--plugin-package', help='Runtime plugin package bound to this engine')
    parser.add_argument('--omit-inherited', nargs='+', default=[],
                        choices=('dw', 'attention', 'nonlinear_attention', 'f32_ffn', 'int8_value', 'int8_residual'),
                        help='Explicit target ablation; retain original ONNX for these mechanisms')
    parser.add_argument('--profile-json', type=Path,
                        help='One explicit min/opt/max shape profile, keyed by dynamic input name')
    parser.add_argument('--max-num-tactics', type=int, default=2**31-2)
    parser.add_argument('--tiling', choices=('FULL', 'FAST'), default='FULL')
    parser.add_argument('--build-route', default='')
    parser.add_argument('--no-torch', action='store_true', help='Compiler diagnosis: avoid loading PyTorch CUDA libraries before TensorRT')
    args = parser.parse_args()
    if args.omit_inherited and not args.inherited:
        parser.error('--omit-inherited requires --inherited')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        parser.error('CUDA_VISIBLE_DEVICES must select exactly --gpu')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    code=(args.plugin_code or root/f'.work/zipvoice/b{args.batch}/code').resolve()
    if not code.is_relative_to(root) or not code.is_dir():
        raise ValueError('Prepared plugin code must be a project directory')
    import re
    package=f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{args.batch}' if args.plugin_package is None else args.plugin_package
    if not re.fullmatch(rf'inspark_infer\.ops\.tensorrt\.zipvoice\.a1007\.b{args.batch}(?:_[a-z0-9_]+)?',package):
        raise ValueError('Invalid target-batch plugin package')
    sys.path.insert(0, str(code))
    lock = (root / '.gpu-inference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.no_torch:
        fields = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=name,compute_cap', '--format=csv,noheader'], text=True).strip().split(',')
        gpu_name = fields[0].strip()
        capability = [int(v) for v in fields[1].strip().split('.')]
    else:
        import torch
        assert torch.cuda.device_count() == 1
        gpu_name = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
    import tensorrt as trt
    if args.plugin_code is not None:
        import triton
        original_compile=triton.compile
        recorded=set()
        def compile_with_resource_record(*compile_args,**compile_kwargs):
            compiled=original_compile(*compile_args,**compile_kwargs)
            ptx=compiled.asm['ptx'];key=hashlib.sha256(ptx.encode()).hexdigest()
            if key not in recorded:
                compiled._init_handles()
                record={'function':compiled.src.fn.__name__,'ptx_sha256':key,
                        'num_warps':compiled.metadata.num_warps,'num_stages':compiled.metadata.num_stages,
                        'shared_bytes':compiled.metadata.shared,'registers':compiled.n_regs,
                        'spills':compiled.n_spills,'global_scratch_bytes':compiled.metadata.global_scratch_size,
                        'rna_conversion':'cvt.rna.tf32.f32' in ptx,
                        'int8_mma':'.s32.s8.s8.s32' in ptx,'tf32_mma':'.tf32.' in ptx and 'mma.sync' in ptx,
                        'scope':'Triton AOT cubin resources/PTX; TensorRT consumes the recorded PTX, not this diagnostic cubin.'}
                with (args.output/'aot-resources.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
                recorded.add(key)
            return compiled
        triton.compile=compile_with_resource_record
    start = time.monotonic()
    report = {'status': 'started', 'source': str(args.onnx.resolve()),
              'source_sha256': digest(args.onnx), 'tensorrt': trt.__version__,
              'gpu': gpu_name, 'physical_gpu': args.gpu,
              'compute_capability': capability, 'torch_preloaded': not args.no_torch,
              'runtime_plugin_package': package,
              'representation': 'unchanged ONNX; no graph/plugin rewrite',
              'performance_acceptance': False}
    import onnx
    source_metadata = onnx.load(args.onnx, load_external_data=False)
    locations = {entry.value for tensor in source_metadata.graph.initializer
                 for entry in tensor.external_data if entry.key == 'location'}
    report['external_data_files'] = []
    for location in sorted(locations):
        data_path = (args.onnx.resolve().parent / location).resolve()
        if not data_path.is_relative_to(root):
            raise RuntimeError('External weights must stay inside the project')
        report['external_data_files'].append({'path': str(data_path), 'sha256': digest(data_path)})
    del source_metadata
    try:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        onnx_parser = trt.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
            raise RuntimeError(' | '.join(str(onnx_parser.get_error(i))
                                         for i in range(onnx_parser.num_errors)))
        if args.inherited:
            import dw_int8_full_rewrite, f32_tf32_rna_rewrite
            if args.batch == 64:
                import online_nlwide_rewrite
            else:
                import normal_tf32_rewrite as online_nlwide_rewrite
            import int8_nonlinear_value_rewrite, i8_residual_rewrite
            report['target_validation'] = 'pending; inherited mechanism coverage is not target correctness'
            report['batch'] = args.batch
            report['omitted_inherited_mechanisms'] = sorted(set(args.omit_inherited))
            report['original_weights_scales_and_quantization_preserved'] = True
            report['attention_math'] = ('B64 original normal IEEE QK/AV' if args.batch == 64 else 'normal nearest TF32 QK/AV') + '; nonlinear IEEE QK/RNA AV; original FP32 storage and normalization'
            if args.batch == 64 and package.endswith(('_normtf32','_normtf32k16','_normtf32k16wp')):
                kernel_source=(code/'online_branch_stats_runtime_kernel.py').read_text()
                wrapper_source=(code/'online_branch_wide_aot.py').read_text()
                assert "input_precision='tf32'" in kernel_source and 'cvt.rna.tf32.f32' in kernel_source
                assert wrapper_source.count("'tf32'")==3 and "'ieee'" not in wrapper_source
                report['attention_math']='Normal ordinary TF32 QK/AV with explicit RNA; nonlinear IEEE QK/RNA AV; Float32 IO, accumulation, stats and unrounded probability denominator'
                report['attention_math_evidence']='Prepared source contract and plugin PTX assertions; inspect aot-resources.jsonl and deployed tactics before acceptance'
            for mechanism, key, module, count in [
                ('dw','dw_replacements',dw_int8_full_rewrite,24),
                ('attention','online_replacements',online_nlwide_rewrite,16),
                ('f32_ffn','f32_replacements',f32_tf32_rna_rewrite,12),
                ('int8_value','int8_nonlinear_value_replacements',int8_nonlinear_value_rewrite,12),
                ('int8_residual','int8_residual_replacements',i8_residual_rewrite,24)]:
                if mechanism in args.omit_inherited:report[key]=[]
                elif mechanism=='attention' and 'nonlinear_attention' in args.omit_inherited:
                    report[key]=module.rewrite(network,args.onnx,trt,skip_nonlinear=True)
                else:report[key]=module.rewrite(network,args.onnx,trt)
                assert len(report[key]) == (0 if mechanism in args.omit_inherited else count), (key,len(report[key]))
            expected_attention=0 if 'attention' in args.omit_inherited else 32 if 'nonlinear_attention' in args.omit_inherited else 48
            assert sum(len(x['outputs']) for x in report['online_replacements'])==expected_attention
            report['plugin_sources']={str(p.relative_to(root)):digest(p) for p in code.glob('*.py')}
            report['representation']='Original INT8 weights/scales; inherited coverage recorded per mechanism, omissions explicit; target correctness pending'
            if 'attention' in args.omit_inherited:
                report['attention_math']='Original ONNX attention handled by TensorRT; inspect actual tactics before acceptance'
            elif 'nonlinear_attention' in args.omit_inherited:
                report['attention_math']='Original ONNX nonlinear attention handled by TensorRT; inherited normal attention/math/stats cache retained. Inspect actual native tactics and quality before acceptance.'
        report['network_layers'] = network.num_layers
        report['network_io'] = [{'name': v.name, 'shape': list(v.shape), 'dtype': str(v.dtype)}
             for v in [*(network.get_input(i) for i in range(network.num_inputs)),
                       *(network.get_output(i) for i in range(network.num_outputs))]]
        config = builder.create_builder_config()
        dynamic_inputs = {network.get_input(i).name: network.get_input(i)
                          for i in range(network.num_inputs)
                          if -1 in network.get_input(i).shape}
        if dynamic_inputs:
            if args.profile_json is None:
                raise RuntimeError('Dynamic ONNX requires an explicit --profile-json')
            shapes = json.loads(args.profile_json.read_text())
            if set(shapes) != set(dynamic_inputs):
                raise RuntimeError('Profile names must exactly match all dynamic inputs')
            profile = builder.create_optimization_profile()
            for name, tensor in dynamic_inputs.items():
                bounds = shapes[name]
                if set(bounds) != {'min', 'opt', 'max'}:
                    raise RuntimeError(f'Expected min/opt/max for {name}')
                for bound in bounds.values():
                    if (len(bound) != len(tensor.shape) or
                            any(type(d) is not int or d <= 0 for d in bound)):
                        raise RuntimeError(f'Invalid dimensions for {name}')
                profile.set_shape(name, bounds['min'], bounds['opt'], bounds['max'])
            if not profile or config.add_optimization_profile(profile) != 0:
                raise RuntimeError('TensorRT rejected the optimization profile')
            report['shape_profile'] = shapes
            report['shape_profile_sha256'] = digest(args.profile_json)
        elif args.profile_json is not None:
            raise RuntimeError('A static graph must not silently ignore a profile')
        config.builder_optimization_level = 5
        config.tiling_optimization_level = getattr(trt.TilingOptimizationLevel, args.tiling)
        # Installed 11.3 API uses an exclusive INT_MAX upper bound.
        config.max_num_tactics = args.max_num_tactics
        assert config.max_num_tactics == args.max_num_tactics
        if args.build_route:
            config.build_route = args.build_route
            assert config.build_route == args.build_route
        config.max_aux_streams = 0
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        (args.output / 'build-routes.json').write_text(config.all_build_routes or '{}')
        report['builder'] = {'optimization_level': config.builder_optimization_level,
            'tiling': str(config.tiling_optimization_level),
            'max_num_tactics': config.max_num_tactics, 'aux_streams': config.max_aux_streams,
            'effective_build_route': config.build_route,
            'build_routes': 'build-routes.json',
            'workspace_limit': config.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE),
            'tactic_shared_memory_limit': config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY),
            'timing_cache_source': str(args.timing_cache) if args.timing_cache else 'empty SM89 cache'}
        cache = config.create_timing_cache(args.timing_cache.read_bytes() if args.timing_cache else b'')
        if not config.set_timing_cache(cache, ignore_mismatch=False):
            raise RuntimeError('Incompatible timing cache')
        (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'event': 'build_begin', 'report': report}), flush=True)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError('TensorRT build failed; see the build log')
        path = args.output / 'engine.plan'
        path.write_bytes(bytes(plan))
        (args.output / 'timing.cache').write_bytes(config.get_timing_cache().serialize())
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError('New SM89 plan failed to deserialize')
        (args.output / 'inspector.json').write_text(
            engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        report.update(status='built_unvalidated', engine_sha256=digest(path),
                      engine_layers=engine.num_layers, device_memory_size=engine.device_memory_size_v2,
                      effective_aux_streams=engine.num_aux_streams)
        if dynamic_inputs:
            report['effective_shape_profile'] = {
                name: [list(shape) for shape in engine.get_tensor_profile_shape(name, 0)]
                for name in dynamic_inputs}
            for name, bounds in report['shape_profile'].items():
                if report['effective_shape_profile'][name] != [bounds[k] for k in ('min', 'opt', 'max')]:
                    raise RuntimeError(f'Built profile differs for {name}')
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
    report['seconds'] = time.monotonic() - start
    (args.output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'built_unvalidated':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
