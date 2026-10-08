"""Minimal target FM execution before application scheduling migration."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--candidate',type=Path,help='Independent FM build directory to validate against native')
    parser.add_argument('--output',type=Path,help='Independent evidence path, required for a candidate')
    args=parser.parse_args();batch=args.batch
    if args.candidate and not args.output:parser.error('--candidate requires --output')
    directories={route:ROOT/f'artifacts/zipvoice/a1007/b{batch}/{route}' for route in ('fm-native','fm-inherited')}
    if args.candidate:directories['fm-inherited']=args.candidate.resolve()
    metadata={route:json.loads((path/'build.json').read_text()) for route,path in directories.items()}
    for route,path in directories.items():
        with (path/'engine.plan').open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
        assert actual==metadata[route]['engine_sha256'],('FM engine hash mismatch',route)
    from inspark_infer.build.zipvoice import plugin_package
    package=plugin_package(batch,metadata['fm-inherited'].get('runtime_plugin_package'))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
    lock=(ROOT/'.gpu-inference.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for module in ('dw_int8_plugin','normal_tf32_plugin' if batch!=64 else 'online_branch_wide_plugin','online_nonlinear_rna_plugin','f32_tf32_rna_plugin','int8_nonlinear_value_plugin','i8_residual_plugin'):
        importlib.import_module(f'{package}.{module}')
    import torch
    import tensorrt as trt
    from inspark_infer.ops.tensorrt.zipvoice.engine import Engine
    torch.set_num_threads(2)
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9)
    result={'status':'running','batch':batch,'physical_gpu':1,'tests':[],
            'scope':'minimal FM only, before application scheduling; numerical differences report-only, full audio quality pending'}
    result['engine_sha256']={route:info['engine_sha256'] for route,info in metadata.items()}
    result['engine_directories']={route:str(path) for route,path in directories.items()}
    result['plugin_package']=package
    result['validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['seed']=9100
    report=args.output or ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history/002-minimal-compute.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    inherited=metadata['fm-inherited']
    required={'dw_replacements':24,'online_replacements':16,'f32_replacements':12,'int8_nonlinear_value_replacements':12,'int8_residual_replacements':24}
    omitted=inherited.get('omitted_inherited_mechanisms',[])
    mechanisms={'dw_replacements':'dw','online_replacements':'attention','f32_replacements':'f32_ffn','int8_nonlinear_value_replacements':'int8_value','int8_residual_replacements':'int8_residual'}
    for key,count in required.items():assert len(inherited[key])==(0 if mechanisms[key] in omitted else count)
    expected_attention=0 if 'attention' in omitted else 32 if 'nonlinear_attention' in omitted else 48
    assert sum(len(x['outputs']) for x in inherited['online_replacements'])==expected_attention
    result['custom_attention_branches']=expected_attention
    result['omitted_inherited_mechanisms']=inherited.get('omitted_inherited_mechanisms',[])
    result['rewrite_coverage']={key:len(inherited[key]) for key in required}
    stream=torch.cuda.Stream()
    for frames in (600,601,759,760,761,919,920):
        generator=torch.Generator(device='cuda').manual_seed(9100)
        inputs={key:torch.randn(batch,frames,100,device='cuda',generator=generator).contiguous() for key in ('x','text_condition','speech_condition')}
        inputs.update(t=torch.full((batch,1,1),.5,device='cuda'),guidance_scale=torch.ones(batch,1,1,device='cuda'),padding_mask=torch.zeros(batch,frames,dtype=torch.bool,device='cuda'))
        # Exercise actual masked suffixes, not only the all-valid branch.
        inputs['padding_mask'][:,-3:]=True
        stream.wait_stream(torch.cuda.current_stream())
        outputs={};permutation={}
        for route in ('fm-native','fm-inherited'):
            engine=Engine(directories[route]/'engine.plan',trt,torch,True,{k:tuple(v.shape) for k,v in inputs.items()})
            arena=torch.empty(engine.output_arena_size(),dtype=torch.uint8,device='cuda')
            engine.context.set_device_memory(arena.data_ptr(),arena.numel());engine.bind_output_arena(arena)
            with torch.cuda.stream(stream):
                for tensor in engine.outputs.values():tensor.fill_(float('nan'))
                got=engine(inputs,stream)
            stream.synchronize()
            assert set(got)=={'velocity'}
            output=got['velocity'].cpu()
            assert output.shape==(batch,frames,100) and torch.isfinite(output).all()
            outputs[route]=output
            if batch>1 and frames==760:
                with torch.cuda.stream(stream):
                    shuffled={k:v.roll(1,0).contiguous() for k,v in inputs.items()}
                    permuted=engine(shuffled,stream)['velocity']
                stream.synchronize()
                permuted=permuted.cpu();expected=output.roll(1,0)
                assert torch.isfinite(permuted).all()
                permutation[route]={'exact':bool(torch.equal(permuted,expected)),
                    'relative_l2':float((permuted-expected).norm()/expected.norm().clamp_min(1e-20)),
                    'max_abs':float((permuted-expected).abs().max())}
            del got,engine,arena
        original=outputs['fm-native'];actual=outputs['fm-inherited']
        result['tests'].append({'frames':frames,'finite_all':True,'all_rows_written':True,'relative_l2':float((actual-original).norm()/original.norm().clamp_min(1e-20)),'max_abs':float((actual-original).abs().max()),'batch_permutation':permutation})
        report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['tests'][-1]),flush=True)
    result['status']='minimal_compute_passed_audio_quality_pending'
    report.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
