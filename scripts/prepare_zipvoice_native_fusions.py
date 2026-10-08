"""Restore original framework regions while testing retained source DW/residual fusions."""
import argparse
import json
from pathlib import Path

from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,choices=(1,2,4),default=[1,2])
    args=parser.parse_args()
    for batch in args.batches:
        source=ROOT/f'.work/zipvoice/b{batch}_geo1/code'
        destination=ROOT/f'.work/zipvoice/b{batch}_geo3/code'
        destination.mkdir(parents=True,exist_ok=False)
        before={p.name:sha(p) for p in source.glob('*.py')}
        for p in source.glob('*.py'):
            (destination/p.name).write_text(p.read_text().replace(f'zipvoice_int8_b{batch}_geo1',f'zipvoice_int8_b{batch}_geo3'))
        assert before=={p.name:sha(p) for p in source.glob('*.py')}
        runtime_package(destination,batch,suffix='_geo3')
        result=json.loads((source.parent/'preparation.json').read_text())
        result.update(status='native_regions_source_fusion_candidate_prepared_not_built',suffix='geo3',
                      plugin_package=f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}_geo3',
                      code=str(destination.relative_to(ROOT)),omit_inherited=['attention','f32_ffn','int8_value'],
                      source_candidate_unchanged=True,source_candidate_preparation_sha256=sha(source.parent/'preparation.json'))
        result['choice']={**result['choice'],'framework_regions':['original_attention','original_f32_ffn','original_int8_value'],
                          'retained_source_custom_regions':['int8_depthwise','int8_projection_residual'],
                          'reason':'Full source geometry and native nonlinear-only hybrid remain slower than strong native baseline; test restoration of shared framework regions while retaining inexpensive source fusions.'}
        result['files']=[{'file':p.name,'source_sha256':before[p.name],'candidate_sha256':sha(p)} for p in sorted(destination.glob('*.py'))]
        (destination.parent/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'batch':batch,'status':result['status'],'omit_inherited':result['omit_inherited']}))


if __name__=='__main__':main()
