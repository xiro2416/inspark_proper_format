"""Isolate native nonlinear attention while retaining the validated geometry candidate."""
import argparse
import ast
import copy
import json
from pathlib import Path

from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,choices=(1,2,4),default=[1,2])
    args=parser.parse_args()
    for batch in args.batches:
        source=ROOT/f'.work/zipvoice/b{batch}_geo1/code'
        destination=ROOT/f'.work/zipvoice/b{batch}_geo2/code'
        destination.mkdir(parents=True,exist_ok=False)
        before={p.name:sha(p) for p in source.glob('*.py')}
        for p in source.glob('*.py'):
            text=p.read_text().replace(f'zipvoice_int8_b{batch}_geo1',f'zipvoice_int8_b{batch}_geo2')
            if p.name=='normal_tf32_rewrite.py':
                tree=ast.parse(text)
                fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rewrite')
                fn.args.args.append(ast.arg(arg='skip_nonlinear'));fn.args.defaults.append(ast.Constant(False))
                loops=[x for x in ast.walk(fn) if isinstance(x,ast.For) and isinstance(x.iter,ast.Call) and x.iter.args and isinstance(x.iter.args[0],ast.Name) and x.iter.args[0].id=='branches']
                assert len(loops)==1
                loops[0].body.insert(0,ast.If(test=ast.parse('i==0 and skip_nonlinear',mode='eval').body,body=[ast.Continue()],orelse=[]))
                adjusted=0
                for node in ast.walk(fn):
                    if isinstance(node,ast.Compare) and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==48 and 'outputs' in ast.unparse(node.left):
                        node.comparators[0]=ast.IfExp(ast.Name(id='skip_nonlinear',ctx=ast.Load()),ast.Constant(32),ast.Constant(48));adjusted+=1
                assert adjusted==1,'Expected the original full attention coverage assertion'
                for node in ast.walk(fn):
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='append' and node.args and isinstance(node.args[0],ast.Dict):
                        d=node.args[0]
                        if any(isinstance(k,ast.Constant) and k.value=='all4normal_heads_preserved' for k in d.keys):
                            d.keys.append(ast.Constant('nonlinear_custom_plugin'));d.values.append(ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='skip_nonlinear',ctx=ast.Load())))
                text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
            (destination/p.name).write_text(text)
        assert before=={p.name:sha(p) for p in source.glob('*.py')}
        runtime_package(destination,batch,suffix='_geo2')
        result=json.loads((source.parent/'preparation.json').read_text())
        result.update(status='target_nonlinear_ablation_prepared_not_built',suffix='geo2',
                      plugin_package=f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}_geo2',
                      code=str(destination.relative_to(ROOT)),omit_inherited=['nonlinear_attention'],
                      source_candidate_unchanged=True,source_candidate_preparation_sha256=sha(source.parent/'preparation.json'))
        result['choice']={**result['choice'],'nonlinear_backend':'original_onnx_TensorRT','reason':'geo1 geometry improves inherited but still slower than native; targeted nonlinear omission, normal write/read cache and other compute mechanisms retained.'}
        result['files']=[{'file':p.name,'source_sha256':before[p.name],'candidate_sha256':sha(p)} for p in sorted(destination.glob('*.py'))]
        (destination.parent/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'batch':batch,'status':result['status'],'omit_inherited':result['omit_inherited']}))


if __name__=='__main__':main()
