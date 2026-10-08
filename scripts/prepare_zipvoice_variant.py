"""Prepare independent geometry candidates from preserved target source code."""
import argparse
import ast
import json
from pathlib import Path
import re

from prepare_zipvoice_a1007 import ROOT, runtime_package, sha


def prepare(batch, suffix, choice_override=None):
    review=ROOT/'reports/sm89/zipvoice/a1007/compute-adaptation-review.json'
    choice=choice_override or json.loads(review.read_text())['candidate_choices'][str(batch)]
    assert batch in (1,2,4,8,16,32)
    original=ROOT/f'.work/zipvoice/b{batch}/code'
    target=ROOT/f'.work/zipvoice/b{batch}_{suffix}/code'
    target.mkdir(parents=True,exist_ok=False)
    before={p.name:sha(p) for p in original.glob('*.py')}
    records=[]
    residual_bm=choice['int8_residual_BM']
    value_bm=choice['int8_value_BM']
    ffn_bm=choice['f32_ffn_BM']
    nonlinear_qb=choice['nonlinear_QB']

    class Constants(ast.NodeTransformer):
        def __init__(self,mapping):self.mapping=mapping
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in self.mapping:
                return ast.copy_location(ast.Constant(self.mapping[node.value]),node)
            return node

    class Geometry(ast.NodeTransformer):
        def __init__(self,module):self.module=module;self.function=None;self.edits=[]
        def visit_FunctionDef(self,node):
            previous=self.function;self.function=node.name
            if self.module=='i8_residual_aot' and node.name=='residual_128x64' and residual_bm!=64:
                node.name=f'residual_{residual_bm}x64'
            if self.module=='int8_nonlinear_value_aot' and node.name=='nonlinear_value_128x32':
                node.name=f'nonlinear_value_{value_bm}x32'
            node=self.generic_visit(node);self.function=previous;return node
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if not isinstance(node.func,ast.Name):return node
            index,value=None,None
            if self.module=='f32_tf32_rna_aot' and node.func.id=='linear_f32_activation':index,value=7,ffn_bm
            if self.module=='online_nonlinear_rna_aot' and node.func.id=='online_nonlinear_rna':index,value=9,nonlinear_qb
            if self.module=='normal_tf32_aot' and node.func.id=='online_branch_stats' and choice['normal_attention']=='candidate64x16':index,value=11,16
            if self.module=='i8_residual_aot' and self.function=='residual_128x64' and residual_bm!=64:index,value=10,residual_bm
            if self.module=='int8_nonlinear_value_aot' and self.function=='nonlinear_value_128x32':index,value=8,value_bm
            if self.module in ('i8_residual_plugin','int8_nonlinear_value_plugin') and node.func.id=='launch' and isinstance(node.args[0],ast.Name):
                if node.args[0].id==f'residual_{residual_bm}x64':index,value=2,residual_bm
                if node.args[0].id==f'nonlinear_value_{value_bm}x32':index,value=2,value_bm
            if index is not None:
                old=ast.literal_eval(node.args[index]);node.args[index]=ast.Constant(value)
                self.edits.append({'function':self.function,'argument':index,'before':old,'after':value})
            return node
        def visit_Assign(self,node):
            node=self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr=='grid_x':
                mapping={127:ffn_bm-1,128:ffn_bm} if self.module=='f32_tf32_rna_plugin' else {31:nonlinear_qb-1,32:nonlinear_qb} if self.module=='online_nonlinear_rna_plugin' else {}
                if mapping:
                    old=ast.unparse(node.value);node.value=Constants(mapping).visit(node.value)
                    self.edits.append({'function':self.function,'grid_before':old,'grid_after':ast.unparse(node.value)})
            return node

    for path in sorted(original.glob('*.py')):
        text=path.read_text().replace(f'zipvoice_int8_b{batch}',f'zipvoice_int8_b{batch}_{suffix}')
        if path.stem in ('i8_residual_plugin','i8_residual_rewrite'):
            text=text.replace('OriginalInt8Residual128x64',f'OriginalInt8Residual{residual_bm}x64')
            if residual_bm!=64:text=text.replace('residual_128x64',f'residual_{residual_bm}x64')
            elif path.stem=='i8_residual_rewrite':pass
        if path.stem in ('int8_nonlinear_value_plugin','int8_nonlinear_value_rewrite'):
            text=text.replace('OriginalInt8NonlinearValue128x32',f'OriginalInt8NonlinearValue{value_bm}x32').replace('nonlinear_value_128x32',f'nonlinear_value_{value_bm}x32')
        # Existing 64x64 residual factory is already correct; avoid registering it twice.
        if path.stem=='i8_residual_plugin' and residual_bm==64:
            text=path.read_text().replace(f'zipvoice_int8_b{batch}',f'zipvoice_int8_b{batch}_{suffix}')
        transformer=Geometry(path.stem)
        tree=transformer.visit(ast.parse(text));tree=ast.fix_missing_locations(tree)
        destination=target/path.name;destination.write_text(ast.unparse(tree)+'\n')
        records.append({'file':path.name,'source_sha256':before[path.name],
                        'candidate_sha256':sha(destination),'geometry_edits':transformer.edits})
    expected={'f32_tf32_rna_aot':1,'f32_tf32_rna_plugin':3,'online_nonlinear_rna_aot':1,'online_nonlinear_rna_plugin':1,'int8_nonlinear_value_aot':1,'int8_nonlinear_value_plugin':1}
    if residual_bm!=64:expected.update(i8_residual_aot=1,i8_residual_plugin=1)
    if choice['normal_attention']=='candidate64x16':expected['normal_tf32_aot']=3
    for module,count in expected.items():
        actual=next(x for x in records if x['file']==module+'.py')
        assert len(actual['geometry_edits'])==count,(module,actual)
    assert before=={p.name:sha(p) for p in original.glob('*.py')},'Baseline plugin source changed'
    runtime_package(target,batch,suffix='_'+suffix)
    result={'status':'geometry_variant_prepared_not_built','batch':batch,'suffix':suffix,
            'plugin_package':f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}_{suffix}',
            'code':str(target.relative_to(ROOT)),
            'source_graph':f'.work/zipvoice/b{batch}/graphs/fm-inherited.onnx',
            'choice':choice,'review_sha256':sha(review),'files':records,
            'baseline_source_unchanged':True,'weight_scale_or_batch_role_changes':False}
    p=target.parent/'preparation.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','choice')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,choices=(1,2,4),default=[1,2,4])
    parser.add_argument('--suffix',default='geo1')
    args=parser.parse_args();assert re.fullmatch('[a-z][a-z0-9_]*',args.suffix)
    for batch in args.batches:prepare(batch,args.suffix)
