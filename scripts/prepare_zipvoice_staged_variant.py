"""Select existing kernel geometries by verified model downsampling stage."""
import argparse
import ast
import copy
import json
from pathlib import Path

from prepare_zipvoice_variant import prepare
from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,choices=(8,16,32),default=[8,16,32])
    parser.add_argument('--suffix',default='geo1')
    args=parser.parse_args()
    config=ROOT/'models/zipvoice/config/model.json'
    factors=json.loads(config.read_text())['model']['fm_decoder_downsampling_factor']
    assert factors==[1,2,4,2,1],'Reassess all geometry choices for a changed model'
    quarter=[i for i,factor in enumerate(factors) if factor==4]
    desired=json.loads((ROOT/'reports/sm89/zipvoice/a1007/compute-adaptation-review.json').read_text())['candidate_choices']
    def condition(variable):
        return ast.parse(' or '.join(f"'/encoders.{i}/' in {variable}" for i in quarter),mode='eval').body
    def read(path):return ast.parse(path.read_text())
    def write(path,tree):path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    for batch in args.batches:
        policy=copy.deepcopy(desired[str(batch)])
        base=copy.deepcopy(policy)
        for key in ('nonlinear_QB','nonlinear_warps','int8_value_BM','int8_residual_BM'):
            if isinstance(base[key],dict):base[key]=base[key]['full']
        prepare(batch,args.suffix,base)
        code=ROOT/f'.work/zipvoice/b{batch}_{args.suffix}/code'
        namespace=f'zipvoice_int8_b{batch}_{args.suffix}'
        edits=[]

        if isinstance(policy['nonlinear_QB'],dict) or isinstance(policy['nonlinear_warps'],dict):
            qb=policy['nonlinear_QB'].get('quarter') if isinstance(policy['nonlinear_QB'],dict) else policy['nonlinear_QB']
            warps=policy['nonlinear_warps'].get('quarter') if isinstance(policy['nonlinear_warps'],dict) else policy['nonlinear_warps']
            tree=read(code/'online_nonlinear_rna_aot.py')
            fn=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='nonlinear_rna_aot'))
            fn.name='nonlinear_rna_quarter_aot'
            for node in ast.walk(fn):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='online_nonlinear_rna':node.args[9]=ast.Constant(qb)
            tree.body.append(fn);write(code/'online_nonlinear_rna_aot.py',tree)
            tree=read(code/'online_nonlinear_rna_plugin.py')
            class Quarter(ast.NodeTransformer):
                def visit_FunctionDef(self,node):
                    node.name=node.name.replace('OnlineNonlinearRNAFloat32','OnlineNonlinearRNAQuarterFloat32')
                    return self.generic_visit(node)
                def visit_Constant(self,node):
                    if isinstance(node.value,str):node.value=node.value.replace('OnlineNonlinearRNAFloat32','OnlineNonlinearRNAQuarterFloat32')
                    return node
                def visit_Name(self,node):
                    if node.id=='nonlinear_rna_aot':node.id='nonlinear_rna_quarter_aot'
                    return node
                def visit_Dict(self,node):
                    node=self.generic_visit(node)
                    for i,key in enumerate(node.keys):
                        if isinstance(key,ast.Constant) and key.value=='num_warps':node.values[i]=ast.Constant(warps)
                    return node
                def visit_Assign(self,node):
                    node=self.generic_visit(node)
                    if len(node.targets)==1 and isinstance(node.targets[0],ast.Attribute):
                        if node.targets[0].attr=='grid_x':node.value=ast.parse(f'(q.shape_expr[2]+{qb-1})//{qb}',mode='eval').body
                        if node.targets[0].attr=='block_x':node.value=ast.Constant(warps*32)
                    return node
            functions=[Quarter().visit(copy.deepcopy(x)) for x in tree.body if isinstance(x,ast.FunctionDef)]
            tree.body.append(ast.ImportFrom(module='online_nonlinear_rna_aot',names=[ast.alias(name='nonlinear_rna_quarter_aot')],level=0))
            tree.body.extend(functions);write(code/'online_nonlinear_rna_plugin.py',tree)
            tree=read(code/'normal_tf32_rewrite.py')
            class NonlinearSelection(ast.NodeTransformer):
                def visit_Attribute(self,node):
                    node=self.generic_visit(node)
                    if node.attr=='OnlineNonlinearRNAFloat32':
                        other=copy.deepcopy(node);other.attr='OnlineNonlinearRNAQuarterFloat32'
                        return ast.IfExp(condition('domain'),other,node)
                    return node
            write(code/'normal_tf32_rewrite.py',NonlinearSelection().visit(tree))
            edits.append({'mechanism':'nonlinear_attention','quarter_stages':quarter,'QB':qb,'KB':16,'warps':warps,'other_stages':'original32x16/w4'})

        if isinstance(policy['int8_value_BM'],dict):
            bm=policy['int8_value_BM']['quarter']
            assert base['int8_value_BM']==128 and bm==64
            tree=read(code/'int8_nonlinear_value_aot.py')
            fn=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='nonlinear_value_128x32'))
            fn.name='nonlinear_value_64x32'
            for node in ast.walk(fn):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='int8_nonlinear_value':node.args[8]=ast.Constant(64)
            tree.body.append(fn);write(code/'int8_nonlinear_value_aot.py',tree)
            tree=read(code/'int8_nonlinear_value_plugin.py')
            class ValueQuarter(ast.NodeTransformer):
                def visit_FunctionDef(self,node):node.name+='quarter64';return self.generic_visit(node)
                def visit_Constant(self,node):
                    if isinstance(node.value,str):node.value=node.value.replace('OriginalInt8NonlinearValue128x32','OriginalInt8NonlinearValue64x32')
                    return node
                def visit_Name(self,node):
                    if node.id=='nonlinear_value_128x32':node.id='nonlinear_value_64x32'
                    return node
                def visit_Call(self,node):
                    node=self.generic_visit(node)
                    if isinstance(node.func,ast.Name) and node.func.id=='launch':node.args[2]=ast.Constant(64)
                    return node
            functions=[ValueQuarter().visit(copy.deepcopy(x)) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('desc_128','aot_128')]
            assert len(functions)==2
            tree.body.append(ast.ImportFrom(module='int8_nonlinear_value_aot',names=[ast.alias(name='nonlinear_value_64x32')],level=0))
            tree.body.extend(functions);write(code/'int8_nonlinear_value_plugin.py',tree)
            tree=read(code/'int8_nonlinear_value_rewrite.py')
            class ValueSelection(ast.NodeTransformer):
                def visit_Call(self,node):
                    node=self.generic_visit(node)
                    if isinstance(node.func,ast.Attribute) and node.func.attr=='OriginalInt8NonlinearValue128x32':
                        other=copy.deepcopy(node.func);other.attr='OriginalInt8NonlinearValue64x32'
                        node.func=ast.IfExp(condition('p'),other,node.func)
                    return node
            write(code/'int8_nonlinear_value_rewrite.py',ValueSelection().visit(tree))
            edits.append({'mechanism':'int8_value','quarter_stages':quarter,'quarter_BM':64,'other_BM':128,'BN':32,'BK':64})

        if isinstance(policy['int8_residual_BM'],dict):
            bm=policy['int8_residual_BM']['quarter'];full=base['int8_residual_BM']
            assert bm==64 and full==16
            tree=read(code/'i8_residual_rewrite.py')
            class ResidualSelection(ast.NodeTransformer):
                def visit_Call(self,node):
                    node=self.generic_visit(node)
                    if isinstance(node.func,ast.Attribute) and node.func.attr=='OriginalInt8Residual16x64':
                        other=copy.deepcopy(node.func);other.attr='OriginalInt8Residual64x64'
                        node.func=ast.IfExp(condition('p'),other,node.func)
                    return node
            write(code/'i8_residual_rewrite.py',ResidualSelection().visit(tree))
            edits.append({'mechanism':'int8_residual','quarter_stages':quarter,'quarter_BM':64,'half_and_full_BM':16,'BN':64,'BK':64})

        runtime_package(code,batch,suffix='_'+args.suffix)
        path=code.parent/'preparation.json';data=json.loads(path.read_text())
        for entry in data['files']:entry['candidate_sha256']=sha(code/entry['file'])
        data.update(choice=policy,uniform_base_choice=base,stage_factory_selections=edits,
                    model_config_sha256=sha(config),downsampling_factors=factors,
                    primary760_internal_frames=[760//f for f in factors],
                    runtime_branch_or_extra_empty_ctas=False)
        path.write_text(json.dumps(data,indent=2)+'\n')
        print(json.dumps({'batch':batch,'status':'stage_specific_variant_prepared_not_built','stage_factory_selections':edits}))


if __name__=='__main__':main()
