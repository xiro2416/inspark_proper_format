"""Check all actual protected FFN operands against GPU TF32 RNA before building."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TRITON_CACHE_DIR']='/workspace/A_1007/.cache/triton-actual-weight-pack';os.environ['CUDA_CACHE_PATH']='/workspace/A_1007/.cache/cuda';os.environ['TMPDIR']='/workspace/A_1007/.cache/tmp';os.environ['XDG_CACHE_HOME']='/workspace/A_1007/.cache'
import argparse,fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
from run_zipvoice_validation import ROOT,sha
from zipvoice_selected_route import idle_gpu_preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=int,choices=(8,16,32,64),default=64);args=parser.parse_args();batch=args.batch
    suffix='normtf32k16wp' if batch==64 else 'wp';control='normtf32k16' if batch==64 else 'inherited'
    lease=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=idle_gpu_preflight()
    import torch,triton,onnx,numpy as np
    source=ROOT/f'.work/zipvoice/b{batch}/graphs/fm-inherited.onnx'
    model=onnx.load(source,load_external_data=True);named={n.name:n for n in model.graph.node};prod={o:n for n in model.graph.node for o in n.output};initial={t.name:t for t in model.graph.initializer}
    def array(name):
        if name in initial:return onnx.numpy_helper.to_array(initial[name]).copy()
        node=prod[name]
        if node.op_type=='Constant':return onnx.numpy_helper.to_array(onnx.helper.get_attribute_value(node.attribute[0])).copy()
        assert node.op_type=='Identity';return array(node.input[0])
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
    helper=ROOT/f'.work/zipvoice/b{batch}_{suffix}/code/tf32_static_weight_pack.py';pack=module('actual_weight_pack',helper).pack_weight
    rounder=module('gpu_operand_round_reference',ROOT/'.work/f32-weight-prepack/kernel.py').round_static_weight
    build=json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{control}/build.json').read_text());rows=[]
    for record in build['f32_replacements']:
        prefix=record['module'];w=array(named[prefix+'in_proj/MatMul'].input[1]);digest=hashlib.sha256(w.tobytes()).hexdigest();assert digest==record['weight_sha256']
        cpu=pack(w);raw=torch.from_numpy(w).cuda();rounded=torch.empty_like(raw)
        rounder[(triton.cdiv(raw.numel(),256),)](raw,rounded,raw.numel(),256)
        torch.cuda.synchronize();got=rounded.cpu().numpy();assert np.array_equal(cpu.view(np.uint32),got.view(np.uint32)),prefix
        rows.append({'module':prefix,'shape':list(w.shape),'source_weight_sha256':digest,'packed_weight_sha256':hashlib.sha256(cpu.tobytes()).hexdigest(),'cpu_pack_gpu_rna_all_bits_exact':True,'source_weight_unchanged':hashlib.sha256(w.tobytes()).hexdigest()==digest})
        del raw,rounded
    assert len(rows)==12
    report={'status':'all12_actual_float32_ffn_weight_operands_gpu_rna_exact','preflight':preflight,'source_graph_sha256':sha(source),'helper_sha256':sha(helper),'gpu_reference_kernel_sha256':sha(ROOT/'.work/f32-weight-prepack/kernel.py'),'rows':rows,'scope':'Derived existing TF32 operand only; original checkpoint, source graph, INT8 weights/scales and recipe untouched. No new rounding/quantization math.'}
    (ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'/('038-actual-weight-prepack.json' if batch==64 else '041-actual-weight-prepack.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'weights':len(rows)}))


if __name__=='__main__':main()
