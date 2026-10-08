"""Isolate existing TF32 operand prepacking while preserving original checkpoints."""
import json,argparse
from pathlib import Path
from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=int,choices=(8,16,32,64),default=64);args=parser.parse_args();batch=args.batch
    suffix='normtf32k16wp' if batch==64 else 'wp'
    source_suffix='normtf32k16' if batch==64 else ''
    screen=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'/('034-f32-weight-prepack.json' if batch==64 else '039-f32-weight-prepack-transfer.json')
    evidence=json.loads(screen.read_text());assert len(evidence['records'])==18 and all(x['all_elements_bitwise_equal'] for x in evidence['records'])
    source=ROOT/f'.work/zipvoice/b{batch}{"_"+source_suffix if source_suffix else ""}/code';target=ROOT/f'.work/zipvoice/b{batch}_{suffix}/code';target.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in sorted(source.glob('*.py')):
        text=path.read_text().replace(f'zipvoice_int8_b{batch}{"_"+source_suffix if source_suffix else ""}',f'zipvoice_int8_b{batch}_{suffix}')
        if path.name=='linear_f32_activation_tf32_rna_kernel.py':
            needle="            w=tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;',constraints='=f,f',args=[w],dtype=tl.float32,is_pure=True,pack=1)"
            assert text.count(needle)==1;text=text.replace(needle,'            # W is the exact existing TF32 RNA operand, prepacked once by the builder.')
        if path.name=='f32_tf32_rna_rewrite.py':
            text=text.replace('import onnx,numpy as np','import onnx,numpy as np\nfrom tf32_static_weight_pack import pack_weight')
            needle='        args=[x]\n';assert text.count(needle)==1
            text=text.replace(needle,"        raw_weight_sha256=hashlib.sha256(w.tobytes()).hexdigest()\n        w=pack_weight(w)\n"+needle)
            text=text.replace("'weight_sha256':hashlib.sha256(w.tobytes()).hexdigest()", "'source_weight_sha256':raw_weight_sha256,'weight_sha256':hashlib.sha256(w.tobytes()).hexdigest(),'weight_transform':'Exact existing TF32 RNA operand cached once; original checkpoint/scales unchanged'")
        dest=target/path.name;dest.write_text(text);records.append({'file':path.name,'source_sha256':sha(path),'candidate_sha256':sha(dest)})
    helper=target/'tf32_static_weight_pack.py'
    helper.write_text('''"""Pack normal finite Float32 constants to their existing TF32 RNA operand."""
import numpy as np

def pack_weight(weight):
    assert weight.dtype==np.float32 and np.isfinite(weight).all()
    magnitude=np.abs(weight)
    assert np.all((magnitude==0)|(magnitude>=np.finfo(np.float32).tiny))
    assert magnitude.max()<np.float32(2.**126)
    bits=weight.copy().view(np.uint32)
    # RNA rounds a tie away from zero. Sign stays intact; normal exponent carry is preserved.
    return ((bits+np.uint32(0x1000))&np.uint32(0xffffe000)).view(np.float32).copy()
''')
    runtime_package(target,batch,suffix='_'+suffix)
    report={'status':'existing_math_f32_prepack_candidate_prepared_not_built','batch':batch,'suffix':suffix,'plugin_package':f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}_{suffix}','code':str(target.relative_to(ROOT)),'source_graph':f'.work/zipvoice/b{batch}/graphs/fm-inherited.onnx','files':records,'helper_sha256':sha(helper),'screen_sha256':sha(screen),'preserved':'Original checkpoints and all INT8 weights/scales/1:3; normalTF32k16, nonlinear and other kernels unchanged. Float32 FFN IO/accum/bias/activation and exact preexisting TF32 operands.','next':'GPU RNA validation of all12actual weights before AOT; full source/candidate model state/PCM identity and matched currentbest E2E required'}
    (target.parent/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/sm89/zipvoice/a1007'/('f32-prepack-preparation.json' if batch==64 else f'b{batch}-f32-prepack-preparation.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'package':report['plugin_package']}))


if __name__=='__main__':main()
