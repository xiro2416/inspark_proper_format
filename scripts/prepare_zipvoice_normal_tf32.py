"""Isolate authorized ordinary TF32 normal attention without changing INT8 recipe."""
import json
from pathlib import Path
from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    baseline=json.loads((ROOT/'reports/sm89/zipvoice/a1007/migration-baseline.json').read_text())
    assert baseline['status']=='all_seven_migrations_accepted_optimization_pending'
    original=ROOT/'.work/zipvoice/b64/code';target=ROOT/'.work/zipvoice/b64_normtf32/code'
    target.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in sorted(original.glob('*.py')):
        text=path.read_text().replace('zipvoice_int8_b64','zipvoice_int8_b64_normtf32')
        edits=[]
        if path.name=='online_branch_stats_runtime_kernel.py':
            lines=text.splitlines();new=[]
            for line in lines:
                if "scores=tl.dot(q,k,input_precision='ieee')" in line:
                    line=line.replace("input_precision='ieee'","input_precision='tf32'");edits.append('QK TF32')
                if 'den=den*alpha+tl.sum(p,1)' in line:
                    line=line.replace('tl.sum(p,1)','p_full_sum');edits.append('denominator remains unrounded F32 probability sum')
                new.append(line)
                if 'q=tl.load(Q+' in line:
                    new.append("    q = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[q], dtype=tl.float32, is_pure=True, pack=1)");edits.append('Q RNA')
                if 'k=tl.load(K+' in line:
                    new.append("        k = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[k], dtype=tl.float32, is_pure=True, pack=1)");edits.append('K RNA')
                if 'v=tl.load(V+' in line:
                    new.extend(["        p_full_sum = tl.sum(p, 1)","        p = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[p], dtype=tl.float32, is_pure=True, pack=1)","        v = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[v], dtype=tl.float32, is_pure=True, pack=1)"]);edits.append('P/V RNA; keep full F32 sum')
            text='\n'.join(new)+'\n';assert len(edits)==5
        elif path.name=='online_branch_wide_aot.py':
            assert text.count("'ieee'")==3;text=text.replace("'ieee'","'tf32'");edits=['All3 normal wrappers AV TF32; geometry64x32 unchanged']
        elif path.name=='online_branch_wide_plugin.py':
            assert text.count("'tf32' not in compiled.asm['ptx'].lower()")==1
            text=text.replace("'tf32' not in compiled.asm['ptx'].lower()","('cvt.rna.tf32.f32' in compiled.asm['ptx'] and 'mma.sync' in compiled.asm['ptx'] and '.tf32.' in compiled.asm['ptx'])");edits=['Require actual RNA and TF32 MMA PTX; retain scratch/shared limits']
        dest=target/path.name;dest.write_text(text);records.append({'file':path.name,'source_sha256':sha(path),'candidate_sha256':sha(dest),'edits':edits})
    runtime_package(target,64,suffix='_normtf32')
    report={'status':'normal_tf32_candidate_prepared_not_built','batch':64,'suffix':'normtf32','plugin_package':'inspark_infer.ops.tensorrt.zipvoice.a1007.b64_normtf32','code':str(target.relative_to(ROOT)),'source_graph':'.work/zipvoice/b64/graphs/fm-inherited.onnx','files':records,'changes':'Normal QK/AV multiply becomes authorized ordinary TF32 RNA; Float32 IO/accum/stats. Original mask/position/online stats dependencies unchanged.','preserved':'All geometry, first4floating/last12INT8, source180weights/scales, nonlinear QK/AV, FFN/value/residual/DW implementations','next':'Matched samebatch 2-normal-branch write/read microprobe first; rebuild+mapping/quality/E2E required for retention'}
    (target.parent/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/sm89/zipvoice/a1007/normal-tf32-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'package':report['plugin_package']}))


if __name__=='__main__':main()
