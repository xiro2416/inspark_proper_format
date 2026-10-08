"""Prepare the screened B64 TF32 normal Q64/K16 candidate independently."""
import json
from pathlib import Path
from prepare_zipvoice_a1007 import ROOT,runtime_package,sha


def main():
    report=ROOT/'reports/sm89/zipvoice/a1007/b64/history/031-normal-tf32-geometry.json'
    screen=json.loads(report.read_text())
    assert screen['status']=='synthetic_normal_tf32_geometry_microprobe_not_deployment_acceptance'
    source=ROOT/'.work/zipvoice/b64_normtf32/code';target=ROOT/'.work/zipvoice/b64_normtf32k16/code'
    assert screen['source_sha256']['q64k16']==sha(source/'online_branch_stats_runtime_kernel.py')
    rows=[x for x in screen['records'] if x['candidate']=='q64k16']
    assert len(rows)==9 and all(x['gain_percent']>0 and x['fits_existing_shared_scratch_contract'] for x in rows)
    target.mkdir(parents=True,exist_ok=False);files=[]
    for path in sorted(source.glob('*.py')):
        text=path.read_text().replace('zipvoice_int8_b64_normtf32','zipvoice_int8_b64_normtf32k16')
        if path.name=='online_branch_wide_aot.py':
            assert text.count("False,64,32,'tf32'")==3
            text=text.replace("False,64,32,'tf32'","False,64,16,'tf32'")
        destination=target/path.name;destination.write_text(text)
        files.append({'file':path.name,'source_sha256':sha(path),'candidate_sha256':sha(destination)})
    runtime_package(target,64,suffix='_normtf32k16')
    result={'status':'screened_normal_tf32_q64_k16_prepared_not_built','batch':64,'suffix':'normtf32k16','plugin_package':'inspark_infer.ops.tensorrt.zipvoice.a1007.b64_normtf32k16','code':str(target.relative_to(ROOT)),'source_graph':'.work/zipvoice/b64/graphs/fm-inherited.onnx','screen_sha256':sha(report),'normal_geometry':{'query':64,'key':16,'warps':4,'stages':1},'files':files,'preserved':'Original source weights/scales, SmoothQuant1:3, all non-normal kernels; Float32 IO/accum/stats and unrounded den; masks/position/stats dependencies unchanged','limits':'Synthetic projected potential1.185percent prompts AOT/complete mapping/quality/currentbestE2E; not deployment acceptance'}
    (target.parent/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/sm89/zipvoice/a1007/normal-k16-preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'package':result['plugin_package']}))


if __name__=='__main__':main()
