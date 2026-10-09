"""Alternating 30-wave controls to resolve the small burst4 decision."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import run
    from benchmarks.unified_first_chunk import compare_reports
    h=ROOT/'deployment/b32/history';paths={'control':h/'implicit-validated-selected.json',
        'burst4':ROOT/'configs/hardware/sm89/indextts/int8_b32_runtime_burst4.json'}
    reports={};summaries=[]
    for index,label in enumerate(('control','burst4','control','burst4')):
        name=f'runtime-recheck-{index}-{label}-b32';out=h/(name+'.json')
        run(name,'benchmarks/benchmark_unified_first_chunk.py','run','--gpu',1,'--batch',32,
            '--manifest','deployment/history/validation-manifest.json','--deployment',paths[label],
            '--config','local_assets/runtime/runtime_fp32_b1.yaml','--out',out,'--warmups',5,'--waves',30,'--power-seconds',0,
            '--label',name,'--quant-recipe','int8_smoothquant_alpha1.0')
        report=json.loads(out.read_text());reports[index]=report
        summaries.append(dict(index=index,label=label,p50_ms=report['summary']['wave_admission_to_last_pcm_ms']['p50']))
        (h/'runtime-burst-recheck.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for a,b in ((0,1),(2,3)):
        c=compare_reports(reports[a],reports[b]);(h/f'compare-runtime-burst4-recheck-{a//2}.json').write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
