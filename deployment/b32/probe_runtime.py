"""Focused B32 runtime options against the retained compute implementation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import benchmark
    from benchmarks.unified_first_chunk import compare_reports
    from inspark_infer.runtime.unified_deployment import validate
    h=ROOT/'deployment/b32/history';source=h/'current-best-selected.json'
    original=json.loads(source.read_text());baseline_path=h/'implicit-validated-selected.json';baseline_path.write_text(json.dumps(original,indent=2)+'\n')
    configurations=[('burst1',dict(graph_burst_rounds=1)),('burst4',dict(graph_burst_rounds=4)),
        ('packing',dict(admission_packing=True,latent_vector_pack=True)),
        ('flat-projection',dict(condition_flat_projection=True))]
    baseline=benchmark(32,baseline_path,'runtime-control-b32');reports={};summary=[]
    for label,changes in configurations:
        candidate=dict(original,**changes,status='local_sm89_runtime_candidate_not_yet_validated');validate(candidate)
        path=ROOT/f'configs/hardware/sm89/indextts/int8_b32_runtime_{label}.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
        result=benchmark(32,path,f'runtime-{label}-b32');reports[label]=result
        if not result['execution_pass'] or any(result['measured_counters'][k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Runtime option lost execution coverage')
        compare=compare_reports(baseline,result);(h/f'compare-runtime-{label}-b32.json').write_text(json.dumps(compare,indent=2)+'\n')
        metric='wave_admission_to_last_pcm_ms';summary.append(dict(label=label,path=str(path),changes=changes,p50_ms=result['summary'][metric]['p50'],baseline_p50_ms=baseline['summary'][metric]['p50'],
            gain_ci=compare['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms'],
            actual_rounds=result['measured_counters']['device_round_launched_rounds'],status_reads=result['measured_counters']['device_round_status_reads']))
        (h/'runtime-options.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary[-1]),flush=True)


if __name__=='__main__':main()
