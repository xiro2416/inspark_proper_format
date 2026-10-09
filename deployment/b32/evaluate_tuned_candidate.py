"""A/B compare only the six measured convolution schedule changes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import benchmark
    from benchmarks.unified_first_chunk import compare_reports
    h=ROOT/'deployment/b32/history';base=h/'implicit-validated-selected.json'
    plan=ROOT/'artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-tuned/model.plan.json'
    p=json.loads(plan.read_text());layers=json.loads(plan.with_name('model.inspector.json').read_text())['Layers']
    conv=sum('b32_implicit_conv_' in str(l) for l in layers);fir=sum('b32_tiled_fir_' in str(l) for l in layers)
    if conv!=76 or fir!=109:raise RuntimeError('Lost required custom compute coverage')
    (h/'tuned-candidate-engine-evidence.json').write_text(json.dumps(dict(engine_sha256=p['sha256'],int8_convolution_plugin_layers=conv,fir_plugin_layers=fir,tuned_convolutions=6,ptx='every AOT compilation asserts signed INT8 MMA'),indent=2)+'\n')
    candidate=json.loads(base.read_text());candidate.update(vocoder_plan=str(plan),status='local_sm89_tuned_candidate_not_yet_validated')
    path=ROOT/'configs/hardware/sm89/indextts/int8_b32_tuned_candidate.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
    control=benchmark(32,base,'optimization-tuned-control-b32');result=benchmark(32,path,'optimization-tuned-candidate-b32')
    c=compare_reports(control,result);(h/'compare-tuned-b32.json').write_text(json.dumps(c,indent=2)+'\n')
    key='wave_admission_to_last_pcm_ms';print(json.dumps(dict(control_p50=control['summary'][key]['p50'],candidate_p50=result['summary'][key]['p50'],gain_ci=c['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms'])))


if __name__=='__main__':main()
