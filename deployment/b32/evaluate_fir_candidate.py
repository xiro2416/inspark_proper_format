"""Paired unprofiled application probe of the rebuilt full B32 FIR candidate."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import benchmark
    from benchmarks.unified_first_chunk import compare_reports
    h=ROOT/'deployment/b32/history';base=h/'migration-selected.json'
    candidate=json.loads(base.read_text())
    plan=ROOT/'artifacts/sm89/int8_smoothquant/b32/vocoder-tiled-fir/model.plan.json'
    p=json.loads(plan.read_text());layers=json.loads(plan.with_name('model.inspector.json').read_text())['Layers']
    integers=[l for l in layers if 'i8i32' in l.get('TacticName','')]
    plugins=[l for l in layers if 'small_fir_activation_tiled' in str(l) or 'b32_tiled_fir_' in str(l)]
    if not integers or len(plugins)!=109:raise RuntimeError('Lost integer compute or incomplete FIR plugin coverage')
    (h/'fir-candidate-engine-evidence.json').write_text(json.dumps(dict(engine_sha256=p['sha256'],int8_tactic_layers=len(integers),fir_plugin_layers=len(plugins),plan=str(plan),inspector=str(plan.with_name('model.inspector.json'))),indent=2)+'\n')
    candidate.update(vocoder_plan=str(plan),status='local_sm89_fir_candidate_not_yet_validated')
    path=ROOT/'configs/hardware/sm89/indextts/int8_b32_tiled_fir_candidate.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
    # Fresh adjacent baseline/candidate comparison, same case sequence and settings.
    baseline=benchmark(32,base,'optimization-fir-control-b32')
    result=benchmark(32,path,'optimization-fir-candidate-b32')
    if any(result['measured_counters'][k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Candidate first-head fallback')
    comparison=compare_reports(baseline,result)
    (h/'compare-fir-b32.json').write_text(json.dumps(comparison,indent=2)+'\n')
    key='wave_admission_to_last_pcm_ms'
    print(json.dumps(dict(control_p50=baseline['summary'][key]['p50'],candidate_p50=result['summary'][key]['p50'],candidate=str(path))),flush=True)


if __name__=='__main__':main()
