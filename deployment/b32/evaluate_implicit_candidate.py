"""Compare complete custom INT8 implicit convolution with the validated FIR best."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    from deployment.validate_matrix import benchmark
    from benchmarks.unified_first_chunk import compare_reports
    h=ROOT/'deployment/b32/history';base=h/'current-best-selected.json'
    original=json.loads(base.read_text());(h/'tiled-fir-validated-selected.json').write_text(json.dumps(original,indent=2)+'\n')
    plan=ROOT/'artifacts/sm89/int8_smoothquant/b32/vocoder-implicit-int8/model.plan.json'
    p=json.loads(plan.read_text());layers=json.loads(plan.with_name('model.inspector.json').read_text())['Layers']
    count=lambda text:sum(text in str(layer) for layer in layers)
    fir=count('b32_tiled_fir_');conv=count('b32_implicit_conv_')
    if fir!=109 or conv!=76:raise RuntimeError(f'Missing complete custom coverage: fir={fir}, conv={conv}')
    (h/'implicit-candidate-engine-evidence.json').write_text(json.dumps(dict(engine_sha256=p['sha256'],fir_plugin_layers=fir,int8_convolution_plugin_layers=conv,
        actual_integer_compute='all AOT compilations assert PTX signed INT8 MMA; discrete geometry and same-scale synthetic tests passed',
        native_int8_tactic_layers=sum('i8i32' in l.get('TacticName','') for l in layers),plan=str(plan)),indent=2)+'\n')
    candidate=dict(original,vocoder_plan=str(plan),status='local_sm89_implicit_candidate_not_yet_validated')
    path=ROOT/'configs/hardware/sm89/indextts/int8_b32_implicit_candidate.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
    baseline=benchmark(32,h/'tiled-fir-validated-selected.json','optimization-implicit-control-b32')
    result=benchmark(32,path,'optimization-implicit-candidate-b32')
    if any(result['measured_counters'][k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Candidate first-head fallback')
    comparison=compare_reports(baseline,result);(h/'compare-implicit-b32.json').write_text(json.dumps(comparison,indent=2)+'\n')
    key='wave_admission_to_last_pcm_ms'
    print(json.dumps(dict(control_p50=baseline['summary'][key]['p50'],candidate_p50=result['summary'][key]['p50'],candidate=str(path))),flush=True)


if __name__=='__main__':main()
