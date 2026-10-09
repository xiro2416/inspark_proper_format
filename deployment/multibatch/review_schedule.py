"""Resolve noisy tile screens and preserve immutable engine/ONNX bindings."""
from deployment.multibatch.matrix import ROOT,BASE,read,save,run,benchmark,compare,lifecycle,audit,digest,verified_engine


def review(b):
    h=BASE/f'b{b}';decision=h/'schedule-review-decision.json'
    if decision.exists():return read(decision)
    baseline=h/'before-schedule-review-selected.json'
    current=read(baseline if baseline.exists() else h/'current-best-selected.json');screen=h/'compare-target-schedule.json'
    report=dict(batch=b,status='not_needed')
    if not screen.exists() or 'vocoder-target-schedule' in current['vocoder_plan']:
        report['reason']='No unresolved schedule screen, or selected schedule already validated';save(decision,report);return report
    ci=read(screen)['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms']
    if ci[1]<0:
        report['reason']='Screen excluded a useful gain';save(decision,report);return report
    directory=ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/vocoder-target-schedule'
    try:bound=verified_engine(directory,b)
    except RuntimeError:
        bound=False;report['historical_binding_mismatch']='Prior unselected candidate is retained as a diagnostic; fresh representation/build in independent directory'
    if not bound:
        directory=ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/vocoder-schedule-review'
        run(b,'schedule-review-export','deployment/multibatch/export_schedule.py','--batch',b,'--probe',h/'large-conv-tile-probe.json',
            '--residual-probe',h/'residual-conv-tile-probe.json','--out-dir',directory,build=True)
    if not verified_engine(directory,b):
        run(b,'schedule-review-build','scripts/build_unified_onnx.py','--gpu',1,'--onnx',directory/'model.onnx','--engine',directory/'model.engine',
            '--optimization-level',5,'--tiling','full','--aux-streams',0,'--small-fir-plugin','--small-fir-layout','all_tiled',
            '--implicit-int8-conv','--target-implicit-int8-schedule',build=True)
    baseline=h/'before-schedule-review-selected.json';save(baseline,current)
    candidate=h/'schedule-review-selected.json';save(candidate,dict(current,vocoder_plan=str(directory/'model.plan.json')))
    control=benchmark(b,baseline,'schedule-review-control',final=True);measured=benchmark(b,candidate,'schedule-review-candidate',final=True)
    ci=compare(b,control,measured,'compare-schedule-review');report.update(status='not_retained',gain_ci_ms=ci)
    key='wave_admission_to_last_pcm_ms'
    if ci[0]<=0 or measured['summary'][key]['p50']>=control['summary'][key]['p50']:
        report['reason']='Bound strong candidate showed no retained matched30wave gain';save(decision,report);return report
    lifecycle(b,candidate,'schedule-review');audit(b,candidate,'schedule-review')
    # Final paired check after complete validation protects against transient gains.
    control=benchmark(b,baseline,'schedule-review-final-control',final=True);measured=benchmark(b,candidate,'schedule-review-final-candidate',final=True)
    ci=compare(b,control,measured,'compare-schedule-review-final');report['final_gain_ci_ms']=ci
    if ci[0]<=0 or measured['summary'][key]['p50']>=control['summary'][key]['p50']:
        report['reason']='Gain did not persist after validation';save(decision,report);return report
    value=read(candidate);value['status']='validated_local_sm89_int8'
    destination=ROOT/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json';old=read(destination);save(destination,value)
    try:
        wav=ROOT/f'outputs/multibatch/b{b}/schedule-review-cli.wav'
        run(b,'schedule-review-cli','deployment/infer.py','--batch',b,'--ref-audio','/workspace/A_TEST_REF/male_news.wav','--text','你好，这是卷积调度的部署验证。','--output',wav)
        import numpy as np,soundfile as sf
        pcm,sr=sf.read(wav);assert sr==22050 and len(pcm)>0 and np.isfinite(pcm).all() and np.any(pcm)
        save(h/'schedule-review-cli.json',dict(passed=True,wav=str(wav),sha256=digest(wav)))
    except Exception:save(destination,old);raise
    save(h/'current-best-selected.json',value)
    import shutil
    shutil.copy2(h/'optimized-capture-acoustics.pt',h/'before-schedule-review-capture-acoustics.pt')
    shutil.copy2(h/'schedule-review-capture-acoustics.pt',h/'optimized-capture-acoustics.pt')
    original_path=h/'original-selected.json' if (h/'original-selected.json').exists() else h/'migration-selected.json'
    original=benchmark(b,original_path,'schedule-review-delivery-original',final=True);best=benchmark(b,h/'current-best-selected.json','schedule-review-delivery-best',final=True)
    gain=compare(b,original,best,'compare-schedule-review-delivery')
    def metrics(r):
        s=r['summary'][key];w=r['sustained_power']
        return dict(p50_ms=s['p50'],p95_ms=s['p95'],mean_power_w=w['power_w']['mean'],peak_power_w=w['power_w']['max'],first_chunk_requests_s=w['requests_per_second'])
    result=read(h/'optimization-complete.json');save(h/'before-schedule-review-optimization-complete.json',result)
    result.update(original=metrics(original),best=metrics(best),gain_ci_ms=gain,lifecycle='schedule-review-lifecycle.json',ar_audit='schedule-review-audit-ar.json',acoustic_audit='schedule-review-audit-acoustics.json',cli='schedule-review-cli.json',schedule_review='schedule-review-decision.json')
    report.update(status='retained_validated',reason='Matched30wave gain persisted; full lifecycle/audits/CLI passed')
    save(decision,report);save(h/'optimization-complete.json',result);return report
