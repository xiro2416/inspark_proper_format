"""One focused composition of independently useful source runtime mechanisms."""
from deployment.multibatch.matrix import ROOT,BASE,read,save,benchmark,compare,lifecycle,audit,run,digest


def review(batch):
    b=batch;h=BASE/f'b{b}';decision=h/'runtime-composition-decision.json'
    if decision.exists():return read(decision)
    baseline=h/'before-runtime-composition-selected.json'
    current=read(baseline if baseline.exists() else h/'current-best-selected.json');rows=read(h/'runtime-options.json')
    reference=h/'composition-reference-selected.json';save(reference,current)
    screen_control=read(h/'runtime-control.json')['summary']['wave_admission_to_last_pcm_ms']['p50']
    unresolved=sorted((r for r in rows if r['paired_mean_gain_ci'][0]<=0<r['paired_mean_gain_ci'][1]
        and 100*(screen_control-r['p50_ms'])/screen_control>=1),key=lambda r:r['p50_ms'])
    # Resolve the strongest material point estimate after the initial screen,
    # against the actual evolving best; uncertainty alone is not a rejection.
    screened=[]
    for r in unresolved:
        value=dict(current,**r['changes'])
        if value==current:continue
        path=h/('uncertain-'+r['name']+'-selected.json');save(path,value)
        control=benchmark(batch,reference,'uncertain-'+r['name']+'-control',final=True)
        measured=benchmark(batch,path,'uncertain-'+r['name']+'-candidate',final=True)
        ci=compare(batch,control,measured,'compare-uncertain-'+r['name'])
        screened.append(dict(name=r['name'],gain_ci_ms=ci,p50_ms=measured['summary']['wave_admission_to_last_pcm_ms']['p50']))
        r.update(paired_mean_gain_ci=ci,p50_ms=measured['summary']['wave_admission_to_last_pcm_ms']['p50'])
    save(h/'runtime-uncertainty-review.json',screened)
    positive=sorted((r for r in rows if r['paired_mean_gain_ci'][0]>0),key=lambda r:r['p50_ms'])
    candidate=dict(current);selected=[];assigned=set()
    for r in positive:
        changes={}
        for k,v in r['changes'].items():
            if k in assigned:continue
            old=current.get(k,2 if k=='graph_burst_rounds' else False)
            if k=='graph_burst_rounds' and old!=2:continue  # preserve accepted burst, rather than choose another competitor
            if old!=v:changes[k]=v;assigned.add(k)
        if changes:candidate.update(changes);selected.append(dict(name=r['name'],changes=changes))
    report=dict(batch=b,status='not_needed',selected_mechanisms=selected,uncertainty_review='runtime-uncertainty-review.json')
    # Need an interaction question: current best must already use a tested change,
    # or at least two individually useful mechanisms must be combined.
    migrated=read(h/'migration-selected.json')
    runtime_changed=any(current.get(k)!=migrated.get(k) for k in (set(current)|set(migrated))-{'vocoder_plan','status'})
    if not selected or (not runtime_changed and len(selected)<2 and not any(r['gain_ci_ms'][0]>0 for r in screened)):
        report['reason']='No unresolved combination of independent positive runtime mechanisms';save(decision,report);return report
    from inspark_infer.runtime.unified_deployment import validate
    validate(candidate)
    baseline=h/'before-runtime-composition-selected.json';save(baseline,current)
    path=h/'runtime-composition-selected.json';save(path,candidate)
    control=benchmark(b,baseline,'composition-control',final=True);measured=benchmark(b,path,'composition-candidate',final=True)
    ci=compare(b,control,measured,'compare-runtime-composition');report['gain_ci_ms']=ci
    key='wave_admission_to_last_pcm_ms'
    if ci[0]<=0 or measured['summary'][key]['p50']>=control['summary'][key]['p50']:
        report.update(status='not_retained',reason='Matched30wave composition did not improve the current best');save(decision,report);return report
    lifecycle(b,path,'runtime-composition');audit(b,path,'runtime-composition')
    final=benchmark(b,path,'composition-final',final=True)
    value=read(path);value['status']='validated_local_sm89_int8'
    destination=ROOT/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json';old=read(destination);save(destination,value)
    try:
        wav=ROOT/f'outputs/multibatch/b{b}/composition-cli.wav'
        run(b,'composition-cli','deployment/infer.py','--batch',b,'--ref-audio','/workspace/A_TEST_REF/male_news.wav','--text','你好，这是调度组合的部署验证。','--output',wav)
        import numpy as np,soundfile as sf
        pcm,sr=sf.read(wav);assert sr==22050 and len(pcm)>0 and np.isfinite(pcm).all() and np.any(pcm)
        save(h/'composition-cli.json',dict(passed=True,wav=str(wav),sha256=digest(wav)))
    except Exception:save(destination,old);raise
    save(h/'current-best-selected.json',value)
    # Keep the frozen capture tied to the newly retained runtime for residual fusion.
    import shutil
    shutil.copy2(h/'optimized-capture-acoustics.pt',h/'before-composition-capture-acoustics.pt')
    shutil.copy2(h/'runtime-composition-capture-acoustics.pt',h/'optimized-capture-acoustics.pt')
    original_path=h/'original-selected.json' if (h/'original-selected.json').exists() else h/'migration-selected.json'
    original=benchmark(b,original_path,'composition-delivery-original',final=True)
    best=benchmark(b,h/'current-best-selected.json','composition-delivery-best',final=True)
    gain=compare(b,original,best,'compare-composition-delivery')
    def metrics(r):
        s=r['summary'][key];w=r['sustained_power']
        return dict(p50_ms=s['p50'],p95_ms=s['p95'],mean_power_w=w['power_w']['mean'],peak_power_w=w['power_w']['max'],first_chunk_requests_s=w['requests_per_second'])
    result=read(h/'optimization-complete.json');save(h/'before-composition-optimization-complete.json',result)
    result.update(original=metrics(original),best=metrics(best),gain_ci_ms=gain,lifecycle='runtime-composition-lifecycle.json',ar_audit='runtime-composition-audit-ar.json',acoustic_audit='runtime-composition-audit-acoustics.json',cli='composition-cli.json',runtime_composition='runtime-composition-decision.json')
    report.update(status='retained_validated',reason='Matched30wave gain, complete lifecycle/AR/acoustic audit and CLI passed')
    save(decision,report);save(h/'optimization-complete.json',result);return report
