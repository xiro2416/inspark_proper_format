"""Matched baseline/A/B/C/D readiness trials on the authorized single GPU."""
import argparse,json,os,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def plan_identity(plan):
    """Compare every deployment field while allowing relocation of its bundle root."""
    anchor = plan['target_plan']
    base, marker, _ = anchor.partition('/artifacts/')
    if not marker:
        raise ValueError('Deployment target path has no artifact root')
    def canonical(value):
        if isinstance(value, str) and value.startswith(base + '/'):
            return 'bundle://' + value[len(base) + 1:]
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        return value
    return canonical(plan)


def main():
    p=argparse.ArgumentParser();p.add_argument('--waves',type=int,default=30);p.add_argument('--warmups',type=int,default=5);p.add_argument('--power-seconds',type=float,default=30);p.add_argument('--trace-mode',choices=('B','D'));p.add_argument('--lifecycle',action='store_true');p.add_argument('--acoustic-priority',type=int,choices=(-1,0),default=0);p.add_argument('--modes',nargs='+',choices=('baseline','A','B','C','D'),default=['baseline','A','B','C','D']);p.add_argument('--reference',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Physical GPU1 only')
    from inspark_infer.runtime.device import select_gpu,GPULease
    select_gpu(1)
    import torch
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.pool import _engine_stats
    from benchmarks.unified_first_chunk import load_manifest,wave_cases,run_wave,distribution
    from benchmarks.benchmark_unified_first_chunk import prepare_references
    from benchmarks.board_power import BoardSampler
    from deployment.ready_pipeline.scheduler import ReadyPipeline,selected,internal_plan
    from deployment.runtime_assets import verify_engine as verified_engine
    report=dict(status='incomplete',gpu=1,modes={},scope='First-chunk readiness experiment; baseline+four scheduling controls, identical64 input cases/seeds per wave, no production default change')
    def save():
        a.out.parent.mkdir(parents=True,exist_ok=True);temp=a.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.out)
    for b in (16,32,48,64):
        for component in ('target','draft'):
            if not verified_engine(ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/{component}',b):raise RuntimeError('Missing validated engine')
    manifest=load_manifest(ROOT/'deployment/multibatch/history/validation-manifest-128.json');cases=manifest['splits']['evaluation']
    report['manifest_sha256']=manifest['manifest_sha256'];report['source_plan']=selected(64)
    fingerprints={};e=None;pipeline=None;sampler=None
    reference=json.loads(a.reference.read_text()) if a.reference else None
    if reference:
        if reference['status']!='validated' or reference['manifest_sha256']!=manifest['manifest_sha256'] or plan_identity(reference['source_plan'])!=plan_identity(selected(64)):raise RuntimeError('Reference identity mismatch')
        for mode in ('baseline','A','C'):
            for wave in reference['modes'][mode]['waves']:
                fingerprints[mode,wave['wave'] if isinstance(wave['wave'],int) else int(wave['wave'].rsplit('-',1)[1])]={r['case_id']:dict(code=r['code_sha256'],pcm=r['pcm_sha256'],accepted=r['accepted'],rounds=r['rounds']) for r in wave['rows']}
        report['reference']=str(a.reference.resolve())

    def trial(mode,i,details=True):
        e.head_ready_pipeline=None if mode=='baseline' else pipeline;pipeline.mode=mode
        value=run_wave(e,wave_cases(cases,64,i),f'{mode}-{i}',details=details,admission_mode='batch')
        rows=value['rows'];ordered=sorted(r['admission_to_pcm_ms'] for r in rows)
        value['latency_summary']=dict(first16_ms=ordered[15],first32_ms=ordered[31],all64_ms=ordered[63],request_p50_ms=statistics.median(ordered),request_p95_ms=distribution(ordered)['p95'])
        if len(rows)!=64 or len({r['case_id'] for r in rows})!=64:raise RuntimeError('Missing/duplicate request')
        if details:
            fp={r['case_id']:dict(code=r['code_sha256'],pcm=r['pcm_sha256'],accepted=r['accepted'],rounds=r['rounds']) for r in rows}
            if reference and (mode,i) in fingerprints and fp!=fingerprints[mode,i]:raise RuntimeError('Same-route repeated output mismatch')
            if mode in ('baseline','A','C'):fingerprints[mode,i]=fp
            if mode in ('A','B'):
                if any(fp[k]['code']!=fingerprints['baseline',i][k]['code'] or fp[k]['accepted']!=fingerprints['baseline',i][k]['accepted'] for k in fp):raise RuntimeError('Same-shape AR schedule mismatch')
            if mode in ('B','D') and fp!=fingerprints[('A' if mode=='B' else 'C'),i]:raise RuntimeError('Serial/overlap route mismatch')
        if mode!='baseline':value['execution']=pipeline.waves[-1]
        s=_engine_stats(e)
        if s['sessions'] or s['active_rows'] or s['error_sessions']:raise RuntimeError('Request state leak')
        if e.device_round_fallbacks or e.student.fallbacks or e.vocoder.fallbacks or pipeline.acoustic.student.fallbacks or pipeline.acoustic.vocoder.fallbacks:raise RuntimeError('Unexpected compute fallback')
        return value
    with GPULease(1) as lease:
        report['initial_board_memory_mib']=lease.initial_memory_mib
        sampler=BoardSampler(1);sampler.start()
        try:
            e=Engine(dict(load(str(ROOT/'local_assets/runtime/runtime_fp32_b1.yaml')),max_batch=64,precision_batches=[64]))
            prepare_references(e,manifest);report['deployment']=e.prepare_deployment(selected(64));pipeline=ReadyPipeline(e,acoustic_priority=a.acoustic_priority);report['acoustic_priority']=a.acoustic_priority
            if reference:
                report['migration_b48']=dict(reference['migration_b48'],reused_from=report['reference'])
            else:
                # Validate internal B48 before enabling the new scheduling route.
                from scripts.audit_unified_ar import target_reference,draft_reference
                from scripts.audit_unified_acoustics import metrics
                with torch.cuda.stream(e.model.stream),torch.inference_mode():
                    batch_cases=wave_cases(cases,64,0)[:48]
                    payload=[dict(request_id=f'migration48-{i}',voice_id=c['voice_id'],text=c['text'],seed=c['seed'],emotion=c['emotion'],finish=True) for i,c in enumerate(batch_cases)]
                    e.admit_batch(payload);sessions=[e.sessions[c['request_id']] for c in payload];rows=e.prepare_rows(sessions,0)
                    for session,row in zip(sessions,rows):session['_row']=row
                    c=pipeline.controllers[48];c.begin(rows);r,provider=c.runtime,c.provider
                    first=r.past+1-r.mel_lengths;dp=first[:,None]+r.step7;tp=first[:,None]+r.step8
                    frozen=dict(anchors=r.last.clone(),draft_cache=provider.draft_cache.clone(),draft_lengths=r.draft_lengths.clone(),draft_positions=dp,target_cache=provider.target_cache.clone(),target_keep=provider.target_keep.clone(),target_lengths=r.past.clone())
                    provider.active.fill_(True)
                    hidden,base=provider.draft(r.last,dp,r.draft_lengths);actual_draft=dict(draft_hidden=hidden.clone(),draft_base=base.clone())
                    proposed,_,_=r.proposal.sample_uniform(hidden,base,r.draws.at(r.rounds)[:,:7],r.last)
                    tokens=torch.cat((r.last[:,None],proposed),1);model=e.rt.engine.target.model
                    frozen['target_x']=model.embeddings(tokens)+model.text_pos_embedding.emb(tp)
                    logits,hidden,final=provider.target(tokens,tp,r.past);actual_target=dict(target_logits=logits.clone(),target_selected=hidden.clone(),target_final=final.clone())
                    ref_draft=draft_reference(e.rt.backbone,frozen);ref_target=target_reference(e.rt.engine.target,frozen,True)
                    audit={k:metrics(ref_draft[k],v) for k,v in actual_draft.items()};audit.update({k:metrics(ref_target[k],v) for k,v in actual_target.items()})
                    if any(not x['finite'] for x in audit.values()):raise RuntimeError('Nonfinite B48 audit')
                    stats=c.run(rows)
                    if any(not x[-1] for x in c.runtime.result()['metadata']):raise RuntimeError('B48 head incomplete')
                    report['migration_b48']=dict(status='validated_compute',audit=audit,stats=stats,scope='real48 prefixes imported once from source64 prefill; direct Draft/Target and same-calibrated Torch audit, complete48 head; fixed original recipe/backend')
                    del frozen,ref_draft,ref_target,actual_target,actual_draft,hidden,base,logits,final
                    for item in payload:e.cancel(item['request_id'])
            save();print('B48 migration compute validated',flush=True)
            modes=a.modes
            for mode in modes:
                for i in range(a.warmups):trial(mode,i)
            for mode in modes:report['modes'][mode]=dict(waves=[])
            if a.trace_mode:
                e.configure_profiling(False,True)
                torch.cuda.profiler.start()
                for i in range(a.waves):report['modes'][a.trace_mode]['waves'].append(trial(a.trace_mode,i))
                torch.cuda.profiler.stop()
            else:
                for i in range(a.waves):
                    # Preserve correctness references A before B and C before D;
                    # alternate the relative order of the two paired families.
                    order=(modes if i%2==0 else list(reversed(modes))) if reference else (['baseline','A','B','C','D'] if i%2==0 else ['baseline','C','D','A','B'])
                    for mode in order:
                        v=trial(mode,i);report['modes'][mode]['waves'].append(v)
                        print(json.dumps(dict(mode=mode,wave=i,**v['latency_summary'])),flush=True)
                    save()
            for mode,entry in report['modes'].items():
                if entry['waves']:entry['summary']={k:distribution(x['latency_summary'][k] for x in entry['waves']) for k in entry['waves'][0]['latency_summary']}
            if a.power_seconds and not a.trace_mode:
                report['sustained']={}
                for mode in modes:
                    for i in range(a.warmups):trial(mode,i,False)
                    before_graphs=dict(e.head_graphs.hits if mode=='baseline' else pipeline.acoustic.head_graphs.hits)
                    start=time.perf_counter();n=0;lat=[]
                    while time.perf_counter()-start<a.power_seconds:
                        v=trial(mode,n%a.waves,False);n+=1;lat.append(v['latency_summary'])
                    end=time.perf_counter();after_graphs=e.head_graphs.hits if mode=='baseline' else pipeline.acoustic.head_graphs.hits
                    # Early EOS/prompt groups can add groups; require all actual
                    # dispatched groups, rather than fabricate four work counts.
                    hits={k:after_graphs[k]-before_graphs[k] for k in after_graphs}
                    if any(v<(n if mode=='baseline' else 4*n) for v in hits.values()):raise RuntimeError('Missing acoustic graph work')
                    report['sustained'][mode]=dict(seconds=end-start,waves=n,requests=64*n,sensor=sampler.window(start,end),graph_hits=hits,latency={k:distribution(x[k] for x in lat) for k in lat[0]},samples=[list(x) for x in sampler.samples if start<=x[0]<=end])
                    print(json.dumps(dict(power_mode=mode,**report['sustained'][mode]['sensor'])),flush=True);save()
            if a.lifecycle:
                lifecycle=[]
                for n in (1,15,17):
                    outputs={}
                    for mode in ('C','D'):
                        e.head_ready_pipeline=pipeline;pipeline.mode=mode
                        v=run_wave(e,wave_cases(cases,64,0)[:n],f'partial-{mode}-{n}',details=True,admission_mode='batch')
                        outputs[mode]={x['case_id']:(x['pcm_sha256'],x['code_sha256'],x['accepted']) for x in v['rows']}
                        state=_engine_stats(e)
                        if state['sessions'] or state['active_rows'] or state['error_sessions']:raise RuntimeError('Partial cancellation leak')
                    if outputs['C']!=outputs['D']:raise RuntimeError('Partial replay mismatch')
                    lifecycle.append(dict(requests=n,serial_overlap_identical=True,cancelled_and_clean=True))
                # Synthetic already-terminal first-head fixture tests the EOS
                # handoff/crop path, not model sampling quality or performance.
                eos_outputs={}
                for mode in ('C','D'):
                    with torch.cuda.stream(e.model.stream),torch.inference_mode():
                        case=wave_cases(cases,64,0)[0];ident='eos-fixture-'+mode
                        e.admit_batch([dict(request_id=ident,voice_id=case['voice_id'],text=case['text'],seed=case['seed'],emotion=case['emotion'],finish=True)])
                        session=e.sessions[ident];row=e.prepare_rows([session],0)[0];session['_row']=row
                        eos=e.rt.engine.target.gpt.stop_mel_token
                        row.codes=[row.codes[0].clone(),torch.tensor([100],device='cuda',dtype=torch.long),torch.tensor([eos],device='cuda',dtype=torch.long)];row.done=True
                        pipeline.mode=mode;items=pipeline.run([row],{id(row):session},None)
                        import hashlib
                        if len(items)!=1 or not items[0]['chunk']['eos'] or len(items[0]['chunk']['pcm'])==0:raise RuntimeError('EOS head fixture failed')
                        eos_outputs[mode]=hashlib.sha256(items[0]['chunk']['pcm'].tobytes()).hexdigest()
                        e.cancel(ident)
                if eos_outputs['C']!=eos_outputs['D']:raise RuntimeError('EOS serial/overlap PCM mismatch')
                lifecycle.append(dict(synthetic_early_eos=True,serial_overlap_pcm_identical=True,cancelled_and_clean=True,scope='handoff and short PCM crop, not model-generated EOS or quality'))
                with torch.cuda.stream(e.model.stream),torch.inference_mode():
                    payload=[dict(request_id=f'move48-{i}',voice_id=x['voice_id'],text=x['text'],seed=x['seed'],emotion=x['emotion'],finish=True) for i,x in enumerate(wave_cases(cases,64,0))]
                    e.admit_batch(payload);sessions=[e.sessions[x['request_id']] for x in payload];rows=e.prepare_rows(sessions,0)
                    for session,row in zip(sessions,rows):session['_row']=row
                    source=pipeline.controllers[64];dest=pipeline.controllers[48];source.begin(rows);source.runtime.advance_burst()
                    from deployment.ready_pipeline.scheduler import state_tensors,transfer
                    indices=torch.tensor(list(range(63,15,-1)),device='cuda',dtype=torch.long)
                    transfer(source,dest,indices)
                    if any(not torch.equal(dst,src.index_select(axis,indices)) for (src,axis),(dst,_) in zip(state_tensors(source),state_tensors(dest))):raise RuntimeError('GPU B48 state migration mismatch')
                    dest.runtime.run()
                    if not dest.runtime.ready.all().item():raise RuntimeError('Migrated B48 continuation incomplete')
                    for item in payload:e.cancel(item['request_id'])
                lifecycle.append(dict(noncontiguous_gpu_transfer='64->48 after2 real rounds, all KV/token/accepted/draw/counters checked; continue to48 ready',cancelled_and_clean=True))
                report['lifecycle']=lifecycle
            report['after']=_engine_stats(e);report['acoustic_stats']=pipeline.acoustic.head_graphs.stats();report['status']='validated';save()
        except Exception as err:
            report.update(status='incomplete',error=repr(err));save();raise
        finally:
            if pipeline:pipeline.close()
            if e:e.head_ready_pipeline=None;e.close()
            if sampler:sampler.stop()

if __name__=='__main__':main()
