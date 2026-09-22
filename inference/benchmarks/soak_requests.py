#!/usr/bin/env python3
"""Full-EOS request soak on one GPU; concurrency is independent of model batch.

Defaults: sequential 1/4/8/16-concurrency tiers, 600 seconds admission per tier,
then drain. No claim of numerical parity or all-TRT coverage is made here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COUNTERS=('target_calls','device_target_steps','draft_backbone_calls','device_draft_steps',
          'native_target_steps','native_draft_steps','device_round_attempts',
          'device_round_successes','device_round_fallbacks','native_cfm_calls',
          'native_cfm_fallbacks','native_vocoder_calls','native_vocoder_fallbacks')


def workspace_path(value):
    path=Path(value).resolve()
    if not path.is_relative_to('/workspace'):raise ValueError('Paths must stay inside /workspace')
    return path


def compact_stats(stats,include_routes=False):
    required=('sessions','active_rows','error_sessions','memory','target_slots','draft_slots','rng_policy',
              'scheduler_max_batch','target_batch_counts',*COUNTERS)
    if any(key not in stats for key in required):raise RuntimeError('Missing lifecycle/memory metrics')
    result={key:stats[key] for key in (*required,'strict_request_isolation') if key in stats}
    for name in ('target_slots','draft_slots'):
        slots=stats[name]
        if slots is not None and (not slots['valid'] or slots['free']!=slots['free_unique']):
            raise RuntimeError(f'Invalid slot ownership: {name}: {slots}')
    for name in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','rss_bytes'):
        if not isinstance(stats['memory'].get(name),int) or stats['memory'][name]<0:
            raise RuntimeError(f'Missing/invalid memory metric: {name}')
    if include_routes:
        result['acoustic_routes']={name:dict(graph_routes=value.get('graph_routes',[]),
                                           wrapper_total=value.get('wrapper_total',{}))
                                   for name,value in stats.get('acoustic_routes',{}).items()}
    return result


def assert_drained(stats):
    if stats['sessions'] or stats['active_rows'] or stats['error_sessions']:
        raise RuntimeError('Requests remain after complete release/cancellation')
    for name in ('target_slots','draft_slots'):
        slots=stats[name]
        if slots is not None and (slots['leased'] or slots['free']!=slots['capacity'] or not slots['valid']):
            raise RuntimeError(f'Slots were not returned after wave: {name}')


def strict_plan(plan,enabled):
    plan=dict(plan)
    overrides={}
    if enabled:
        for key,value in (('strict_request_isolation',True),('batched_proposal_rng',False),
                          ('device_round_b8',False),('device_parent_graph',False),
                          ('device_residual',False),('device_accept_plan',False)):
            if plan.get(key)!=value:overrides[key]=dict(before=plan.get(key),after=value)
            plan[key]=value
    return plan,overrides


def request_record(identifier,state,started,finished,first_received,cancelled):
    if state.get('error'):raise RuntimeError(f'{identifier}: {state["error"]}')
    chunks=state.get('chunks',[]);end=0;digest=hashlib.sha256();counts=[]
    for index,chunk in enumerate(chunks):
        if chunk['index']!=index or chunk['sample_start']!=end:
            raise RuntimeError(f'{identifier}: non-contiguous PCM chunks')
        size=chunk['sample_end']-chunk['sample_start']
        if size!=len(chunk['pcm']) or size<0:raise RuntimeError(f'{identifier}: invalid PCM extent')
        if index==0 and not (size==44*256 or (chunk['eos'] and 0<size<44*256)):
            raise RuntimeError(f'{identifier}: first packet violates 44-frame/early-EOS contract')
        digest.update(chunk['pcm'].tobytes());end=chunk['sample_end']
        counts.append(dict(index=index,samples=size,eos=bool(chunk['eos']),
                           cfm_batch=chunk['cfm_batch'],vocoder_batch=chunk['vocoder_batch']))
    if not cancelled and (not state.get('complete') or not chunks or not chunks[-1]['eos'] or end<=0):
        raise RuntimeError(f'{identifier}: completion without full EOS audio')
    return dict(id=identifier,outcome='cancelled' if cancelled else 'complete_eos',
                complete=bool(state.get('complete')),eos=bool(chunks and chunks[-1]['eos']),
                chunks=counts,samples=end,pcm16_sha256=digest.hexdigest(),
                code_count=len(state.get('codes',[])),rounds=state.get('rounds'),
                kv_length=state.get('kv_head_lengths'),
                first_chunk_ms=None if first_received is None else (first_received-started)*1000,
                lifetime_ms=(finished-started)*1000)


def check_expected_errors(pool):
    """Real worker error path, followed by cancellation and reuse of the same ID."""
    identifier='soak-error-reuse'
    checks=[]
    for _ in range(2):
        pool.create_session(identifier,'reference',123)
        try:
            try:pool.finish_input(identifier)
            except RuntimeError as error:
                if 'No spoken text in request' not in str(error):raise
                checks.append('empty_input_rejected')
            else:raise RuntimeError('Empty input was unexpectedly accepted')
        finally:pool.cancel(identifier)
    assert_drained(compact_stats(pool.stats()[0]))
    return checks


RNG_HASHES=('state_sha256','next_uniform_prefix_sha256','cfm_noise_prefix_sha256')


def compare_rng_snapshots(before,after):
    for snapshot in (before,after):
        if any(not isinstance(snapshot.get(key),str) or len(snapshot[key])!=64 for key in RNG_HASHES):
            raise RuntimeError('Missing actual request RNG hash')
    changed=[key for key in (*RNG_HASHES,'seed','rounds','code_count','phase') if before.get(key)!=after.get(key)]
    return dict(passed=not changed,changed_fields=changed,before=before,after=after)


def check_runtime_rng(pool,case):
    """Real request boundaries only; not a claim of tokens equal across batches."""
    witness='soak-rng-reuse';other='soak-rng-other';created=set();checks={}
    def create(identifier,seed):
        pool.create_session(identifier,'reference',seed,case['emotion']);created.add(identifier)
    def cancel(identifier):pool.cancel(identifier);created.remove(identifier)
    try:
        create(witness,case['seed']);initial=pool.request_rng_snapshot(witness)
        create(other,case['seed']+1);pool.push_text(other,case['text']);pool.finish_input(other)
        pool.tick();cancel(other)
        checks['other_inference_and_cancel']=compare_rng_snapshots(initial,pool.request_rng_snapshot(witness))
        cancel(witness);create(witness,case['seed'])
        checks['cancel_recreate_same_seed']=compare_rng_snapshots(initial,pool.request_rng_snapshot(witness))
        pool.push_text(witness,case['text']);pool.finish_input(witness);pool.tick()
        active=pool.request_rng_snapshot(witness)
        create(other,case['seed']+2);cancel(other)
        checks['active_row_admit_cancel']=compare_rng_snapshots(active,pool.request_rng_snapshot(witness))
        checks['active_row_observed']=active['phase']=='active_row'
        cancel(witness);create(witness,case['seed'])
        checks['active_cancel_recreate_same_seed']=compare_rng_snapshots(initial,pool.request_rng_snapshot(witness))
        cancel(witness)
        checks['passed']=checks['active_row_observed'] and all(value['passed'] for value in checks.values() if isinstance(value,dict))
        checks['scope']='Unchanged witness progress across unrelated operations; post-numerical-branch stream/token equality is not asserted'
        assert_drained(compact_stats(pool.stats()[0]))
        return checks
    finally:
        import sys
        from acc_infer_clear.runtime.cleanup import cleanup_all
        cleanup_all([(identifier,lambda identifier=identifier:pool.cancel(identifier))
                     for identifier in created],primary=sys.exception())


def run_wave(pool,cases,concurrency,wave,timeout,cancel_every=11,clock=time.perf_counter):
    """Bounded waves prevent head-priority scheduling from starving old tails."""
    active={};first={};records=[];created=[];phase='admit';current=None
    long_cases=sorted(cases,key=lambda case:len(case['text']))[-max(1,len(cases)//4):]
    def collect(events):
        for event in events:
            identifier=event['request_id']
            if identifier not in active:raise RuntimeError(f'Event from inactive request: {identifier}')
            first.setdefault(identifier,event.get('received',clock()))
            if event['complete']:
                row=active.pop(identifier);state=pool.result(identifier)
                record=request_record(identifier,state,row['started'],clock(),first[identifier],False)
                record.update(case_id=row['case']['id'],long_text=row['long'],text_characters=len(row['case']['text']),seed=row['case']['seed'])
                records.append(record);pool.release(identifier)
    try:
        for lane in range(concurrency):
            sequence=wave*concurrency+lane;long=sequence%7==0
            case=(long_cases if long else cases)[sequence%len(long_cases if long else cases)]
            # IDs are reused every wave; seed remains the unchanged corpus request seed.
            identifier=f'soak-lane-{lane}'
            current=identifier
            started=clock();pool.create_session(identifier,'reference',case['seed'],case['emotion'],arrival=started)
            created.append(identifier);active[identifier]=dict(started=started,case=case,long=long,
                cancel=bool(cancel_every and sequence%cancel_every==cancel_every-1))
            pool.push_text(identifier,case['text']);pool.finish_input(identifier)
        admitted=compact_stats(pool.stats()[0])
        if admitted['sessions']!=concurrency:raise RuntimeError('Observed admission concurrency differs from request')
        # One bounded round permits cancellation while Target/Draft slots are live.
        phase='bounded_tick';current=None
        collect(pool.tick())
        phase='cancel'
        for identifier,row in list(active.items()):
            if row['cancel']:
                current=identifier
                state=pool.result(identifier)
                record=request_record(identifier,state,row['started'],clock(),first.get(identifier),True)
                record.update(case_id=row['case']['id'],long_text=row['long'],text_characters=len(row['case']['text']),seed=row['case']['seed'])
                pool.cancel(identifier);active.pop(identifier);records.append(record)
        phase='complete_eos_drain';current=None
        while active:
            if clock()-min(row['started'] for row in active.values())>timeout:
                raise TimeoutError('Full-EOS request exceeded the declared timeout')
            events=pool.run_ready()
            if not events:raise RuntimeError('No scheduler progress while full-EOS requests remain')
            collect(events)
        snapshot=compact_stats(pool.stats()[0]);assert_drained(snapshot)
        snapshot['admitted_concurrency']=admitted['sessions']
        if pool.owners:raise RuntimeError('Pool ownership survived wave cleanup')
        return records,snapshot
    except Exception as error:
        error.soak_partial_wave=dict(phase=phase,current_request=current,completed_records=records,
            pending=[dict(id=identifier,case_id=row['case']['id'],seed=row['case']['seed'],
                          text_characters=len(row['case']['text'])) for identifier,row in active.items()])
        raise
    finally:
        # Preserve the original failure while still trying every remaining request.
        import sys
        from acc_infer_clear.runtime.cleanup import cleanup_all
        primary=sys.exception()
        cleanup_all([(identifier,lambda identifier=identifier:pool.cancel(identifier))
                     for identifier in created if identifier in pool.owners],primary=primary)


def distribution(values):
    if not values:return dict(n=0)
    values=sorted(values)
    return dict(n=len(values),min=values[0],median=statistics.median(values),
                p95=values[round((len(values)-1)*.95)],max=values[-1],mean=statistics.fmean(values))


def memory_gate(before,snapshots,args):
    points=[dict(elapsed_s=0,memory=before['memory']),*snapshots]
    late=points[len(points)//2:]
    span=late[-1]['elapsed_s']-late[0]['elapsed_s'] if late else 0
    sufficient=len(late)>=args.memory_min_samples and span>=args.memory_min_span_seconds
    results={}
    for name,growth_limit,slope_limit in (
        ('cuda_allocated_bytes',args.max_live_growth_mib,args.max_live_slope_mib_per_min),
        ('rss_bytes',args.max_rss_growth_mib,args.max_rss_slope_mib_per_min)):
        baseline=before['memory'][name]
        growth=[(point['memory'][name]-baseline)/1024**2 for point in points]
        mean_x=statistics.fmean(point['elapsed_s'] for point in late)
        mean_y=statistics.fmean(point['memory'][name]/1024**2 for point in late)
        denominator=sum((point['elapsed_s']-mean_x)**2 for point in late)
        slope=(sum((point['elapsed_s']-mean_x)*(point['memory'][name]/1024**2-mean_y) for point in late)
               /denominator*60) if denominator else None
        results[name]=dict(final_growth_mib=growth[-1],peak_drained_growth_mib=max(growth),
            growth_limit_mib=growth_limit,growth_passed=max(growth)<=growth_limit,
            late_slope_mib_per_min=slope,slope_limit_mib_per_min=slope_limit,
            trend_passed=(slope is not None and slope<=slope_limit) if sufficient else None)
    return dict(passed=all(value['growth_passed'] and value['trend_passed'] is True for value in results.values()),
        absolute_passed=all(value['growth_passed'] for value in results.values()),
        trend_sufficient=sufficient,late_samples=len(late),late_span_seconds=span,
        min_samples=args.memory_min_samples,min_span_seconds=args.memory_min_span_seconds,
        metrics=results,scope='Post-wave fully drained live CUDA/RSS memory; reserved allocator cache is recorded but not treated as a leak')


def pass_gate(result,args):
    completed=[row for row in result['requests'] if row['outcome']=='complete_eos']
    witnesses=[dict(id=row['id'],case_id=row['case_id'],kv_length=row['kv_length'],code_count=row['code_count'],
                    text_characters=row.get('text_characters')) for row in completed
               if isinstance(row.get('kv_length'),int) and row['kv_length']>=args.min_kv_length]
    target_delta={batch:count-result['before']['target_batch_counts'].get(batch,0)
                  for batch,count in result['after']['target_batch_counts'].items()}
    target_batches=[int(batch) for batch,count in target_delta.items() if count>0]
    acoustic_batches=[chunk[key] for row in result['requests'] for chunk in row['chunks'] for key in ('cfm_batch','vocoder_batch')]
    observed_max=max(target_batches+acoustic_batches+[0])
    memory=memory_gate(result['before'],result['snapshots'],args)
    soak=args.seconds>=600
    checks=dict(requested_duration=result['elapsed_s']>=args.seconds,complete_eos=bool(completed),
        long_sequence=bool(witnesses),actual_concurrency=bool(result['snapshots']) and
            all(point['admitted_concurrency']==result['concurrency'] for point in result['snapshots']),
        microbatch_bound=observed_max<=result['model_max_batch'] and
            all(point['scheduler_max_batch']==result['model_max_batch'] for point in result['snapshots']) and
            all(count>=0 for count in target_delta.values()),
        per_request_legacy_rng=result['after']['rng_policy']=='legacy_per_request',
        rng_boundaries=result['rng_boundary_checks']['passed'],empty_input_error_recovery=len(result['expected_error_checks'])==2,
        memory_absolute=memory['absolute_passed'],memory_trend=(memory['passed'] if soak else True))
    passed=all(checks.values())
    return dict(passed=passed,validation_level='soak' if soak else 'smoke',soak_qualified=passed and soak,
        checks=checks,failed_checks=[name for name,value in checks.items() if not value],
        long_sequence=dict(min_kv_length=args.min_kv_length,completed_count=len(witnesses),witnesses=witnesses[:8]),
        microbatch=dict(configured=result['model_max_batch'],observed_max=observed_max,
                        target_batches=target_batches,target_batch_counts_delta=target_delta),
        memory=memory,smoke_limit='Below 600 seconds is only a functional smoke; insufficient trend duration never qualifies as soak')


def may_skip_oom(error,concurrency,allowed):
    text=str(error).lower()
    cuda_oom='cuda out of memory' in text or 'cuda error: out of memory' in text
    return concurrency>=16 and concurrency in allowed and cuda_oom


def run_tier(args,config,plan,cases,concurrency,pool_type,result=None,checkpoint=lambda:None,clock=time.perf_counter):
    config=dict(config);config['max_batch']=min(concurrency,args.batch)
    result={} if result is None else result
    result.update(concurrency=concurrency,model_max_batch=config['max_batch'],
                scheduling='bounded_waves_with_complete_eos_drain',requested_seconds=args.seconds,
                status='running',requests=[],snapshots=[],warmups=args.warmups,
                effective_runtime_config=config)
    setup_started=clock()
    with pool_type(config,args.gpu,1) as pool:
        pool.prepare_reference('reference',str(args.reference))
        deployment=pool.prepare_deployment(plan)[0]
        result['deployment']={key:deployment[key] for key in ('hardware','sm','requested','request_isolation','compute_backend') if key in deployment}
        result['expected_error_checks']=check_expected_errors(pool)
        result['setup_seconds']=clock()-setup_started
        probe_started=clock()
        result['rng_boundary_checks']=check_runtime_rng(pool,max(cases,key=lambda case:len(case['text'])))
        result['rng_probe_seconds']=clock()-probe_started
        if not result['rng_boundary_checks']['passed']:raise RuntimeError('Actual request RNG boundary checks failed')
        warmup_started=clock()
        for warmup in range(args.warmups):
            run_wave(pool,cases,concurrency,warmup,args.request_timeout_seconds,cancel_every=0,clock=clock)
        result['warmup_seconds']=clock()-warmup_started
        before=compact_stats(pool.stats()[0],include_routes=True);assert_drained(before)
        if args.strict_isolation and (not before['strict_request_isolation'] or before['rng_policy']!='legacy_per_request'):
            raise RuntimeError('Strict request isolation was not active in worker')
        result['before']=before;started=clock();wave=0
        result['timing_scope']='Admission through full EOS/cancel and drain; excludes setup, error/RNG probes and warmup; includes sampling/reporting overhead'
        while clock()-started<args.seconds:
            try:
                records,snapshot=run_wave(pool,cases,concurrency,wave,args.request_timeout_seconds,args.cancel_every,clock=clock)
            except Exception as error:
                result['partial_wave']=getattr(error,'soak_partial_wave',None)
                result['elapsed_s']=clock()-started
                try:result['failure_snapshot']=compact_stats(pool.stats()[0],include_routes=True)
                except Exception as diagnostic_error:result['failure_snapshot_error']=str(diagnostic_error)
                raise
            elapsed=clock()-started
            result['requests'].extend(records);result['snapshots'].append(dict(elapsed_s=elapsed,**snapshot));wave+=1
            checkpoint()
            print(json.dumps(dict(concurrency=concurrency,wave=wave,elapsed_s=elapsed,
                                  complete_eos=sum(row['outcome']=='complete_eos' for row in result['requests']),
                                  cancelled=sum(row['outcome']=='cancelled' for row in result['requests']),
                                  memory=snapshot['memory']),ensure_ascii=False),flush=True)
        after=compact_stats(pool.stats()[0],include_routes=True);assert_drained(after)
        result['after']=after;result['elapsed_s']=clock()-started;result['waves']=wave
        result['counter_delta']={name:after[name]-before[name] for name in COUNTERS if name in before and name in after}
        records=result['requests'];completed=[row for row in records if row['outcome']=='complete_eos']
        result['summary']=dict(admitted=len(records),complete_eos=len(completed),cancelled=len(records)-len(completed),
            completed_requests_per_s=len(completed)/result['elapsed_s'],
            complete_latency_ms=distribution([row['lifetime_ms'] for row in completed]),
            first_chunk_ms=distribution([row['first_chunk_ms'] for row in completed if row['first_chunk_ms'] is not None]),
            max_code_count=max((row['code_count'] for row in completed),default=0),
            max_observed_kv_length=max((row['kv_length'] or 0 for row in records),default=0),
            long_text_completed=sum(row['long_text'] for row in completed),
            reused_request_ids=max(0,wave-1)*concurrency,
            allocated_growth_bytes=after['memory']['cuda_allocated_bytes']-before['memory']['cuda_allocated_bytes'],
            reserved_growth_bytes=after['memory']['cuda_reserved_bytes']-before['memory']['cuda_reserved_bytes'])
        result['pass_gate']=pass_gate(result,args)
        if not result['pass_gate']['passed']:
            raise RuntimeError('Soak pass gate failed: '+', '.join(result['pass_gate']['failed_checks']))
        result['status']='passed' if result['pass_gate']['soak_qualified'] else 'smoke_passed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--deployment',type=workspace_path,default=None,
                        help='Default: matching configs/sm89_trt113_safe_b{1,4,8}.json per concurrency tier')
    parser.add_argument('--reference',type=workspace_path,required=True)
    parser.add_argument('--config',type=workspace_path,default=ROOT/'configs/runtime_reference.yaml')
    parser.add_argument('--corpus',type=workspace_path,default=ROOT/'configs/sm89_quality_256.json')
    parser.add_argument('--concurrency',nargs='+',type=int,choices=(1,4,8,16,32,64),default=[1,4,8,16])
    parser.add_argument('--batch',type=int,choices=(1,4,8),default=8,help='Maximum model microbatch, not request concurrency')
    parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--cancel-every',type=int,default=11,help='Cancel one request after a bounded tick every N admissions; 0 disables')
    parser.add_argument('--request-timeout-seconds',type=float,default=300)
    parser.add_argument('--strict-isolation',action='store_true',help='Explicitly disable shared batch/device RNG; preserve per-request legacy seed mapping')
    parser.add_argument('--min-kv-length',type=int,default=129,help='At least one completed request must show this real Target KV length')
    parser.add_argument('--max-live-growth-mib',type=float,default=64,help='Maximum post-drain CUDA live-memory growth from post-warmup baseline')
    parser.add_argument('--max-rss-growth-mib',type=float,default=256)
    parser.add_argument('--max-live-slope-mib-per-min',type=float,default=1)
    parser.add_argument('--max-rss-slope-mib-per-min',type=float,default=16)
    parser.add_argument('--memory-min-samples',type=int,default=3,help='Minimum samples in latter half for a qualified memory trend')
    parser.add_argument('--memory-min-span-seconds',type=float,default=60)
    parser.add_argument('--allow-oom-skip-concurrency',nargs='*',type=int,choices=(16,32,64),default=[],
                        help='Only CUDA OOM in listed tiers may be skipped; completed_with_allowed_skip is not a full soak pass')
    parser.add_argument('--output',type=workspace_path,required=True)
    args=parser.parse_args()
    if args.seconds<=0 or args.warmups<0 or args.cancel_every<0 or args.request_timeout_seconds<=0:
        parser.error('Invalid duration/warmup/cancellation/timeout value')
    budgets=(args.max_live_growth_mib,args.max_rss_growth_mib,args.max_live_slope_mib_per_min,
             args.max_rss_slope_mib_per_min,args.memory_min_span_seconds)
    if any(not math.isfinite(value) or value<0 for value in budgets):parser.error('Memory budgets must be finite and nonnegative')
    if not math.isfinite(args.seconds) or not math.isfinite(args.request_timeout_seconds):parser.error('Durations must be finite')
    if args.min_kv_length<129:parser.error('Long-sequence evidence requires KV length greater than 128')
    if args.memory_min_samples<3:parser.error('Memory trend requires at least three late samples')
    from acc_infer_clear.runtime.config import load,atomic_json
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.pool import Pool
    sys.path.insert(0,str(ROOT))
    from scripts.trt113_provenance import source_identity,file_record
    config=load(args.config)
    corpus=json.loads(args.corpus.read_text());cases=corpus['cases']
    if not cases or any(not case.get('text') or len(case.get('emotion',[]))!=8 for case in cases):
        raise ValueError('Expected real corpus cases with text, seed and eight-value emotion')
    report=dict(scope='complete_eos_soak_not_numerical_or_quality_audit',status='running',
        hardware_scope='runtime-reported GPU only; no transfer of SM89 results to other architectures',
        memory_claim='Explicit post-warmup live CUDA/RSS growth and late-trend budgets; not an unbounded leak-free guarantee',
        corpus=dict(path=str(args.corpus),sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest()),
        reference=dict(path=str(args.reference),sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest()),
        source=source_identity(),runtime_file=file_record(args.config,'runtime_config'),
        tiers=[])
    try:
        with GPULease(args.gpu) as lease:
            report['gpu_preflight']=dict(physical_gpu=args.gpu,external_memory_mib=lease.initial_memory_mib,
                                        initial_utilization_percent=lease.initial_utilization,shared=lease.shared)
            for concurrency in args.concurrency:
                deployment_path=args.deployment or ROOT/f'configs/sm89_trt113_safe_b{min(concurrency,args.batch)}.json'
                plan,overrides=strict_plan(load_deployment(deployment_path),args.strict_isolation)
                tier=dict(concurrency=concurrency,deployment_path=str(deployment_path),strict_overrides=overrides,
                          deployment_file=file_record(deployment_path,'deployment'))
                report['tiers'].append(tier)
                try:
                    run_tier(args,config,plan,cases,concurrency,Pool,result=tier,
                             checkpoint=lambda:atomic_json(args.output,report))
                except Exception as error:
                    tier['error']=dict(type=type(error).__name__,message=str(error),notes=getattr(error,'__notes__',[]))
                    tier.setdefault('pass_gate',dict(passed=False,soak_qualified=False,failed_checks=['runtime_error']))
                    if may_skip_oom(error,concurrency,args.allow_oom_skip_concurrency):
                        tier['status']='skipped_cuda_oom'
                        atomic_json(args.output,report)
                        continue
                    tier['status']='failed'
                    raise
                atomic_json(args.output,report)
        skipped=[tier['concurrency'] for tier in report['tiers'] if tier['status']=='skipped_cuda_oom']
        report['pass_gate']=dict(passed=not skipped,execution_completed=True,full_soak_pass=not skipped and
            all(tier['pass_gate']['soak_qualified'] for tier in report['tiers']),
            all_requested_tiers_passed=not skipped,skipped_tiers=skipped,
            soak_qualified=not skipped and all(tier['pass_gate']['soak_qualified'] for tier in report['tiers']))
        report['status']='completed_with_allowed_skip' if skipped else ('passed' if report['pass_gate']['soak_qualified'] else 'smoke_passed')
    except Exception as error:
        report['status']='failed';report['error']=dict(type=type(error).__name__,message=str(error))
        report['pass_gate']=dict(passed=False,execution_completed=False,full_soak_pass=False,
                                 soak_qualified=False,all_requested_tiers_passed=False)
        if report['tiers']:report['tiers'][-1]['status']='failed'
        raise
    finally:atomic_json(args.output,report)


if __name__=='__main__':main()
