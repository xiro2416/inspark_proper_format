#!/usr/bin/env python3
"""Validate full-batch first-chunk TRT routing, not full EOS or numerical parity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
TEXT='他正在整理文件。'
COUNTERS=('target_calls','device_target_steps','draft_backbone_calls','device_draft_steps',
          'native_target_steps','native_draft_steps','device_round_fallbacks')


class AuditError(ValueError):
    pass


def counter(mapping,name,where):
    if name not in mapping:raise AuditError(f'Missing metric: {where}.{name}')
    value=mapping[name]
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise AuditError(f'Invalid nonnegative integer metric: {where}.{name}={value!r}')
    return value


def difference(before,after,name,where):
    value=counter(after,name,where+'.after')-counter(before,name,where+'.before')
    if value<0:raise AuditError(f'Counter decreased: {where}.{name}')
    return value


def route_inventory(snapshot,component):
    acoustic=snapshot.get('acoustic_routes')
    if not isinstance(acoustic,dict) or component not in acoustic:
        raise AuditError(f'Missing metric: acoustic_routes.{component}')
    rows=acoustic[component].get('graph_routes')
    if not isinstance(rows,list):raise AuditError(f'Missing route metrics: {component}.graph_routes')
    inventory={}
    for row in rows:
        route=row.get('route')
        if not isinstance(route,dict) or route.get('component')!=component:
            raise AuditError(f'Invalid route identity: {component}')
        key=json.dumps(route,sort_keys=True)
        if key in inventory:raise AuditError(f'Duplicate route metrics: {component}')
        for field in ('prepare','direct','replay','fallback'):counter(row,field,component)
        inventory[key]=row
    return inventory,acoustic[component]


def acoustic_window(before,after,component):
    previous,old_wrapper=route_inventory(before,component)
    current,new_wrapper=route_inventory(after,component)
    if set(previous)-set(current):raise AuditError(f'Route metrics disappeared: {component}')
    counts=dict(total=0,native=0,non_trt=0,fallback=0,direct=0,replay=0)
    rows=[]
    for key,row in current.items():
        old=previous.get(key,dict(prepare=0,direct=0,replay=0,fallback=0))
        delta={name:difference(old,row,name,component) for name in ('prepare','direct','replay','fallback')}
        if delta['prepare']:raise AuditError(f'Online graph capture inside measurement: {component}')
        executions=delta['direct']+delta['replay']
        if not executions and not delta['fallback']:continue
        route=row['route']
        kind=route.get('kind');backend=route.get('backend')
        if kind not in ('tensorrt','eager') or backend not in ('tensorrt113','eager'):
            raise AuditError(f'Unclassified runtime route: {component}: {route}')
        native=kind=='tensorrt' and backend=='tensorrt113'
        if (kind=='tensorrt')!=(backend=='tensorrt113'):
            raise AuditError(f'Contradictory runtime route: {component}')
        if native and (not route.get('plan') or not (route.get('sha256') or route.get('engine_sha256'))):
            raise AuditError(f'Missing native plan/hash identity: {component}')
        if delta['fallback']>executions:raise AuditError(f'Invalid fallback count: {component}')
        counts['total']+=executions
        counts['native' if native else 'non_trt']+=executions
        for name in ('fallback','direct','replay'):counts[name]+=delta[name]
        rows.append(dict(route=route,**delta))
    # Direct wrapper calls outside the graph bank have no per-call route record.
    # Fail closed instead of inferring an engine hit from the configured backend.
    for value,label in ((old_wrapper,'before'),(new_wrapper,'after')):
        if not isinstance(value.get('wrapper_direct'),dict):
            raise AuditError(f'Missing metric: {component}.{label}.wrapper_direct')
    direct_wrapper={name:difference(old_wrapper['wrapper_direct'],new_wrapper['wrapper_direct'],name,
                                   component+'.wrapper_direct') for name in ('calls','fallbacks')}
    if sum(direct_wrapper.values())!=counts['direct']:
        raise AuditError(f'Unaccounted direct backend calls: {component}')
    counts['routes']=rows
    counts['wrapper_direct']=direct_wrapper
    return counts


def validate_window(before,after):
    delta={name:difference(before,after,name,'runtime') for name in COUNTERS}
    old_buckets=before.get('draft_execution_buckets',{})
    new_buckets=after.get('draft_execution_buckets',{})
    if not isinstance(old_buckets,dict) or not isinstance(new_buckets,dict):
        raise AuditError('Draft execution bucket metrics are missing')
    draft_buckets={name:difference({name:old_buckets.get(name,0)},
                                   {name:new_buckets.get(name,0)},name,'draft_execution_buckets')
                   for name in sorted(set(old_buckets)|set(new_buckets))}
    if sum(draft_buckets.values())!=delta['draft_backbone_calls']:
        raise AuditError('Draft execution buckets do not account for every backbone call')
    target_total=delta['device_target_steps']+delta['target_calls']
    draft_total=delta['device_draft_steps']+delta['draft_backbone_calls']
    components={
        'target':dict(total=target_total,native=delta['native_target_steps'],
                      non_trt=target_total-delta['native_target_steps']),
        'draft':dict(total=draft_total,native=delta['native_draft_steps'],
                     non_trt=draft_total-delta['native_draft_steps']),
        'cfm':acoustic_window(before,after,'cfm'),
        'vocoder':acoustic_window(before,after,'vocoder'),
    }
    errors=[]
    if delta['device_round_fallbacks']:errors.append('device-round fallback occurred')
    for name,counts in components.items():
        if counts['native']<=0:errors.append(f'{name}: no native execution observed')
        if counts['total']!=counts['native']:errors.append(f'{name}: native/total execution counts differ')
        if counts.get('fallback',0):errors.append(f'{name}: runtime fallback occurred')
    return dict(passed=not errors,errors=errors,counter_delta=delta,
                draft_execution_buckets=draft_buckets,components=components)


def request_record(ident,event,state,arrival):
    chunk=event['chunk']
    samples=chunk['sample_end']-chunk['sample_start']
    if chunk['index']!=0 or chunk['sample_start']!=0:
        raise AuditError(f'{ident}: expected first chunk starting at sample zero')
    if samples!=len(chunk['pcm']) or samples<=0 or samples%256:
        raise AuditError(f'{ident}: invalid PCM sample count {samples}')
    frames=samples//256
    if frames!=44 and not (bool(chunk['eos']) and 0<frames<44):
        raise AuditError(f'{ident}: first chunk has {frames} frames without an early EOS')
    if state.get('error'):raise AuditError(f'{ident}: {state["error"]}')
    if len(state['chunks'])!=1:raise AuditError(f'{ident}: measurement advanced beyond the first chunk')
    latency=(event['received']-arrival)*1000
    if latency<0:raise AuditError(f'{ident}: invalid receive timestamp')
    return dict(id=ident,first_chunk_ms=latency,sample_count=samples,frame_count=frames,
                eos=bool(chunk['eos']),complete=bool(state['complete']),chunk_count=len(state['chunks']),
                code_count=len(state['codes']),rounds=state.get('rounds'),
                accepted_counts=state.get('accepted'),kv_head_length=state.get('kv_head_lengths'),
                cfm_batch=chunk['cfm_batch'],vocoder_batch=chunk['vocoder_batch'])


def percentile(values,q):
    values=sorted(values);position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def distribution(values):
    return dict(n=len(values),min=min(values),mean=statistics.fmean(values),
                p50=percentile(values,.5),p95=percentile(values,.95),p99=percentile(values,.99),max=max(values))


def run_wave(pool,batch,prefix):
    before=pool.stats()[0]
    identifiers=[f'{prefix}-{index}' for index in range(batch)]
    arrivals={};events={};created=[]
    started=time.perf_counter()
    try:
        for index,ident in enumerate(identifiers):
            arrivals[ident]=time.perf_counter()
            pool.create_session(ident,'reference',index,arrival=arrivals[ident]);created.append(ident)
            pool.push_text(ident,TEXT);pool.finish_input(ident)
        pending=set(identifiers)
        while pending:
            emitted=pool.run_ready()
            if not emitted:raise AuditError(f'Scheduler stalled with {len(pending)} first chunks pending')
            for event in emitted:
                ident=event['request_id']
                if ident not in pending:raise AuditError(f'Unexpected or duplicate first-chunk event: {ident}')
                events[ident]=event;pending.remove(ident)
        completed=max(event['received'] for event in events.values())
        after=pool.stats()[0]
        records=[request_record(ident,events[ident],pool.result(ident),arrivals[ident]) for ident in identifiers]
        audit=validate_window(before,after)
        for component in ('cfm','vocoder'):
            for row in audit['components'][component]['routes']:
                if row['route'].get('batch')!=batch:
                    audit['errors'].append(f'{component}: observed route batch differs from declared batch {batch}')
        for record in records:
            if record['cfm_batch']!=batch or record['vocoder_batch']!=batch:
                audit['errors'].append(f'{record["id"]}: acoustic execution left the declared full-batch scope')
        audit['passed']=not audit['errors']
        return dict(requests=records,all_first_chunks_ms=(completed-started)*1000,
                    requests_per_s=batch/(completed-started),audit=audit)
    finally:
        for ident in created:pool.cancel(ident)


def workspace_path(value):
    path=Path(value).resolve()
    if not path.is_relative_to('/workspace'):raise argparse.ArgumentTypeError('Paths must remain under /workspace')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--batch',type=int,choices=(1,4,8),required=True)
    parser.add_argument('--deployment',type=workspace_path,required=True)
    parser.add_argument('--reference',type=workspace_path,required=True)
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=workspace_path,required=True)
    parser.add_argument('--config',type=workspace_path,default=ROOT/'configs/common/runtime.yaml')
    args=parser.parse_args()
    if args.gpu<0 or args.warmups<0 or args.repeats<1:parser.error('gpu/warmups must be nonnegative; repeats must be positive')
    from inspark_infer.runtime.config import atomic_json, load
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.pool import Pool
    report=dict(schema=1,status='running',scope='first_chunk_only',full_eos_verified=False,
        numerical_parity_verified=False,concurrency_verified=[args.batch],soak_verified=False,
        full_trt_scope=['target_verify','draft_backbone','cfm_solver','vocoder'],
        excluded_from_full_trt_claim=['prefill','proposal_rnn','context_append','conditioning','latent','streaming_tail'],
        gpu=args.gpu,batch=args.batch,workers=1,text=TEXT,request_seed_policy='row index, unchanged across waves',
        warmups=args.warmups,repeats=args.repeats,reference=str(args.reference),
        deployment=str(args.deployment),timing='per-request enqueue to host-received first PCM; includes admission and copies',
        warmup_results=[],runs=[])
    try:
        report['deployment_sha256']=hashlib.sha256(args.deployment.read_bytes()).hexdigest()
        report['reference_sha256']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
        report['source_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        report['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
        from trt113_provenance import source_identity
        report['source_identity']=source_identity(ROOT)
        config=load(args.config);config['max_batch']=args.batch
        deployment=load_deployment(args.deployment)
        with GPULease(args.gpu) as lease,Pool(config,args.gpu,1) as pool:
            report['device_preflight']=dict(shared=lease.shared,existing_memory_mib=lease.initial_memory_mib,
                existing_utilization_pct=lease.initial_utilization,
                limitation='External allocations preserved; not a dedicated-device performance guarantee')
            pool.prepare_reference('reference',str(args.reference))
            manifest=pool.prepare_deployment(deployment)[0]
            report['profile']={key:manifest.get(key) for key in ('resolved_precision','resolved_rnn_precision','sm')}
            report['profile']['status']=manifest['requested']['status']
            report['ar_artifacts']={component:manifest.get('tensorrt113_'+component+'_full',{}).get('artifacts',{})
                                    for component in ('target','draft')}
            if report['profile']['sm']<80:raise AuditError('This route requires SM80 or newer')
            hardware=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader,nounits'],text=True)
            report['hardware']=dict(zip(('name','uuid','driver_version','memory_mib'),
                                        (value.strip() for value in next(csv.reader([hardware.strip()])))))
            for index in range(args.warmups):
                row=run_wave(pool,args.batch,f'warmup-{index}')
                report['warmup_results'].append(dict(audit=row['audit'],all_first_chunks_ms=row['all_first_chunks_ms']))
                if not row['audit']['passed']:raise AuditError(f'Warmup {index} failed: {row["audit"]["errors"]}')
            for index in range(args.repeats):
                row=run_wave(pool,args.batch,f'run-{index}');row['repeat']=index
                report['runs'].append(row);atomic_json(args.output,report)
                if not row['audit']['passed']:raise AuditError(f'Measurement {index} failed: {row["audit"]["errors"]}')
        report['summary']=dict(all_first_chunks_ms=distribution([row['all_first_chunks_ms'] for row in report['runs']]),
            request_first_chunk_ms=distribution([request['first_chunk_ms'] for row in report['runs'] for request in row['requests']]),
            requests_per_s=distribution([row['requests_per_s'] for row in report['runs']]))
        report['status']='passed'
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}'
    atomic_json(args.output,report)
    print(json.dumps(dict(status=report['status'],output=str(args.output),scope=report['scope'],
                          completed_repeats=len(report['runs']),error=report.get('error')),ensure_ascii=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':sys.exit(main())
