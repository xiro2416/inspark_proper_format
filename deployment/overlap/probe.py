"""Same work: one-bank serial versus independent banks on one CUDA context."""
import argparse,concurrent.futures,hashlib,json,os,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=(4,8,16,32),default=16);p.add_argument('--waves',type=int,default=10);p.add_argument('--warmups',type=int,default=3);p.add_argument('--trace',action='store_true');p.add_argument('--trace-mode',default='two_bank_concurrent');p.add_argument('--single-only',action='store_true');p.add_argument('--paired',action='store_true');p.add_argument('--power-seconds',type=float,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('GPU1 only; source deployment/multibatch/env.sh')
    from inspark_infer.runtime.device import GPULease,select_gpu
    select_gpu(1)
    import torch
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.pool import _engine_stats
    from benchmarks.unified_first_chunk import load_manifest,run_wave,wave_cases,distribution,counter_delta
    from benchmarks.benchmark_unified_first_chunk import prepare_references
    from benchmarks.board_power import BoardSampler
    manifest=load_manifest(ROOT/'deployment/history/validation-manifest.json');cases=manifest['splits']['evaluation'];total=a.batch if a.single_only else 2*a.batch
    if len(cases)<total:raise RuntimeError('Need a full independent case set')
    plan=json.loads((ROOT/f'configs/hardware/sm89/indextts/int8_b{a.batch}_selected.json').read_text());plan=dict(plan,batch_text_dedup=False)
    report=dict(batch_per_bank=a.batch,total_requests_per_wave=total,physical_gpu=1,model_banks=[],modes={},inputs_manifest_sha256=manifest['manifest_sha256'],deployment=plan,
        scope='First-chunk full-work probe, original recipe/backend; distinct model/engine/graph/KV/RNG/output banks share one CUDA primary context, independent model streams; not a deployment change',trace=a.trace)
    engines=[];pool=None;sampler=None
    def save():a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    def create():
        c=dict(load(str(ROOT/'local_assets/runtime/runtime_fp32_b1.yaml')),max_batch=a.batch,precision_batches=[a.batch]);e=Engine(c);engines.append(e)
        prepare_references(e,manifest);att=e.prepare_deployment(plan);e.configure_profiling(False,a.trace)
        report['model_banks'].append(dict(stream=e.model.stream.cuda_stream,attestation=att));save()
        return e
    def lane(i,items,label):
        e=engines[i]
        if a.trace:torch.cuda.nvtx.range_push('BANK'+str(i)+'/'+label)
        try:
            value=run_wave(e,items,label,details=True,admission_mode='batch')
            value['native_thread_id']=threading.get_native_id();return value
        finally:
            if a.trace:torch.cuda.nvtx.range_pop()
    references={}
    def execute(mode,index,warm=False):
        items=wave_cases(cases,total,index);start=time.perf_counter()
        if mode in ('single_bank','one_bank_serial','resident_one_bank_serial'):
            groups=[lane(0,items,f'{mode}-{index}')]
        elif mode=='two_bank_serial':
            groups=[lane(0,items[:a.batch],f'{mode}-{index}-0'),lane(1,items[a.batch:],f'{mode}-{index}-1')]
        else:
            ready=threading.Barrier(2)
            def task(i):ready.wait();return lane(i,items[i*a.batch:(i+1)*a.batch],f'{mode}-{index}-{i}')
            futures=[pool.submit(task,i) for i in range(2)];groups=[f.result() for f in futures]
        end=max(g['measurement_end'] for g in groups);rows=[r for g in groups for r in g['rows']]
        fingerprints={r['case_id']:dict(pcm=r['pcm_sha256'],codes=r['code_sha256'],accepted=r['accepted'],rounds=r['rounds'],seed=r['seed']) for r in rows}
        if len(rows)!=total or len(fingerprints)!=total:raise RuntimeError('Incomplete or duplicated wave')
        if any(r['cfm_batch']!=a.batch or r['vocoder_batch']!=a.batch for r in rows):raise RuntimeError('Unexpected acoustic shape')
        if mode in ('one_bank_serial','single_bank'):references[index]=fingerprints
        elif fingerprints!=references[index]:raise RuntimeError('Cross-bank output/state mismatch: '+mode)
        row=dict(wave=index,group_ms=1000*(end-start),rows=rows,thread_ids=[g['native_thread_id'] for g in groups],fingerprints=fingerprints,
            host_ranges=[dict(start=g['measurement_start'],end=g['measurement_end']) for g in groups])
        if not warm:report['modes'][mode]['waves'].append(row)
        return row
    def power_measure(mode):
        def wave(index):
            items=wave_cases(cases,total,index%a.waves);started=time.perf_counter()
            def run(i,part):return run_wave(engines[i],part,f'power-{mode}-{index}-{i}',details=False,admission_mode='batch')
            if mode in ('single_bank','resident_one_bank_serial'):
                groups=[run(0,items)]
            else:
                ready=threading.Barrier(2)
                def task(i):ready.wait();return run(i,items[i*a.batch:(i+1)*a.batch])
                futures=[pool.submit(task,i) for i in range(2)];groups=[f.result() for f in futures]
            rows=[r for g in groups for r in g['rows']]
            if len(rows)!=total or any(r['cfm_batch']!=a.batch or r['vocoder_batch']!=a.batch for r in rows):raise RuntimeError('Power wave incomplete or wrong shape')
            return 1000*(max(g['measurement_end'] for g in groups)-started)
        for i in range(a.warmups):wave(i)
        torch.cuda.synchronize();before=[_engine_stats(e) for e in engines]
        started=time.perf_counter();latencies=[]
        while time.perf_counter()-started<a.power_seconds:latencies.append(wave(len(latencies)))
        ended=time.perf_counter();after=[_engine_stats(e) for e in engines]
        counters=[counter_delta(x,y) for x,y in zip(before,after)]
        if any(c[k] for c in counters for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Power fallback')
        if any(sum(c['head_graph_hits'][k] for c in counters)!=(1 if a.single_only else 2)*len(latencies) for k in ('cfm','vocoder')):raise RuntimeError('Power missing graph work')
        if any(s['sessions'] or s['active_rows'] or s['error_sessions'] for s in after):raise RuntimeError('Power leaked request state')
        sensor=sampler.window(started,ended)
        if sensor['sensor_samples']<20:raise RuntimeError('Insufficient power samples')
        report.setdefault('sustained',{})[mode]=dict(seconds=ended-started,waves=len(latencies),requests=len(latencies)*total,group_ms=distribution(latencies),sensor=sensor,counters=counters,
            sensor_samples=[list(x) for x in sampler.samples if started<=x[0]<=ended],scope='Back-to-back complete 32-request first-chunk waves plus cancellation/dispatch; no loading/warmup/hash work. Whole-board instantaneous power sampled at 20ms.')
        save();print(json.dumps(dict(power_mode=mode,**sensor)),flush=True)
    def measure(mode):
        report['modes'][mode]=dict(waves=[])
        for i in range(a.warmups):execute(mode,i,warm=True)
        torch.cuda.synchronize()
        before=[_engine_stats(e) for e in engines]
        if a.trace and mode==a.trace_mode:torch.cuda.profiler.start()
        try:
            for i in range(a.waves):
                r=execute(mode,i)
                print(json.dumps(dict(mode=mode,wave=i,group_ms=r['group_ms'])),flush=True)
        finally:
            if a.trace and mode==a.trace_mode:torch.cuda.profiler.stop()
        after=[_engine_stats(e) for e in engines]
        counters=[counter_delta(x,y) for x,y in zip(before,after)]
        for c in counters:
            if any(c[k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Unexpected fallback')
        if any(sum(c['head_graph_hits'][k] for c in counters)!=(1 if a.single_only else 2)*a.waves for k in ('cfm','vocoder')):raise RuntimeError('Missing graph work')
        entry=report['modes'][mode];entry.update(group_ms=distribution(r['group_ms'] for r in entry['waves']),counters=counters,bank_memory=[s['memory'] for s in after],correctness='Same-seed full PCM/code/accepted-sequence identities match serial reference; all sessions cancelled')
        if any(s['sessions'] or s['active_rows'] or s['error_sessions'] for s in after):raise RuntimeError('Leaked request state')
        save()
        if a.power_seconds and mode in ('single_bank','resident_one_bank_serial','two_bank_concurrent'):power_measure(mode)
    with GPULease(1) as lease:
        report['initial_board_memory_mib']=lease.initial_memory_mib
        sampler=BoardSampler(1);sampler.start()
        try:
            create();measure('single_bank' if a.single_only else 'one_bank_serial')
            if not a.single_only:
                create();pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
                measure('resident_one_bank_serial');measure('two_bank_serial');measure('two_bank_concurrent')
            if a.paired and not a.single_only:
                paired={'resident_one_bank_serial':[], 'two_bank_concurrent':[]}
                before=[_engine_stats(e) for e in engines]
                for i in range(a.waves):
                    order=list(paired) if i%2==0 else list(reversed(paired))
                    for mode in order:
                        paired[mode].append(execute(mode,i,warm=True))
                after=[_engine_stats(e) for e in engines]
                deltas=[counter_delta(x,y) for x,y in zip(before,after)]
                if any(c[k] for c in deltas for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):raise RuntimeError('Paired fallback')
                if any(sum(c['head_graph_hits'][k] for c in deltas)!=4*a.waves for k in ('cfm','vocoder')):raise RuntimeError('Paired missing work')
                if any(s['sessions'] or s['active_rows'] or s['error_sessions'] for s in after):raise RuntimeError('Paired leaked state')
                report['paired']={k:dict(waves=v,group_ms=distribution(r['group_ms'] for r in v)) for k,v in paired.items()}
                report['paired_counters']=deltas
            report['status']='probe_validated';save()
        except Exception as error:
            report.update(status='incomplete',error=str(error));save();raise
        finally:
            if pool:pool.shutdown(wait=True)
            for e in reversed(engines):e.close()
            sampler.stop();report['board_lifecycle']=dict(memory_mib=distribution(x[1] for x in sampler.samples),power_w=distribution(x[2] for x in sampler.samples),scope='whole setup/warmup/probe/teardown; diagnostic only, not sustained-serving power');save()


if __name__=='__main__':main()
