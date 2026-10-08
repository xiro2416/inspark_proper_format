"""Matched unprofiled migration baselines and sustained real-load board power."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import time

from run_zipvoice_validation import ROOT,invoke,sha

def percentile(values,p):
    values=sorted(values);position=(len(values)-1)*p;low=int(position);high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',type=int,default=[1,2,4,8,16,32,64])
    args=parser.parse_args()
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # CPU evaluation changes CPU delivery load; do not overlap decisive E2E work.
    processes=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
    if any('python' in x and 'evaluate_zipvoice_audio.py --device' in x for x in processes):
        raise RuntimeError('Wait for CPU quality evaluators before matched E2E measurement')
    for attempt in range(10):
        active=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        initial=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True).strip()
        memory,utilization,limit,enforced=map(float,initial.split(','))
        if limit!=400 or enforced!=400:raise RuntimeError(f'GPU1 power cap changed: {initial}')
        # The previously observed 4108 MiB allocation is now visible to NVML
        # under a host PID absent from this PID namespace. It is not our process
        # and must remain untouched. Record it rather than claiming exclusivity.
        known_idle_context=(active=='1820496, 4082' and not Path('/proc/1820496').exists() and memory==4108)
        if (not active or known_idle_context) and memory<=4352 and utilization==0:break
        # Let the preceding diagnostic context and NVML utilization window drain.
        time.sleep(1)
    else:raise RuntimeError(f'GPU1 preflight changed from known idle allocation: {initial}; listed processes: {active}')
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    cases={'short':min(manifest['quality_cases'],key=lambda c:c['total_frames']),
           'primary760':manifest['primary_760'],
           'long':max(manifest['quality_cases'],key=lambda c:c['total_frames'])}
    for batch in args.batches:
        board=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'
        proof=json.loads((board/'003-application.json').read_text())
        assert proof['status']=='application_real_boundary_mixed_passed_quality_performance_pending'
        quality=json.loads((board/'004-quality.json').read_text())
        assert quality['status']=='complete'
        assert quality['input_inventory_sha256']==sha(ROOT/f'outputs/zipvoice-validation/b{batch}/quality-inputs.json')
        summary={'status':'running','batch':batch,'physical_gpu':1,'clock':'prepared CPU conditions including shaping/H2D to all ordered PCM; excludes frontend/init/warmup/capture/WAV writes','shapes':{}}
        summary['preflight']={'memory_mib':memory,'utilization_percent':utilization,'configured_w':limit,'enforced_w':enforced,'listed_compute_processes':active,'scope':'Known idle 4108MiB allocation retained; host PID1820496 is outside this PID namespace. Idle observation is not proof of exclusive ownership; no external process killed/reset. See idle-context-observation.json.'}
        for label,case in cases.items():
            collected={'native':[],'inherited':[]};blocks=[]
            for index,route in enumerate(('native','inherited','inherited','native','inherited','native','native','inherited')):
                folder=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/migration/{label}/{index}-{route}'
                report,_=invoke(batch,route,case,folder,repetitions=5,warmup=3,functional=False)
                times=[x['all_pcm_s'] for x in report['results']]
                assert len(times)==5
                collected[route].extend(times)
                blocks.append({'order':index,'route':route,'median_s':statistics.median(times),'report':str(folder/'report.json')})
            stats={route:{'count':len(v),'median_s':statistics.median(v),'p95_s':percentile(v,.95),'mean_s':statistics.mean(v)} for route,v in collected.items()}
            assert all(x['count']==20 for x in stats.values())
            value={'total_frames':case['total_frames'],'padded_tokens':case['padded_tokens'],'reference_seconds':4,
                   'raw_target_seconds':(case['workload']['target_frames']-1)*256/24000,'routes':stats,'blocks':blocks,
                   'gain_percent':100*(stats['native']['median_s']-stats['inherited']['median_s'])/stats['native']['median_s']}
            summary['shapes'][label]=value
            (board/'005-migration-performance.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({'batch':batch,'case':label,'frames':case['total_frames'],'routes':stats,'gain_percent':value['gain_percent']}),flush=True)
        folder=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/migration/sustained760'
        report,_=invoke(batch,'inherited',cases['primary760'],folder,extra=['--minimum-seconds','32'],warmup=5,functional=False)
        results=report['results'];seconds=(results[-1]['request_done_ns']-results[0]['request_start_ns'])/1e9
        assert seconds>=30
        summary['sustained_power']={'wall_seconds':seconds,'requests':len(results),'report':str(folder/'report.json'),'telemetry':report['power_telemetry'],'configured_enforced_w':report['power_limits_after_w']}
        summary['status']='matched_migration_measurements_complete_review_pending'
        (board/'005-migration-performance.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
