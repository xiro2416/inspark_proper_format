"""Formal matched, unprofiled E2E and sustained power for a validated candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import time

from run_zipvoice_validation import ROOT,invoke,sha,inventory
from measure_zipvoice_baselines import percentile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--route',required=True)
    parser.add_argument('--control',choices=('native','inherited','normtf32k16'),required=True)
    parser.add_argument('--application',choices=('a1007','a1007_delivery','a1007_graph','a1007_delivery_graph'))
    parser.add_argument('--control-application',choices=('a1007','a1007_delivery','a1007_graph','a1007_delivery_graph'))
    parser.add_argument('--pcm-chunk',type=int)
    parser.add_argument('--pcm-workers',type=int)
    parser.add_argument('--current-best-review',type=Path)
    args=parser.parse_args();batch=args.batch;candidate=args.route
    extra=[]
    if args.pcm_chunk is not None:extra+=['--pcm-chunk',str(args.pcm_chunk)]
    if args.pcm_workers is not None:extra+=['--pcm-workers',str(args.pcm_workers)]
    if args.current_best_review:
        current=json.loads(args.current_best_review.read_text())['batches'][str(batch)]
        assert current['route']==args.control and current['application']==args.control_application
        assert current['pcm_policy']=={'chunk':args.pcm_chunk,'workers':args.pcm_workers}
    def call(route,case,folder,**kwargs):
        app=args.application if route==candidate else args.control_application
        return invoke(batch,route,case,folder,application=app,extra=extra,**kwargs)
    history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'
    application=json.loads((history/f'018-{candidate}-application.json').read_text())
    assert application['status']=='variant_full_application_mapping_passed_quality_pending'
    quality=json.loads((history/f'019-{candidate}-quality.json').read_text())
    assert quality['status']=='complete'
    inputs=ROOT/f'outputs/zipvoice-validation/b{batch}/{candidate}-validation/quality-inputs.json'
    assert quality['input_inventory_sha256']==sha(inputs)
    assert not any('evaluate_zipvoice_audio.py --device' in line for line in subprocess.check_output(['ps','-eo','args='],text=True).splitlines()),'Wait for CPU quality jobs before decisive E2E'
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for attempt in range(10):
        row=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu,power.limit,enforced.power.limit','--format=csv,noheader,nounits'],text=True).strip()
        memory,util,cap,enforced=map(float,row.split(','))
        processes=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
        known=processes=='1820496, 4082' and not Path('/proc/1820496').exists() and memory==4108
        assert cap==enforced==400
        if util==0 and memory<=4352 and (not processes or known):break
        time.sleep(1)
    else:raise RuntimeError(f'GPU1 occupancy differs from known idle state: {row}, {processes}')
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    cases={'short':min(manifest['quality_cases'],key=lambda x:x['total_frames']),
           'primary760':manifest['primary_760'],
           'long':max(manifest['quality_cases'],key=lambda x:x['total_frames'])}
    identities={route:inventory(batch,route,cases['primary760']['workload']) for route in (args.control,candidate)}
    assert identities[candidate]['engines']['fm']['sha256']==application['engine_sha256']
    evidence=history/f'020-{candidate}-performance.json'
    result={'status':'running','batch':batch,'candidate':candidate,'current_best_control':args.control,
            'applications':{'control':args.control_application,'candidate':args.application},
            'pcm_policy':{'chunk':args.pcm_chunk,'workers':args.pcm_workers},
            'current_best_review_sha256':sha(args.current_best_review) if args.current_best_review else None,
            'clock':'Prepared CPU conditions including shaping/H2D to all ordered PCM; excludes init/warmup/capture/WAV writes',
            'engine_identities':{route:{'engine_sha256':value['engines']['fm']['sha256'],'plugin_package':value['plugin_package']} for route,value in identities.items()},
            'application_evidence_sha256':sha(history/f'018-{candidate}-application.json'),
            'quality_evidence_sha256':sha(history/f'019-{candidate}-quality.json'),
            'preflight':{'memory_mib':memory,'gpu_util_percent':util,'processes':processes,
                         'power_cap_w':cap,'note':'Known idle allocation retained; not exclusive device ownership.'},'shapes':{}}
    for label,case in cases.items():
        values={args.control:[],candidate:[]};blocks=[]
        order=(args.control,candidate,candidate,args.control,candidate,args.control,args.control,candidate)
        for index,route in enumerate(order):
            folder=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/{candidate}-formal/{label}/{index}-{route}'
            report,_=call(route,case,folder,warmup=3,repetitions=5,functional=False)
            times=[x['all_pcm_s'] for x in report['results']];assert len(times)==5
            values[route]+=times;blocks.append({'route':route,'median_s':statistics.median(times),'report':str(folder/'report.json')})
        medians={route:statistics.median(v) for route,v in values.items()}
        result['shapes'][label]={'frames':case['total_frames'],'padded_tokens':case['padded_tokens'],
                                'statistics':{route:{'count':len(v),'median_s':statistics.median(v),'p95_s':percentile(v,.95)} for route,v in values.items()},
                                'gain_percent':100*(medians[args.control]-medians[candidate])/medians[args.control],
                                'blocks':blocks}
        evidence.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'batch':batch,'candidate':candidate,'shape':label,**{k:v for k,v in result['shapes'][label].items() if k!='blocks'}}),flush=True)
    dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/{candidate}-formal/sustained760'
    report,_=invoke(batch,candidate,cases['primary760'],dest,warmup=5,application=args.application,extra=[*extra,'--minimum-seconds','32'],functional=False)
    values=report['results'];duration=(values[-1]['request_done_ns']-values[0]['request_start_ns'])/1e9
    assert duration>=30
    result.update(status='formal_candidate_e2e_quality_power_complete_review_pending',
                  sustained_power={'duration_s':duration,'requests':len(values),'report':str(dest/'report.json'),
                                   'telemetry':report['power_telemetry'],'configured_enforced_w':report['power_limits_after_w']})
    evidence.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
