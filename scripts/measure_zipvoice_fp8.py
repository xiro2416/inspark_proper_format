"""Matched unprofiled E2E measurements and explicit per-batch PCM selection."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,write,private_report
from validate_zipvoice_fp8 import invoke


def statistics_for(report):
    import numpy as np
    times=[x['all_pcm_s']*1000 for x in report['results']]
    batch=report['shape'][0];seconds=(report['workload']['target_frames']-1)*256/24000
    return dict(median_ms=statistics.median(times),p95_ms=float(np.percentile(times,95)),
                throughput_requests_s=batch/(statistics.mean(times)/1000),
                throughput_audio_seconds_s=batch*seconds/(statistics.mean(times)/1000),
                samples=len(times),power=report['power_telemetry'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=list(BATCHES))
    p.add_argument('--variant',choices=['native','attention','attentiongeo'],default='native')
    p.add_argument('--phase',choices=['migration','optimization'],default='migration')
    p.add_argument('--repetitions',type=int,default=20)
    args=p.parse_args()
    if args.repetitions<20:raise ValueError('Decisive measurements require at least20 samples')
    data=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text())
    primary=data['primary_760']
    if primary is None:raise RuntimeError('No natural exact760 case; do not force original duration')
    short=min(data['quality'],key=lambda x:x['total_frames'])
    long=max(data['quality'],key=lambda x:x['total_frames'])
    from inspark_infer.runtime.device import GPULease
    if args.phase=='optimization':
        for b in BATCHES:
            accepted=json.loads((private_report(b,'006-migration')).read_text())
            if not accepted['migration_accepted']:raise RuntimeError('All seven migrations must pass before optimization')
    with GPULease(3):
        for batch in args.batches:
            policy=dict(chunk={1:16,2:16,4:1,8:1,16:4,32:6,64:16}[batch],workers=4)
            result=dict(batch=batch,phase=args.phase,status='running',primary_frames=760,full_text_encoder=True,
                        reference_boundary='Prepared CPU condition shaping/H2D through all ordered PCM',
                        exclusions=['frontend','initialization','warmup','capture','WAV writes'])
            if args.phase=='migration':
                comparisons=[]
                for order,mode in enumerate(['direct','model-graph','model-graph','direct']):
                    out=ROOT/f'outputs/fp8/b{batch}/migration-measure/{order}-{mode}'
                    report=invoke(batch,primary,out,variant=args.variant,execution=mode,repetitions=args.repetitions,
                                  functional=False,full_text=True,**policy)
                    comparisons.append(dict(mode=mode,report=str(out/'report.json'),**statistics_for(report)))
                result['matched_application_comparison']=comparisons
                result['cases']={}
                for label,case in [('short',short),('primary',primary),('long',long)]:
                    out=ROOT/f'outputs/fp8/b{batch}/migration-measure/{label}'
                    report=invoke(batch,case,out,variant=args.variant,repetitions=args.repetitions,functional=False,
                                  full_text=True,minimum_seconds=30 if label=='primary' else 0,**policy)
                    result['cases'][label]=dict(frames=case['total_frames'],report=str(out/'report.json'),**statistics_for(report))
                application=json.loads((private_report(batch,'003-application')).read_text())
                quality_path=private_report(batch,'005-quality')
                quality=json.loads(quality_path.read_text()) if quality_path.is_file() else {'status':'pending'}
                ready=application['status']=='application_and_boundaries_passed_quality_audit_pending' and quality['status']=='paired_quality_metrics_complete'
                result.update(status='migration_accepted' if ready else 'migration_performance_measured_quality_pending',
                              migration_accepted=ready,pcm_policy=policy,
                              application_status=application['status'],quality_status=quality['status'])
                write(private_report(batch,'006-migration'),result)
            else:
                # Existing source policy is the legal comparator. First screen
                # delivery variations, then confirm worthwhile winners in ABBA.
                candidates=[policy]
                for chunk,workers in [(1,1),(4,4),(8,4),(16,4),(16,8),(32,4)]:
                    candidate=dict(chunk=chunk,workers=workers)
                    if candidate not in candidates:candidates.append(candidate)
                screens=[]
                for i,candidate in enumerate(candidates):
                    out=ROOT/f'outputs/fp8/b{batch}/optimization-delivery-{args.variant}/screen-{i}'
                    report=invoke(batch,primary,out,variant=args.variant,repetitions=args.repetitions,functional=False,full_text=True,**candidate)
                    screens.append(dict(policy=candidate,report=str(out/'report.json'),**statistics_for(report)))
                winner=min(screens,key=lambda x:x['median_ms'])
                candidate=winner['policy'];confirmations=[]
                for i,c in enumerate([policy,candidate,candidate,policy]):
                    out=ROOT/f'outputs/fp8/b{batch}/optimization-delivery-{args.variant}/confirm-{i}'
                    report=invoke(batch,primary,out,variant=args.variant,repetitions=args.repetitions,functional=False,full_text=True,**c)
                    confirmations.append(dict(policy=c,report=str(out/'report.json'),**statistics_for(report)))
                before=statistics.median([x['median_ms'] for x in [confirmations[0],confirmations[3]]])
                after=statistics.median([x['median_ms'] for x in [confirmations[1],confirmations[2]]])
                gain=100*(before-after)/before
                chosen=candidate if gain>0 else policy
                result.update(status='delivery_review_complete_compute_review_pending',screens=screens,
                              confirmations=confirmations,pcm_policy=chosen,matched_gain_percent=gain,
                              compute_review_complete=False,optimization_review_complete=False)
                write(private_report(batch,'007-delivery-'+args.variant),result)
            print(json.dumps({'event':'measurement_phase_complete','batch':batch,'phase':args.phase}),flush=True)


if __name__=='__main__':main()
