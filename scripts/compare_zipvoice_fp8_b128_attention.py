"""Matched ABBA target comparison, full candidate checks and independent lengths."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import BATCHES,write,private_report
from validate_zipvoice_fp8_b128 import invoke,boundaries
from measure_zipvoice_fp8_b128 import statistics_for


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=[128])
    p.add_argument('--baseline',choices=['native','attention'],default='native')
    p.add_argument('--variant',choices=['attention','attentiongeo','attentionres','attentionff','attentionprotected'],default='attention')
    args=p.parse_args();candidate=args.variant;baseline=args.baseline;data=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text());primary=data['primary_760']
    short=min(data['quality'],key=lambda c:c['total_frames']);long=max(data['quality'],key=lambda c:c['total_frames'])
    from inspark_infer.runtime.device import GPULease
    with GPULease(3):
        for batch in args.batches:
            rows=[]
            for i,variant in enumerate([baseline,candidate,candidate,baseline]):
                out=ROOT/f'outputs/fp8/b{batch}/{candidate}-matched/{i}-{variant}'
                r=invoke(batch,primary,out,variant=variant,repetitions=20,functional=False,full_text=True)
                rows.append(dict(variant=variant,report=str(out/'report.json'),**statistics_for(r)))
                print(json.dumps(dict(event='matched',batch=batch,variant=variant,median_ms=rows[-1]['median_ms'])),flush=True)
            before=statistics.median(r['median_ms'] for r in rows if r['variant']==baseline);after=statistics.median(r['median_ms'] for r in rows if r['variant']==candidate)
            result=dict(status='matched_performance_quality_pending',batch=batch,baseline_variant=baseline,rows=rows,baseline_ms=before,candidate_ms=after,gain_percent=100*(before-after)/before)
            write(private_report(batch,'008-'+candidate+'-comparison'),result)
            if after>=before:
                result.update(status=f'candidate_regressed_{baseline}_retained',candidate_retained=False)
                write(private_report(batch,'008-'+candidate+'-comparison'),result);continue
            for i,case in enumerate(data['quality']):
                invoke(batch,case,ROOT/f'outputs/fp8/b{batch}/quality-{candidate}/{i:03d}',variant=candidate,functional=True)
            for i,case in enumerate(boundaries(batch,primary)):
                r=invoke(batch,case,ROOT/f'outputs/fp8/b{batch}/{candidate}-boundaries/{i}',variant=candidate,functional=True)
                if 'mixed' in case['kind']:assert not r['text_reuse']
            invoke(batch,primary,ROOT/f'outputs/fp8/b{batch}/{candidate}-controls/direct',variant=candidate,execution='direct',functional=True)
            invoke(batch,primary,ROOT/f'outputs/fp8/b{batch}/{candidate}-controls/full-text',variant=candidate,full_text=True,functional=True)
            result['length_comparisons']={}
            for label,case in [('short',short),('long',long)]:
                length_rows=[]
                for i,variant in enumerate([baseline,candidate,candidate,baseline]):
                    out=ROOT/f'outputs/fp8/b{batch}/{candidate}-lengths/{label}-{i}-{variant}'
                    r=invoke(batch,case,out,variant=variant,repetitions=20,functional=False,full_text=True)
                    length_rows.append(dict(variant=variant,report=str(out/'report.json'),**statistics_for(r)))
                result['length_comparisons'][label]=dict(frames=case['total_frames'],rows=length_rows)
            result.update(status='candidate_application_lengths_passed_quality_pending',candidate_retained=None)
            write(private_report(batch,'008-'+candidate+'-comparison'),result)
            print(json.dumps(dict(event='candidate_validation_complete',batch=batch,gain_percent=result['gain_percent'])),flush=True)


if __name__=='__main__':main()
