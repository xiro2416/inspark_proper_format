"""Revalidate explicit reviewed route selections and bind final deployment evidence."""
import argparse,json,subprocess,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import BATCHES,write,sha,private_report
from validate_zipvoice_fp8_b128 import invoke,boundaries,inventory
from measure_zipvoice_fp8_b128 import statistics_for

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True);args=p.parse_args()
 review=json.loads(args.review.read_text())
 if review['status']!='b128_compute_review_complete' or set(review['batches'])!=set(map(str,BATCHES)):
  raise ValueError('B128 explicit completed execution reviews required')
 data=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text());primary=data['primary_760']
 short=min(data['quality'],key=lambda c:c['total_frames']);long=max(data['quality'],key=lambda c:c['total_frames'])
 selection=dict(status='running',batches={})
 from inspark_infer.runtime.device import GPULease
 with GPULease(3):
  for b in BATCHES:
   r=review['batches'][str(b)];variant=r['variant'];policy=r['pcm_policy']
   migration=private_report(b,'006-migration');quality=private_report(b,'005-quality'+('' if variant=='native' else '-'+variant))
   if not json.loads(migration.read_text())['migration_accepted'] or json.loads(quality.read_text())['status']!='paired_quality_metrics_complete':
    raise ValueError('Selected route migration/quality incomplete')
   evidence={str(x.relative_to(ROOT)):sha(x) for x in (migration,quality,args.review.resolve())}
   checks=[]
   for i,c in enumerate([primary,*boundaries(b,primary)]):
    out=ROOT/f'outputs/fp8/b{b}/final-b128-check/{i}'
    rep=invoke(b,c,out,variant=variant,functional=True,full_text=True,**policy)
    checks.append(dict(condition=str(Path(c['condition']).relative_to(ROOT)),condition_sha256=sha(c['condition']),workload={**c['workload'],'batch':b},seed=9102,text_reuse_disabled=True,
        wav_sha256={str(j):sha(out/f'{j:04d}.wav') for j in sorted({0,min(1,b-1),b-1})}))
    evidence[str((out/'report.json').relative_to(ROOT))]=sha(out/'report.json')
   direct=ROOT/f'outputs/fp8/b{b}/final-b128-check/direct';invoke(b,primary,direct,variant=variant,execution='direct',functional=True,full_text=True,**policy)
   evidence[str((direct/'report.json').relative_to(ROOT))]=sha(direct/'report.json')
   matched=[]
   source_policy=dict(chunk={128:32}[b],workers=4)
   for i,(label,v,pcm) in enumerate([('native','native',source_policy),('retained',variant,policy),('retained',variant,policy),('native','native',source_policy)]):
    out=ROOT/f'outputs/fp8/b{b}/final-b128-matched/{i}-{label}'
    rep=invoke(b,primary,out,variant=v,repetitions=20,functional=False,full_text=True,**pcm)
    stats=statistics_for(rep);stats.pop('power')
    matched.append(dict(label=label,variant=v,pcm_policy=pcm,**stats))
    evidence[str((out/'report.json').relative_to(ROOT))]=sha(out/'report.json')
   before=statistics.median(x['median_ms'] for x in matched if x['label']=='native')
   after=statistics.median(x['median_ms'] for x in matched if x['label']=='retained')
   comparison=dict(rows=matched,native_ms=before,retained_ms=after,gain_percent=100*(before-after)/before)
   if variant!='native' and after>=before:raise ValueError('Final retained route regressed against native baseline; review selection')
   measured={}
   for label,c in [('short',short),('primary',primary),('long',long)]:
    out=ROOT/f'outputs/fp8/b{b}/final-b128-measure/{label}'
    rep=invoke(b,c,out,variant=variant,repetitions=20,functional=False,full_text=True,minimum_seconds=30 if label=='primary' else 0,**policy)
    measured[label]=dict(frames=c['total_frames'],**statistics_for(rep),allocated_context_memory_bytes=rep['allocated_context_memory_bytes'],context_memory_bytes=rep['context_memory_bytes'])
    power=measured[label]['power']
    if 'request_windows' in power:power['request_windows']={k:v for k,v in power['request_windows'].items() if k!='requests'}
    evidence[str((out/'report.json').relative_to(ROOT))]=sha(out/'report.json')
   performance=private_report(b,'012-final-performance');write(performance,dict(status='retained_route_measured',batch=b,variant=variant,pcm_policy=policy,matched_primary=comparison,cases=measured))
   evidence[str(performance.relative_to(ROOT))]=sha(performance)
   inv=ROOT/f'outputs/fp8/b{b}/final-b128-check/0/inventory.json'
   selection['batches'][str(b)]=dict(migration_accepted=True,optimization_review_complete=True,variant=variant,inventory=str(inv.relative_to(ROOT)),pcm_policy=policy,evidence=evidence,fresh_cases=checks)
   write(ROOT/'outputs/fp8/b128/final-selection.json',selection)
   public=ROOT/f'reports/sm120/zipvoice/fp8/b{b}/history/002-final-performance.json'
   write(public,dict(status='final_route_local_validation_complete_publication_pending',batch=b,variant=variant,pcm_policy=policy,matched_primary=comparison,cases=measured,quality=json.loads(quality.read_text())['summary'] if 'summary' in json.loads(quality.read_text()) else json.loads(quality.read_text()).get('aggregate'),evidence_sha256=evidence,limits='Finite held-out corpus; quality report-only, no perceptual equivalence claim. Prepared-input clock. Power is whole-board NVML.'))
   print(json.dumps(dict(event='final_target_validated',batch=b,variant=variant)),flush=True)
 selection['status']='b128_fp8_migrated_optimized_validated';write(ROOT/'outputs/fp8/b128/final-selection.json',selection)
if __name__=='__main__':main()
