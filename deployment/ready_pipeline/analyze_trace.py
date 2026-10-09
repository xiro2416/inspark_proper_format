"""GPU-kernel union intersection for the explicit AR/acoustic NVTX lanes."""
import collections,json,sqlite3,sys
from pathlib import Path
p=Path(sys.argv[1]);c=sqlite3.connect(p)
ranges=c.execute('select text,globalTid,start,end from NVTX_EVENTS where end is not null and text is not null').fetchall()
lanes={tid:('AR' if name=='READY/ar' else 'acoustic') for name,tid,_,_ in ranges if name in ('READY/ar','READY/acoustic')}
bythread=collections.defaultdict(list)
for name,tid,start,end in ranges:
 if name in ('READY/ar','latent','condition','cfm4','vocoder'):bythread[tid].append((start,end,name))
events=[];streams=collections.defaultdict(lambda:dict(kernels=0,kernel_ms=0.0));contexts=set()
for start,end,stream,ctx,tid,launch in c.execute('select k.start,k.end,k.streamId,k.contextId,r.globalTid,r.start from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r using(correlationId) order by k.start'):
 lane=lanes.get(tid)
 if lane is None:raise RuntimeError('Unattributed CUDA kernel thread')
 stage=next((name for a,b,name in sorted(bythread[tid],reverse=True) if a<=launch<=b),'other')
 contexts.add(ctx);entry=streams[str(stream)];entry['lane']=lane;entry['kernels']+=1;entry['kernel_ms']+=(end-start)/1e6
 events.extend([(start,1,lane,stage),(end,-1,lane,stage)])
active={k:collections.Counter() for k in ('AR','acoustic')};previous=None;overlap=collections.Counter();active_union=collections.Counter()
for timestamp,delta,lane,stage in sorted(events):
 if previous is not None and timestamp>previous:
  for k in active:
   if active[k]:active_union[k]+=timestamp-previous
  if all(active.values()):overlap[' / '.join('+'.join(sorted(active[k])) for k in active)]+=timestamp-previous
 active[lane][stage]+=delta
 if active[lane][stage]==0:del active[lane][stage]
 previous=timestamp
if len(contexts)!=1:raise RuntimeError('Expected same primary CUDA context')
r=dict(contexts=sorted(contexts),streams=dict(streams),lane_gpu_union_ms={k:v/1e6 for k,v in active_union.items()},ar_acoustic_kernel_overlap_ms=sum(overlap.values())/1e6,stage_overlap_ms={k:v/1e6 for k,v in overlap.most_common()},scope='Two first-head B64 waves, diagnostic profiled GPU intervals; union removes within-lane overlap, NVTX phase assigned at same-thread CUDA launch. Not an E2E saving.')
p.with_name('trace-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
