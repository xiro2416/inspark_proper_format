"""Attribute actual GPU intervals to host NVTX launch stages, then intersect streams."""
import collections,json,sqlite3,sys
from pathlib import Path
path=Path(sys.argv[1]);c=sqlite3.connect(path)
ranges=c.execute('select text,globalTid,start,end from NVTX_EVENTS where end is not null and text is not null').fetchall()
stages=('text_prepare','prefill','unified_dspark','latent','condition','cfm4','vocoder')
bythread=collections.defaultdict(list)
for name,tid,start,end in ranges:
    if name in stages:bythread[tid].append((start,end,name))
banks={tid:name.split('/')[0] for name,tid,_,_ in ranges if name.startswith('BANK')}
streams=collections.defaultdict(list)
for start,end,stream,ctx,tid,launch in c.execute('select k.start,k.end,k.streamId,k.contextId,r.globalTid,r.start from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r using(correlationId) order by k.start'):
    stage=next((name for a,b,name in bythread[tid] if a<=launch<=b),'other')
    streams[stream].append((start,end,stage,ctx,banks.get(tid,'unknown')))
if len(streams)!=2:raise RuntimeError('Expected exactly two bank compute streams')
events=[]
for stream,rows in streams.items():
    for start,end,stage,ctx,bank in rows:
        events.extend([(start,1,bank,stage,ctx),(end,-1,bank,stage,ctx)])
if len({x[3] for rows in streams.values() for x in rows})!=1:raise RuntimeError('Expected one CUDA context')
active={bank:collections.Counter() for bank in ('BANK0','BANK1')}
overlap=collections.Counter();examples=[];previous=None
for timestamp,delta,bank,stage,ctx in sorted(events):
    if previous is not None and timestamp>previous and all(active.values()):
        names=['+'.join(sorted(active[b])) for b in ('BANK0','BANK1')]
        overlap[' / '.join(names)]+=timestamp-previous
        if names[0]!=names[1] and len(examples)<12:examples.append(dict(start_ns=previous,end_ns=timestamp,bank0_stage=names[0],bank1_stage=names[1]))
    active[bank][stage]+=delta
    if not active[bank][stage]:del active[bank][stage]
    previous=timestamp
sync_evidence=collections.defaultdict(lambda:dict(calls=0,host_wait_ms=0.0,other_bank_kernel_ms_during_wait=0.0))
for start,end,tid,name in c.execute("select r.start,r.end,r.globalTid,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like '%Synchronize%'"):
    bank=banks.get(tid)
    if bank is None:continue
    stage=next((n for a,b,n in bythread[tid] if a<=start<=b),'other')
    key=bank+'/'+stage+'/'+name
    out=sync_evidence[key];out['calls']+=1;out['host_wait_ms']+=(end-start)/1e6
    intervals=sorted((max(start,a),min(end,b)) for rows in streams.values() for a,b,_,_,owner in rows if owner!=bank and a<end and b>start)
    last=end_total=None;duration=0
    for a,b in intervals:
        if last is None or a>end_total:
            if last is not None:duration+=end_total-last
            last,end_total=a,b
        else:end_total=max(end_total,b)
    if last is not None:duration+=end_total-last
    out['other_bank_kernel_ms_during_wait']+=duration/1e6
report=dict(contexts=sorted({x[3] for v in streams.values() for x in v}),streams={str(k):dict(bank=v[0][4],kernels=len(v),kernel_ms=sum(x[1]-x[0] for x in v)/1e6) for k,v in streams.items()},cross_bank_kernel_overlap_ms=sum(overlap.values())/1e6,stage_overlap_ms={k:v/1e6 for k,v in overlap.most_common()},cross_stage_examples=examples,synchronization_overlap=dict(sync_evidence),scope='Two profiled 32-request first-chunk waves. Stage assigned by CUDA runtime launch inside same-thread NVTX range; overlap uses GPU kernel interval unions (no double-counting), not host range overlap. Profiler timings diagnostic only.')
path.with_name('trace-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
