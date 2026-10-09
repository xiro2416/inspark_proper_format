"""Opt-in B64 request-ready pipeline; shared read-only model, owned stage buffers."""
from __future__ import annotations
import copy
import concurrent.futures
import json
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
FIELDS=('tokens','accepted','token_lengths','past','draft_lengths','mel_lengths','last',
        'done','ready','rounds','committed','active')


def state_tensors(controller):
    r,p=controller.runtime,controller.provider
    return [(getattr(r,name),0) for name in FIELDS]+[(r.draws.values,0),
            (p.target_cache,2),(p.target_keep,0),(p.draft_cache,2)]


def transfer(source,destination,indices):
    """Move committed and speculative bookkeeping without reset/re-seeding."""
    if source is destination:raise ValueError('Cannot compact a bank into itself')
    if indices.ndim!=1 or indices.dtype!=torch.long:raise ValueError('Expected unique int64 request indices')
    if indices.device.type=='cpu' and len(set(indices.tolist()))!=indices.numel():raise ValueError('Duplicate request slot')
    n=indices.numel()
    if not 0<n<=destination.runtime.batch:raise ValueError('Compaction exceeds destination')
    for (src,axis),(dst,other_axis) in zip(state_tensors(source),state_tensors(destination)):
        if axis!=other_axis:raise ValueError('State axis mismatch')
        dst.zero_();dst.narrow(axis,0,n).copy_(src.index_select(axis,indices))
    r=destination.runtime
    r.tokens[n:].fill_(r.eos);r.token_lengths[n:].fill_(1);r.last[n:].fill_(r.eos)
    r.done[n:].fill_(True);r.ready[n:].fill_(True);r.accepted[n:].fill_(-1)
    r.failures.copy_(source.runtime.failures);r.capacity_failures.copy_(source.runtime.capacity_failures);r.status.zero_()


def selected(batch):
    from inspark_infer.runtime.bundle_paths import read_json
    return read_json(ROOT/f'configs/hardware/sm89/indextts/int8_b{batch}_selected.json')


def internal_plan(batch):
    if batch!=48:return selected(batch)
    p=dict(selected(64),batch=48)
    for component in ('target','draft'):p[component+'_plan']=str(ROOT/f'artifacts/sm89/int8_smoothquant/b48/{component}/model.plan.json')
    return p


class ReadyPipeline:
    def __init__(self,engine,acoustic_priority=0,mode="C"):
        from inspark_infer.ops.tensorrt.unified_ar import StaticARProvider
        from inspark_infer.runtime.unified_deployment import FirstChunkController
        from inspark_infer.ops.tensorrt.unified_prefix import NativePrefixEngine
        from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113,NativeVocoder113
        from inspark_infer.runtime.asset_identity import calibration_digest
        from inspark_infer.runtime.graphs import HeadGraphs
        e=self.engine=engine
        if e.config['max_batch']!=64 or e.unified_first_chunk.runtime.backend!='framework_dspark_adapter':raise ValueError('Explicit framework B64 head required')
        self.closed=False;self.controllers={64:e.unified_first_chunk};self.mode=mode;self.waves=[];self.stream=torch.cuda.Stream(priority=acoustic_priority)
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='ready-acoustic')
        with torch.cuda.stream(e.model.stream),torch.inference_mode():
            for b in (48,32,16):
                plan=internal_plan(b)
                provider=StaticARProvider(e.rt,plan['target_plan'],plan['draft_plan'],b)
                for component in ('target','draft'):
                    asset=getattr(provider,component+'_engine').plan
                    origin=getattr(self.controllers[64].provider,component+'_engine').plan
                    hashes=lambda value:{r['role']:r['sha256'] for r in value['provenance']['model_sources']}
                    if hashes(asset)!=hashes(origin):raise ValueError('AR checkpoint identity mismatch')
                    recipe=asset['quantization_recipe']
                    if recipe['scheme']!=plan['precision'] or recipe['calibration']['sha256']!=calibration_digest(plan,component):raise ValueError('AR calibration identity mismatch')
                self.controllers[b]=FirstChunkController(e,provider,e.unified_first_chunk.runtime.proposal,True,graph_burst_rounds=2)
            # Shallow stage view shares model/weights/sessions; all mutable compute
            # providers, latent scratch, graphs and phase lists are independent.
            p=selected(16)
            if p['model_tensor_hashes']!=selected(64)['model_tensor_hashes'] or any(calibration_digest(p,k)!=calibration_digest(selected(64),k) for k in ('target','cfm','vocoder')):raise ValueError('Acoustic checkpoint/calibration identity mismatch')
            v=self.acoustic=copy.copy(e);v.config=dict(e.config,max_batch=16,head_graph_batches=[16])
            v.rt=copy.copy(e.rt);v.rt.latent=copy.copy(e.rt.latent)
            bank=copy.copy(e.unified_prefix);bank.engine=v;bank.backends=dict(bank.backends)
            bank.backends['latent']=NativePrefixEngine(p['latent_plan'],kind='latent',batch=16,calibration_sha256=calibration_digest(p,'target'),scheme=p['precision'])
            bank.hits={'prefill':0,'latent':0};v.rt.latent.body=bank.latent;v.unified_prefix=bank
            v.student=NativeCFMSolver113(p['cfm_plan'],e.student.eager)
            v.vocoder=NativeVocoder113(p['vocoder_plan'],e.vocoder.eager)
            if any(e.model.bank.get(voice)['values']['voice.cache_mel'].shape[-1]!=258 for voice in list(e.model.bank.entries)):raise ValueError('Ready pipeline requires declared prompt258 references')
            v.head_graphs=HeadGraphs();v.head_graphs.prepare(v)
            v.output_d2h_stream=None;v.stages=[];v.failures=[];v.profile_spans=[];v.rt.events=[]
        self.stream.wait_stream(e.model.stream)

    def close(self):
        if self.closed:return
        self.closed=True
        self.executor.shutdown(wait=True)
        self.stream.synchronize()

    def render(self,rows,owner,event):
        v=self.acoustic
        with torch.inference_mode(),torch.cuda.stream(self.stream):

            for dependency in event:self.stream.wait_event(dependency)
            if getattr(v,'trace_ranges',False):torch.cuda.nvtx.range_push('READY/acoustic')
            try:v.acoustic_rows(rows,owner,0)
            finally:
                if getattr(v,'trace_ranges',False):torch.cuda.nvtx.range_pop()
        return rows

    def run(self,rows,owner,on_chunk):
        if self.closed:raise RuntimeError('Ready pipeline closed')
        e=self.engine;compact=self.mode in ('C','D');overlap=self.mode in ('B','D')
        if not self.controllers[64].supports(rows) or any(e.model.bank.get(owner[id(row)]['case']['voice_id'])['values']['voice.cache_mel'].shape[-1]!=258 for row in rows):raise ValueError('Ready pipeline input outside declared first-head profile')
        if self.mode not in ('A','B','C','D'):raise ValueError('Unknown readiness policy')
        c=self.controllers[64];c.begin(rows);mapping=list(range(len(rows)));snapshots=set();queue=[];events=[];future=None
        logical_rounds=0;row_rounds=0;status_reads=0;wait_ms=0.;bursts=[];compactions=[];groups=[]
        self.acoustic.trace_ranges=getattr(e,'trace_ranges',False)
        def publish(done):
            for row in done:
                s=owner[id(row)]
                if s['error']:raise RuntimeError(s['error'])
                e._finish_or_drain(s);e._release_row(row);s.pop('_row',None)
                item=dict(request_id=s['case']['id'],chunk=s['chunks'][-1],complete=s['complete'])
                if getattr(e,'trace_ranges',False):torch.cuda.nvtx.mark('READY/PCM/'+s['case']['id'])
                events.append(item)
                if on_chunk:on_chunk(item)
        def flush_future(block=False):
            nonlocal future
            if future is not None and (block or future.done()):
                publish(future.result());future=None
        def dispatch(all_ready=False):
            nonlocal future
            flush_future()
            while future is None and queue:
                # The fixed native acoustic profile is F310/P258. Other prompts
                # are outside this explicit trial; never silently capture eager.
                eligible=queue[:16]
                if len(eligible)<16 and not all_ready:return
                del queue[:len(eligible)];batch=[rows[i] for i in eligible]
                event=list({id(row.head_ready_event):row.head_ready_event for row in batch}.values())
                groups.append(dict(indices=eligible,ar_round=logical_rounds,rows=len(batch)))
                if overlap:future=self.executor.submit(self.render,batch,owner,event)
                else:publish(self.render(batch,owner,event))
        def service_outputs():
            dispatch(False)
            return future is not None
        try:
            while logical_rounds<64:
                flush_future()
                if getattr(e,'trace_ranges',False):torch.cuda.nvtx.range_push('READY/ar')
                try:ob=c.runtime.advance_burst(on_wait=service_outputs if overlap else None)
                finally:
                    if getattr(e,'trace_ranges',False):torch.cuda.nvtx.range_pop()
                logical_rounds+=ob['launched_rounds'];row_rounds+=c.runtime.batch*ob['launched_rounds'];status_reads+=1;wait_ms+=ob['status_wait_ms']
                newly=[i for i,original in enumerate(mapping) if ob['ready'][i] and original not in snapshots]
                bursts.append(dict(round=logical_rounds,engine_batch=c.runtime.batch,active_rows=sum(not ob['ready'][i] for i in range(len(mapping))),newly_ready=len(newly)))
                if newly:
                    result=c.runtime.result();idx=torch.tensor(newly,device='cuda',dtype=torch.long)
                    history=result['accepted'].index_select(0,idx).cpu().tolist()
                    for slot,accepted in zip(newly,history):
                        original=mapping[slot];row=rows[original];count,past,dl,n,done,last,ready=result['metadata'][slot]
                        if not ready:raise RuntimeError('Snapshot without readiness')
                        tokens=result['tokens'][slot].clone()
                        row.codes=list(tokens[:count].split(1));row.device_codes_buffer=tokens;row.device_code_count=count;row.device_round_final=True
                        row.last_token_host=last;row.accepted=accepted[:n];row.past_length=past;row.done=bool(done)
                        owner[id(row)]['rounds']+=n;snapshots.add(original);queue.append(original)
                    snapshot_done=torch.cuda.Event();snapshot_done.record(e.model.stream)
                    for slot in newly:rows[mapping[slot]].head_ready_event=snapshot_done
                all_ready=len(snapshots)==len(rows)
                dispatch(all_ready)
                if all_ready:break
                if compact:
                    live=[i for i in range(len(mapping)) if not ob['ready'][i]]
                    b=next(b for b in (16,32,48,64) if b>=len(live))
                    if b<c.runtime.batch:
                        dest=self.controllers[b];idx=torch.tensor(live,device='cuda',dtype=torch.long)
                        started=time.perf_counter();transfer(c,dest,idx)
                        compactions.append(dict(from_batch=c.runtime.batch,to_batch=b,active_rows=len(live),host_enqueue_ms=1000*(time.perf_counter()-started)))
                        mapping=[mapping[i] for i in live];c=dest
            else:raise RuntimeError('Readiness controller round limit')
            while future is not None or queue:
                flush_future(block=True);dispatch(True)
            if len(events)!=len(rows) or len({x['request_id'] for x in events})!=len(rows):raise RuntimeError('Incomplete/duplicate delivery')
        finally:
            # Join before exposing errors/cancellation to callers; no live stage
            # may retain a request buffer after an inference boundary returns.
            if future is not None:future.result()
        e.device_round_attempts+=1;e.device_round_successes+=1;e.device_round_status_reads+=status_reads
        e.device_round_status_wait_ms+=wait_ms;e.device_round_launched_rounds+=logical_rounds
        e.rt.native_target_steps+=logical_rounds;e.rt.device_target_steps=getattr(e.rt,'device_target_steps',0)+logical_rounds
        e.rt.backbone.native_full_steps=getattr(e.rt.backbone,'native_full_steps',0)+logical_rounds
        self.waves.append(dict(mode=self.mode,logical_rounds=logical_rounds,computed_row_rounds=row_rounds,bursts=bursts,compactions=compactions,acoustic_groups=groups))
        return events
