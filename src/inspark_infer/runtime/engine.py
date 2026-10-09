"""Incremental input API and first-packet-priority eager scheduler."""
import time
import math
from inspark_infer.models.indextts2.streaming import StreamingCore
from inspark_infer.runtime.splitter import Splitter
from inspark_infer.models.indextts2.audio import head_ready

class Engine(StreamingCore):
    def __init__(self,config):
        import torch
        from inspark_infer.models.indextts2.loader import Model
        from inspark_infer.runtime.indextts2.runtime import Runtime
        self.torch=torch;self.config=config;self.model=Model(config);self.tts=self.model.tts;self.student=self.model.student
        with torch.cuda.stream(self.model.stream),torch.inference_mode():self.rt=Runtime(self.model)
        self.vocoder=self.tts.bigvgan.forward;self.steps=4;self.sessions={};self.stages=[];self.failures=[];self.closed=False
        self.head_graphs=None
        self.deployment_state='raw'
        self.overlap_acoustics=False;self.acoustic_stream=None
        self.head_batch_barrier=False # opt-in experiment: wait for the selected head group
        self.device_round_b8=False
        self.device_round_bank=None
        self.device_round_batches=set()
        self.device_round_attempts=0
        self.device_round_successes=0
        self.device_round_fallbacks=0
        self.device_round_fallback_reasons={}
        self.device_round_status_reads=0
        self.device_round_status_wait_ms=0.0
        self.device_round_launched_rounds=0
        self.device_code_direct_rows=0
        self.speech_codes_host_rows=0
        self.output_d2h_bytes=0
        self.output_d2h_wait_ms=0.0
        self.output_d2h_transfer_ms=0.0
        self.strict_request_isolation=False
        self.rng_policy='legacy_per_request'
        self.profile_cuda=False;self.profile_spans=[]
    def validate_request_isolation(self,strict=False):
        """Preserve legacy seed mapping; shared RNG remains explicitly experimental."""
        shared=[]
        if getattr(self.rt.proposal,'batched_rng',False):shared.append('batched_proposal_rng')
        changed=[]
        if getattr(getattr(self.rt,'residual',None),'device_normal',False):changed.append('device_residual')
        if strict and (shared or changed):raise ValueError('Strict request isolation rejects changed/shared RNG: '+', '.join(shared+changed))
        self.strict_request_isolation=bool(strict)
        self.rng_policy=('legacy_shared' if shared else 'legacy_device_residual' if changed else
                         'request_owned_'+self.unified_first_chunk.runtime.backend if getattr(self,'unified_first_chunk',None) is not None else
                         'request_owned_device_round' if getattr(self,'device_round_b8',False) else 'legacy_per_request')
        return dict(strict_request_isolation=self.strict_request_isolation,rng_policy=self.rng_policy,
                    shared_rng_paths=shared,changed_sampler_paths=changed,
                    request_stream_v1_implemented=bool(getattr(self,'device_round_b8',False)))
    def request_rng_snapshot(self,request_id,prefix_draws=32):
        """Explicit diagnostic boundary; clone state without advancing request RNG.

        This synchronizes only the requested diagnostic tensors. Never call from
        a measured inference step, and do not equate post-branch state divergence
        with request contamination without checking the numerical branch trace.
        """
        import hashlib
        if not isinstance(prefix_draws,int) or not 1<=prefix_draws<=256:raise ValueError('prefix_draws must be 1..256')
        session=self.sessions[request_id];row=session.get('_row')
        state=row.state['cuda_rng'] if row is not None else session['gen'].get_state()
        device=session['gen'].device
        clone=self.torch.Generator(device=device);clone.set_state(state)
        digest=lambda value:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        noise=session.get('noise')
        from contextlib import nullcontext
        scope=self.torch.cuda.stream(self.model.stream) if device.type=='cuda' else nullcontext()
        with scope:
            prefix=self.torch.rand(prefix_draws,device=device,generator=clone)
            prefix_hash=digest(prefix)
            noise_hash=None if noise is None else digest(noise[:,:,:16])
        return dict(request_id=request_id,seed=session['case']['seed'],
                    rng_policy=getattr(self,'rng_policy','legacy_per_request'),
                    state_sha256=digest(state),next_uniform_prefix_sha256=prefix_hash,
                    cfm_noise_prefix_sha256=noise_hash,
                    prefix_draws=prefix_draws,rounds=session.get('rounds',0),
                    code_count=len(row.codes) if row is not None else len(session['codes']),
                    phase='active_row' if row is not None else 'session_boundary')
    def prepare_deployment(self,plan):
        from inspark_infer.runtime.deployment import prepare
        result=prepare(self,plan)
        if plan.get('static_gc_freeze',False):
            if self.sessions:raise RuntimeError('Freeze static objects before admitting requests')
            from inspark_infer.runtime.static_gc import StaticGCGuard
            self.static_gc_guard=StaticGCGuard()
            result['gc_policy']=self.static_gc_guard.start()
        return result
    def configure_profiling(self,enabled=True,trace_ranges=False):
        """Enable explicit post-capture observation; never alters deployment choices."""
        if getattr(self,'deployment_state','raw')!='ready':raise RuntimeError('Configure profiling after deployment preparation')
        self.profile_cuda=bool(enabled);self.trace_ranges=bool(trace_ranges)
        self.rt.profile_cuda=bool(enabled);self.rt.trace_ranges=bool(trace_ranges)
        self.profile_spans=[];self.rt.profile_spans=[]
        return dict(cuda_events=self.profile_cuda,trace_ranges=self.trace_ranges)
    def take_profile(self):
        self.torch.cuda.synchronize()
        def resolve(rows):
            return [dict(name=row['name'],batch=row['batch'],gpu_ms=row['start'].elapsed_time(row['end']),
                         host_ms=row['host_ms'],metadata=row['metadata']) for row in rows]
        result=dict(stages=resolve(self.profile_spans),ar_spans=resolve(getattr(self.rt,'profile_spans',[])))
        self.profile_spans=[];self.rt.profile_spans=[]
        return result
    def _release_row(self,row):
        if row is None:return
        from inspark_infer.runtime.cleanup import cleanup_all
        actions=[]
        for owner,field in ((self.rt.target,'kv'),(getattr(self.rt,'context',None),'cache')):
            release=getattr(owner,'release',None)
            value=getattr(row,field,None)
            if value is None:continue
            def discard(release=release,value=value,field=field):
                if release is not None:release(value)
                setattr(row,field,None)
            actions.append((field,discard))
        cleanup_all(actions)
    def prepare_head_graphs(self):
        """Explicit pre-admission deployment step; never invoked by run_ready."""
        from inspark_infer.runtime.graphs import HeadGraphs
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            self.model._acquire('capture')
            try:
                bank=HeadGraphs();bank.prepare(self);self.head_graphs=bank
                return bank.stats()
            finally:self.model._release()
    def prepare_reference(self,voice_id,source):
        if any(not s['complete'] for s in self.sessions.values()):raise RuntimeError('Cannot replace reference while requests are active')
        if self.head_graphs is not None:raise RuntimeError('Reference set sealed by deployment capture; rebuild deployment explicitly')
        result=self.model.prepare_reference(voice_id,source);self.rt.frontend.prime_voices();return result
    def create_session(self,request_id,voice_id,seed=0,emotion=None,arrival=None):
        if self.closed:raise RuntimeError('Engine closed')
        if getattr(self,'deployment_state','raw') in ('preparing','failed'):raise RuntimeError('Deployment is not usable; reconstruct the engine')
        if request_id in self.sessions:raise ValueError('Duplicate request id')
        if voice_id not in self.model.bank.entries:raise KeyError('Unregistered reference')
        case=dict(id=request_id,text='',voice_id=voice_id,seed=int(seed),emotion=list(emotion or [0.]*8))
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            s=self.new_sessions([case],time.perf_counter() if arrival is None else arrival)[0]
        s.update(splitter=Splitter(),input_closed=False,complete=False)
        self.sessions[request_id]=s
    def _cancel_text_futures(self,s):
        for _,future in s.pop('text_futures',{}).values():future.cancel()
    def _prefetch_next_text(self,s):
        """At most one unconsumed text job, using the original segment prefix."""
        if getattr(self,'_defer_text_prefetch',False):return
        if s.get('complete') or s.get('error'):
            self._cancel_text_futures(s);return
        index=len(s['chunks']);parts=s['parts'];queued=s.setdefault('text_futures',{})
        prefix=''.join(parts[:index+1]) if index<len(parts) else None
        for key,(text,future) in list(queued.items()):
            if key!=index or text!=prefix:
                future.cancel();queued.pop(key)
        if prefix is None or index in queued:return
        row=s.get('_row')
        # A paused AR row already owns this prepared prefix. EOF drain can also
        # reuse that text on demand, without queuing a redundant background job.
        if row is not None and getattr(row,'request',{}).get('text')==prefix:return
        parents=getattr(self,'_batch_text_parents',None)
        if parents is not None:
            from inspark_infer.runtime.shared_text_work import request_future
            if prefix not in parents:parents[prefix]=self.rt.frontend.pool.submit(self.rt.frontend._text,{'text':prefix})
            future=request_future(parents[prefix])
        else:future=self.rt.frontend.pool.submit(self.rt.frontend._text,{'text':prefix})
        queued[index]=(prefix,future)
    def admit_batch(self,requests):
        """Atomically admit pre-grouped requests through one worker message.

        Each record contains request_id, voice_id, text; seed/emotion/arrival
        are optional and finish defaults to True. Existing streaming methods
        remain available for records admitted with finish=False.
        """
        if self.closed:raise RuntimeError('Engine closed')
        if getattr(self,'deployment_state','raw') in ('preparing','failed'):
            raise RuntimeError('Deployment is not usable; reconstruct the engine')
        started=time.perf_counter();prepared=[];seen=set()
        if not isinstance(requests,(list,tuple)) or not requests:raise ValueError('Expected a nonempty request batch')
        allowed={'request_id','voice_id','text','seed','emotion','arrival','finish'}
        for record in requests:
            if not isinstance(record,dict) or not {'request_id','voice_id','text'}<=set(record) or set(record)-allowed:
                raise ValueError('Invalid batch request fields')
            identifier=record['request_id'];voice=record['voice_id'];text=record['text']
            if identifier in seen or identifier in self.sessions:raise ValueError('Duplicate request id')
            if voice not in self.model.bank.entries:raise KeyError('Unregistered reference')
            if not isinstance(text,str):raise TypeError('Text delta must be str')
            finish=record.get('finish',True)
            if type(finish) is not bool:raise ValueError('finish must be boolean')
            emotion=list(record.get('emotion') or [0.]*8)
            if len(emotion)!=8 or any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in emotion):
                raise ValueError('Invalid emotion vector')
            arrival=record.get('arrival')
            if arrival is not None and not math.isfinite(float(arrival)):raise ValueError('Invalid arrival time')
            splitter=Splitter();parts=splitter.feed(text)
            if finish:
                parts.extend(splitter.feed(final=True))
                if not parts:raise ValueError('No spoken text in request')
            prepared.append((identifier,voice,int(record.get('seed',0)),emotion,arrival,splitter,parts,finish))
            seen.add(identifier)
        created=[];prior=getattr(self,'_defer_text_prefetch',False)
        prior_parents=getattr(self,'_batch_text_parents',None)
        self._defer_text_prefetch=True
        try:
            for identifier,voice,seed,emotion,arrival,splitter,parts,finish in prepared:
                self.create_session(identifier,voice,seed,emotion,arrival)
                created.append(identifier)
                self.sessions[identifier].update(splitter=splitter,parts=parts,input_closed=finish)
            self._defer_text_prefetch=prior
            if getattr(self,'batch_text_dedup',False):self._batch_text_parents={}
            for identifier in created:self._prefetch_next_text(self.sessions[identifier])
        except BaseException as error:
            from inspark_infer.runtime.cleanup import cleanup_all
            owned=[row[0] for row in prepared if row[0] in self.sessions]
            cleanup_all([('batch request '+str(identifier),lambda rid=identifier:self.cancel(rid))
                         for identifier in reversed(owned)],primary=error)
            raise
        finally:
            self._defer_text_prefetch=prior
            self._batch_text_parents=prior_parents
        return dict(request_ids=created,admitted=len(created),server_admission_ms=(time.perf_counter()-started)*1000,
                    prefetch_jobs=sum(len(self.sessions[rid].get('text_futures',{})) for rid in created))
    def push_text(self,request_id,delta):
        s=self.sessions[request_id]
        if s['input_closed']:raise ValueError('Input is already closed')
        if not isinstance(delta,str):raise TypeError('Text delta must be str')
        parts=s['splitter'].feed(delta);s['parts'].extend(parts)
        self._prefetch_next_text(s)
        return list(parts)
    def finish_input(self,request_id):
        s=self.sessions[request_id]
        if s['input_closed']:raise ValueError('Input is already closed')
        s['parts'].extend(s['splitter'].feed(final=True));s['input_closed']=True
        if not s['parts']:raise ValueError('No spoken text in request')
        self._finish_or_drain(s)
        self._prefetch_next_text(s)
    def _finish_or_drain(self,s):
        if s['input_closed'] and len(s['chunks'])==len(s['parts']) and s['chunks']:
            if s['chunks'][-1]['eos'] and s['emitted']>=int(len(s['codes'])*1.72)*256:s['complete']=True
            else:s['parts'].append('') # EOF after head: drain remaining speech, without inventing text.
        # This also runs before each PCM publication. Never start background
        # tail tokenization here: it can contend for the GIL while the rest of
        # a ready head group is being delivered. The next scheduler turn's
        # frontend.prepare submits missing text, or explicit push/finish can
        # prefetch it. Requests cancelled after their head do no tail work.
        if s.get('complete'):self._cancel_text_futures(s)
    def ready(self):
        return [s for s in self.sessions.values() if not s['complete'] and not s['error'] and len(s['chunks'])<len(s['parts'])]
    def run_ready(self,on_chunk=None):
        return self._advance(on_chunk,None)
    def tick(self,on_chunk=None):
        """At most one speculative round, so the owner can admit input between rounds."""
        return self._advance(on_chunk,1)
    def _advance(self,on_chunk,max_rounds):
        if self.closed:raise RuntimeError('Engine closed')
        if getattr(self,'strict_request_isolation',False):self.validate_request_isolation(strict=True)
        ready=self.ready()
        if not ready:return []
        heads=[s for s in ready if not s['chunks']];pool=heads or ready
        ongoing=[s for s in pool if '_row' in s]
        index=len((ongoing or pool)[0]['chunks'])
        eligible=sorted((s for s in pool if len(s['chunks'])==index),key=lambda s:'_row' not in s)
        if index==0 and eligible and getattr(self,'head_batch_barrier',False):
            # Prefer the oldest ready request's static CFM prompt profile;
            # fill remaining places immediately, without a batching timer.
            def prompt_frames(session):
                values=self.model.bank.get(session['case']['voice_id'])['values']
                return values['voice.cache_mel'].shape[-1]
            anchor=prompt_frames(eligible[0])
            eligible.sort(key=lambda session:prompt_frames(session)!=anchor)
        group=eligible[:self.config['max_batch']]
        self.stages=[];self.failures=[]
        self.profile_spans=[];self.rt.profile_spans=[]
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            self.model._acquire('streaming')
            try:
                fresh=[s for s in group if '_row' not in s]
                if fresh:
                    for s,row in zip(fresh,self.prepare_rows(fresh,index)):
                        s['_row']=row
                rows=[s['_row'] for s in group]
                owner={id(s['_row']):s for s in group}
                pipeline=getattr(self,'head_ready_pipeline',None)
                if pipeline is not None and index==0 and max_rounds is None:
                    try:return pipeline.run(rows,owner,on_chunk)
                    except Exception as error:
                        self.unified_first_chunk.failure_count+=1
                        # The controller may already own tentative KV. Retain
                        # cleanup handles, but never retry from stale row caches.
                        for row in rows:owner[id(row)]['error']='Ready pipeline failed: '+str(error)
                        raise
                def is_ready(row):
                    return row.done or (index==0 and head_ready(len(row.codes),False))
                barrier=getattr(self,'head_batch_barrier',False) and index==0
                executed=0
                unified_runner=getattr(self,'unified_first_chunk',None)
                if (unified_runner is not None and index==0 and max_rounds is None and
                        not getattr(unified_runner,'supports',lambda _rows:True)(rows)):
                    # Select the declared same-recipe general path BEFORE any
                    # compact-cache mutation; no history truncation or online build.
                    self.device_round_attempts+=1
                    self.device_round_fallbacks+=1
                    reason='outside_static_first_chunk_profile'
                    self.device_round_fallback_reasons[reason]=self.device_round_fallback_reasons.get(reason,0)+1
                    unified_runner=None
                if unified_runner is not None and index==0 and max_rounds is None:
                    self.device_round_attempts+=1
                    try:
                        self.phase('unified_dspark',len(rows),lambda:unified_runner.run(rows))
                    except Exception:
                        unified_runner.failure_count+=1
                        # Provider owns tentative KV after entry. Never resume
                        # the host path with stale prefill cache on failure.
                        raise
                    self.device_round_successes+=1
                    self.device_round_status_reads+=unified_runner.status_reads
                    self.device_round_status_wait_ms+=unified_runner.status_wait_ms
                    self.device_round_launched_rounds+=unified_runner.launched_rounds
                    for row in rows:owner[id(row)]['rounds']+=len(row.accepted)
                while not (all if barrier else any)(is_ready(row) for row in rows):
                    active=[row for row in rows if not is_ready(row)]
                    self.phase('draft_verify_accept',len(active),
                               lambda:self.rt._step(active,self.config['max_speech_tokens']),
                               kv_lengths=[r.past_length for r in active])
                    for row in active:owner[id(row)]['rounds']+=1
                    executed+=1
                    if max_rounds is not None and executed>=max_rounds:break
                acoustic=[row for row in rows if is_ready(row)]
                if barrier and len(acoustic)!=len(rows):return []
                if not acoustic:return []
                dispatch=time.perf_counter()
                for row in acoustic:owner[id(row)]['acoustic_ready']=dispatch
                events=[]
                def publish(s):
                    self._finish_or_drain(s)
                    row=s.get('_row')
                    self._release_row(row)
                    s.pop('_row',None)
                    event=dict(request_id=s['case']['id'],chunk=s['chunks'][-1],complete=s['complete'])
                    if getattr(self,'trace_ranges',False):self.torch.cuda.nvtx.mark('PCM/'+str(index)+'/'+s['case']['id'])
                    events.append(event)
                    if on_chunk is not None:on_chunk(event)
                remaining=[row for row in rows if not is_ready(row)]
                if self.overlap_acoustics and remaining and max_rounds is None:
                    if self.acoustic_stream is None:self.acoustic_stream=self.torch.cuda.Stream(priority=-1)
                    self.acoustic_stream.wait_stream(self.model.stream)
                    def advance_remaining():
                        with self.torch.cuda.stream(self.model.stream):
                            self.phase('overlapped_ar',len(remaining),lambda:self.rt._step(remaining,self.config['max_speech_tokens']))
                            for row in remaining:owner[id(row)]['rounds']+=1
                    with self.torch.cuda.stream(self.acoustic_stream):
                        self.acoustic_rows(acoustic,owner,index,on_chunk=publish,on_enqueued=advance_remaining)
                else:self.acoustic_rows(acoustic,owner,index,on_chunk=publish)
                for row in acoustic:
                    s=owner[id(row)]
                    if s['error']:
                        failed=s.get('_row')
                        self._release_row(failed)
                        s.pop('_row',None)
                        raise RuntimeError(s['error'])
                return events
            finally:self.model._release()
    def cancel(self,request_id):
        # Cancellation is at an inference-step boundary; call between run_ready invocations.
        s=self.sessions[request_id]
        self._cancel_text_futures(s)
        try:self._release_row(s.get('_row'))
        except Exception as error:
            # Retain the cleanup handle for retry, but never schedule it again.
            s['error']='Cancellation cleanup failed: '+str(error)
            raise
        s.pop('_row',None)
        return self.sessions.pop(request_id)
    def release(self,request_id):
        s=self.sessions[request_id]
        if not s['complete']:raise RuntimeError('Request has not completed; use cancel to abandon')
        self._cancel_text_futures(s)
        return self.sessions.pop(request_id)
    def close(self):
        if self.closed:return
        from inspark_infer.runtime.cleanup import cleanup_all
        guard=getattr(self,'static_gc_guard',None)
        if guard is not None:guard.close()
        self.closed=True
        actions=[]
        pipeline=getattr(self,'head_ready_pipeline',None)
        if pipeline is not None:actions.append(('ready pipeline',pipeline.close))
        for s in self.sessions.values():self._cancel_text_futures(s)
        # Only final shutdown drains streams; normal cancel/reuse relies on owner-stream order.
        for name,stream in (('model',getattr(self.model,'stream',None)),
                            ('acoustic',getattr(self,'acoustic_stream',None))):
            if stream is not None:actions.append((name+' stream',stream.synchronize))
        actions.extend(('session '+str(key),lambda s=s:self._release_row(s.get('_row')))
                       for key,s in self.sessions.items())
        actions.extend((('runtime',self.rt.close),('model',self.model.close)))
        try:cleanup_all(actions)
        finally:
            self.sessions.clear();self.head_graphs=None
            self.head_ready_pipeline=None
            self.unified_first_chunk=None
            if hasattr(self,'prefix_graphs'):self.prefix_graphs=None
            self.device_round_bank=None
            self.rt=None;self.model=None;self.tts=None;self.student=None;self.vocoder=None
