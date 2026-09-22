"""Incremental input API and first-packet-priority eager scheduler."""
import time
from .core import StreamingCore
from .splitter import Splitter
from acc_infer_clear.audio.boundary import head_ready

class Engine(StreamingCore):
    def __init__(self,config):
        import torch
        from acc_infer_clear.models.loader import Model
        from acc_infer_clear.dspark.runtime import Runtime
        self.torch=torch;self.config=config;self.model=Model(config);self.tts=self.model.tts;self.student=self.model.student
        with torch.cuda.stream(self.model.stream),torch.inference_mode():self.rt=Runtime(self.model)
        self.vocoder=self.tts.bigvgan.forward;self.steps=2;self.sessions={};self.stages=[];self.failures=[];self.closed=False
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
        self.profile_cuda=False;self.profile_spans=[]
    def prepare_deployment(self,plan):
        from acc_infer_clear.runtime.deployment import prepare
        return prepare(self,plan)
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
    def prepare_precision(self,mode,components,convolutions=False):
        from acc_infer_clear.models.precision import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            return prepare(self,mode,components,convolutions)
    def prepare_acoustic_kernels(self,plan_path=None,mode='alias'):
        from acc_infer_clear.models.acoustic_kernels import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            return prepare(self,plan_path,mode)
    def prepare_acoustic_stage2(self,plan_path):
        from acc_infer_clear.models.acoustic_stage2 import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_acoustic_pipeline(self,plan_path):
        from acc_infer_clear.acoustic_pipeline.deploy import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_acoustic_refine(self,plan_path):
        from acc_infer_clear.acoustic_refine.deploy import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_ar_refine(self,plan_path):
        from acc_infer_clear.ar_refine.deploy import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_ar_pipeline(self,plan_path):
        from acc_infer_clear.ar_pipeline.deploy import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_slot_target(self,graphs=False):
        if self.sessions:raise RuntimeError('Prepare Target before admitting requests')
        from acc_infer_clear.dspark.slot_target import SlotTarget
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            self.rt.target=SlotTarget(self.rt.engine.target,self.config['max_batch'],consumer_layout=getattr(self,'attention_consumer_layout',False))
            if graphs:self.rt.target.prepare_graphs()
            return self.rt.target.stats()
    def prepare_target_seven(self,plan_path):
        from acc_infer_clear.target_seven.deploy import prepare
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return prepare(self,plan_path)
    def prepare_proposal_graphs(self):
        if self.sessions:raise RuntimeError('Capture before admitting requests')
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            return self.rt.proposal.prepare_graphs(self.config['max_batch'])
    def prepare_rnn_precision(self,mode):
        if self.sessions:raise RuntimeError('Prepare before admitting requests')
        from acc_infer_clear.kernels.plan_cache import PlanCache
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():return self.rt.proposal.prepare_precision(mode,PlanCache(self,'rnn_'+mode))
    def prepare_draft_graphs(self):
        if self.sessions:raise RuntimeError('Capture before admitting requests')
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            return self.rt.backbone.prepare_graphs(self.config['max_batch'])
    def prepare_slot_draft(self):
        if self.sessions or self.rt.backbone.graphs:raise RuntimeError('Prepare Draft slots before requests/graphs')
        from acc_infer_clear.dspark.slot_draft import DraftPool,SlotDraft
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            pool=DraftPool(self.rt.engine.draft,self.config['max_batch'])
            self.rt.context.pool=pool;self.rt.context.persistent=True
            self.rt.backbone=SlotDraft(self.rt.engine.draft,pool,consumer_layout=getattr(self,'attention_consumer_layout',False));self.rt.proposal.backbone=self.rt.backbone
    def _release_row(self,row):
        release=getattr(self.rt.target,'release',None)
        if row is not None and release is not None:release(row.kv)
        release=getattr(getattr(self.rt,'context',None),'release',None)
        if row is not None and release is not None:release(row.cache)
    def prepare_acceptance_fusion(self):
        if self.sessions:raise RuntimeError('Prepare before admitting requests')
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():self.rt.accept.prepare_fusion()
    def prepare_prefix_graphs(self):
        if self.sessions:raise RuntimeError('Capture before admitting requests')
        from acc_infer_clear.runtime.prefix_graphs import PrefixGraphs
        with self.torch.cuda.stream(self.model.stream),self.torch.inference_mode():
            bank=PrefixGraphs(self);bank.prepare(self.config['max_batch'])
            self.rt.target.prefill_body=bank.prefill;self.rt.latent.body=bank.latent
            self.prefix_graphs=bank
            return bank.stats()
    def prepare_head_graphs(self):
        """Explicit pre-admission deployment step; never invoked by run_ready."""
        from acc_infer_clear.runtime.graphs import HeadGraphs
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
    def push_text(self,request_id,delta):
        s=self.sessions[request_id]
        if s['input_closed']:raise ValueError('Input is already closed')
        if not isinstance(delta,str):raise TypeError('Text delta must be str')
        parts=s['splitter'].feed(delta);s['parts'].extend(parts);return list(parts)
    def finish_input(self,request_id):
        s=self.sessions[request_id]
        if s['input_closed']:raise ValueError('Input is already closed')
        s['parts'].extend(s['splitter'].feed(final=True));s['input_closed']=True
        if not s['parts']:raise ValueError('No spoken text in request')
        self._finish_or_drain(s)
    def _finish_or_drain(self,s):
        if not s['input_closed'] or len(s['chunks'])!=len(s['parts']) or not s['chunks']:return
        if s['chunks'][-1]['eos'] and s['emitted']>=int(len(s['codes'])*1.72)*256:s['complete']=True
        else:s['parts'].append('') # EOF after head: drain remaining speech, without inventing text.
    def ready(self):
        return [s for s in self.sessions.values() if not s['complete'] and not s['error'] and len(s['chunks'])<len(s['parts'])]
    def run_ready(self,on_chunk=None):
        return self._advance(on_chunk,None)
    def tick(self,on_chunk=None):
        """At most one speculative round, so the owner can admit input between rounds."""
        return self._advance(on_chunk,1)
    def _advance(self,on_chunk,max_rounds):
        ready=self.ready()
        if not ready:return []
        heads=[s for s in ready if not s['chunks']];pool=heads or ready
        ongoing=[s for s in pool if '_row' in s]
        index=len((ongoing or pool)[0]['chunks'])
        group=sorted((s for s in pool if len(s['chunks'])==index),key=lambda s:'_row' not in s)[:self.config['max_batch']]
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
                def is_ready(row):
                    return row.done or (index==0 and head_ready(len(row.codes),False))
                barrier=getattr(self,'head_batch_barrier',False) and index==0
                executed=0
                use_device=(getattr(self,'device_round_b8',False) and index==0 and
                            max_rounds is None and len(rows) in self.device_round_batches and
                            max(r.past_length for r in rows)+8<=128 and
                            max(r.cache.length for r in rows)+7<=128)
                if use_device:
                    self.device_round_attempts+=1
                    if self.device_round_bank is not None:
                        device_runner=self.device_round_bank;device_runner.run(rows)
                    else:
                        from acc_infer_clear.dspark.device_round import DeviceRoundHead
                        device_runner=DeviceRoundHead(self.rt,rows,self.config['max_speech_tokens']);device_runner.run()
                    if device_runner.failed:self.device_round_fallbacks+=1
                    else:
                        self.device_round_successes+=1
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
                    row=s.pop('_row',None)
                    self._release_row(row)
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
                        failed=s.pop('_row',None)
                        self._release_row(failed)
                        raise RuntimeError(s['error'])
                return events
            finally:self.model._release()
    def cancel(self,request_id):
        # Cancellation is at an inference-step boundary; call between run_ready invocations.
        s=self.sessions.pop(request_id)
        row=s.pop('_row',None)
        self._release_row(row)
        return s
    def release(self,request_id):
        s=self.sessions[request_id]
        if not s['complete']:raise RuntimeError('Request has not completed; use cancel to abandon')
        return self.sessions.pop(request_id)
    def close(self):
        if self.closed:return
        self.sessions.clear();self.rt.close();self.model.close()
        self.head_graphs=None
        if hasattr(self,'prefix_graphs'):self.prefix_graphs=None
        self.rt=None;self.model=None;self.tts=None;self.student=None;self.vocoder=None;self.closed=True
