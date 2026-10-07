"""Explicit streaming acoustic protocol. No execution-backend switches."""
import time
from dataclasses import dataclass
from inspark_infer.models.indextts2.audio import head_ready, head_window, HOP, RIGHT
from inspark_infer.runtime.splitter import split

@dataclass
class AcousticOp:
    kind: str
    args: tuple
    kwargs: dict
    cached_prefix: object = None

class StreamingCore:

    def phase(self, name, batch, fn, **metadata):
        cuda_start=cuda_end=None
        if getattr(self,'profile_cuda',False):
            cuda_start=self.torch.cuda.Event(enable_timing=True);cuda_end=self.torch.cuda.Event(enable_timing=True);cuda_start.record()
        start = time.perf_counter()
        if getattr(self,'trace_ranges',False):
            with self.torch.profiler.record_function(name):
                self.torch.cuda.nvtx.range_push(name)
                try:out=fn()
                finally:self.torch.cuda.nvtx.range_pop()
        else:out = fn()
        end = time.perf_counter()
        if cuda_end is not None:
            cuda_end.record();self.profile_spans.append(dict(name=name,batch=batch,start=cuda_start,end=cuda_end,host_ms=(end-start)*1000,metadata=metadata))
        self.stages.append(dict(stage=name, batch=batch, start=start, end=end, host_ms=(end - start) * 1000, **metadata))
        return out

    def new_sessions(self, cases, arrival):
        torch = self.torch
        return [dict(case=c, arrival=arrival, text='', codes=[], emitted=0, pending=None, chunks=[], rounds=0, accepted=[], error=None, gen=torch.Generator(device='cuda').manual_seed(c['seed']), noise=torch.randn(1, 80, 8192, device='cuda', generator=torch.Generator(device='cuda').manual_seed(c['seed'] + 1)), parts=split(c['text'])) for c in cases]

    def prepare_rows(self, sessions, index):
        """Prefill once; the returned rows survive subsequent scheduler calls."""
        from inspark_infer.runtime.indextts2.core import BatchRow
        
        
        torch = self.torch
        rt = self.rt
        head = index == 0
        rows = []
        owner = {}
        b = len(sessions)
        requests = []
        for s in sessions:
            c = s['case']
            text_value=s['text']+s['parts'][index]
            request=dict(id=c['id'], text=text_value, voice_id=c['voice_id'], seed=c['seed'], emotion=c['emotion'], arrival=s['arrival'])
            queued=s.get('text_futures',{}).pop(index,None)
            if queued is not None and queued[0]==text_value:
                request['text_future_text']=text_value;request['text_future']=queued[1]
            requests.append(request)
        prepared = self.phase('text_prepare', b, lambda: rt.frontend.prepare(requests))
        for s, r in zip(sessions, requests):
            p = prepared[r['id']]
            if len(p.prepared_segments) != 1:
                raise ValueError('Text context cap exceeded')
            row = BatchRow(r, p, s['gen'], state={'cuda_rng': s['gen'].get_state()})
            rows.append(row)
            owner[id(row)] = s

        def prefill():
            jobs = []
            for row in rows:
                s = owner[id(row)]
                seg = row.prepared.prepared_segments[0]
                emb = seg.prefix
                mask = seg.mask
                if s['codes']:
                    old = torch.tensor([s['codes']], device='cuda', dtype=torch.long)
                    tm = rt.engine.target.model
                    positions = torch.arange(2, len(s['codes']) + 2, device='cuda')[None]
                    emb = torch.cat((emb, tm.embeddings(old) + tm.text_pos_embedding.emb(positions)), 1)
                    mask = torch.nn.functional.pad(mask, (0, len(s['codes'])), value=1)
                row.mask = mask
                row.prefix_length = seg.prefix.shape[1]
                row.mel_length = row.prefix_length - 1
                row.past_length = emb.shape[1]
                jobs.append((emb, mask))
            outputs = rt.target.prefill(jobs)
            # Own every imported slot before any later allocation/sample can fail.
            for row, output in zip(rows, outputs):row.kv=output[1]
            ctx = []
            for row, (logits, kv, selected, final) in zip(rows, outputs):
                s = owner[id(row)]
                row.kv = kv
                row.cache = rt.engine.draft.empty_cache(1, 'cuda', torch.float32)
                ctx.append(((row.cache, selected, final), {}))
            overlap=head and getattr(self,'prefill_context_overlap',False)
            context_stream=getattr(self,'prefill_context_stream',None) if overlap else None
            parent=torch.cuda.current_stream()
            if context_stream is not None:
                context_stream.wait_stream(parent)
                with torch.cuda.stream(context_stream):
                    for _,_,selected,final in outputs:
                        selected.record_stream(context_stream);final.record_stream(context_stream)
                    rt.context(ctx)
                    for row in rows:
                        for value in (*row.cache.keys,*row.cache.values):value.record_stream(parent)
            try:
                for row,(logits,_,_,_) in zip(rows,outputs):
                    s=owner[id(row)]
                    row.codes=[torch.tensor([c],device='cuda',dtype=torch.long) for c in s['codes']]+[rt.sample(row,logits[:,-1])]
                if context_stream is None:rt.context(ctx)
            finally:
                if context_stream is not None:parent.wait_stream(context_stream)
            if getattr(self,'batch_prefill_eos',False):
                ended=torch.cat([row.codes[-1].reshape(1) for row in rows]).eq(int(rt.engine.target.gpt.stop_mel_token)).cpu().tolist()
                for row,value in zip(rows,ended):row.done=bool(value)
            else:
                for row in rows:
                    row.done = int(row.codes[-1].item()) == int(rt.engine.target.gpt.stop_mel_token)
        rng_states=[s['gen'].get_state() for s in sessions]
        try:
            self.phase('prefill_rebuild' if not head else 'prefill', b, prefill)
        except Exception as error:
            from inspark_infer.runtime.cleanup import cleanup_all
            actions=[]
            for row in rows:
                release=getattr(rt.target,'release',None)
                if row.kv is not None and release is not None:
                    actions.append(('prepare Target KV',lambda row=row,release=release:release(row.kv)))
                release=getattr(rt.context,'release',None)
                if row.cache is not None and release is not None:
                    actions.append(('prepare Draft KV',lambda row=row,release=release:release(row.cache)))
            actions.extend(('prepare RNG',lambda s=s,state=state:s['gen'].set_state(state))
                           for s,state in zip(sessions,rng_states))
            cleanup_all(actions,primary=error)
            raise
        for s,request in zip(sessions,requests):s['text']=request['text']
        return rows

    def acoustic_rows(self, rows, owner, index, on_chunk=None, on_enqueued=None):
        """Render only ready requests, never wait for another row's AR."""
        torch, rt = self.torch, self.rt
        head = index == 0
        live = []
        code_tensors = []
        ops = []
        eos = int(rt.engine.target.gpt.stop_mel_token)
        for row in rows:
            s = owner[id(row)]
            gpu_codes=head and getattr(row,'device_round_final',False)
            if gpu_codes:
                self.device_code_direct_rows=getattr(self,'device_code_direct_rows',0)+1
                count=row.device_code_count
                ended=row.last_token_host==eos
                codes=None
            else:
                self.speech_codes_host_rows=getattr(self,'speech_codes_host_rows',0)+1
                codes = (self.phase('speech_codes_d2h',1,lambda:torch.cat(row.codes).cpu().tolist())
                         if getattr(self,'trace_ranges',False) else torch.cat(row.codes).cpu().tolist())
                ended = bool(codes and codes[-1] == eos)
                count=len(codes)
            if row.done and (not ended):
                s['error'] = 'No EOS before original1500 total-token cap'
                self.failures.append(dict(id=s['case']['id'], segment=index, error=s['error']))
                continue
            speech_count=count-int(ended)
            if speech_count<=0 or (speech_count == len(s['codes']) and not ended):
                s['error'] = 'Empty speech continuation'
                self.failures.append(dict(id=s['case']['id'], segment=index, error=s['error']))
                continue
            if codes is not None:
                codes=codes[:-1] if ended else codes
                assert codes[:len(s['codes'])] == s['codes']
                s['new_codes'] = codes
            s['new_codes_count']=speech_count
            s['eos'] = ended
            s['accepted'] += row.accepted
            s['kv_head_lengths'] = row.past_length
            seg = row.prepared.prepared_segments[0]
            v = self.model.bank.get(s['case']['voice_id'])['values']
            current=(row.device_codes_buffer[:speech_count][None] if gpu_codes else
                     torch.tensor([codes], device='cuda', dtype=torch.long))
            # Each row is already cropped to its exact text/code length. Pass
            # Python lengths to the legacy GPT input preparation so its
            # per-row padding checks do not read two CUDA scalars back to CPU.
            if getattr(self,'latent_gpu_scalar_lengths',not hasattr(self,'unified_first_chunk')):
                text_length=torch.tensor([seg.text_ids.shape[-1]],device='cuda')
                code_length=torch.tensor([speech_count],device='cuda')
            else:
                text_length=[seg.text_ids.shape[-1]]
                code_length=[speech_count]
            args = (seg.speech_latent, seg.text_ids, text_length, current, code_length, v['voice.cache_emo_cond'])
            cached_head=(seg.prefix if head and hasattr(self,'unified_first_chunk')
                         and getattr(self,'latent_cached_prefix_enabled',True) else None)
            # The cached-prefix path reads args/cached_prefix only. Avoid three
            # unused CUDA allocations per row; retain legacy kwargs on fallback.
            kw = ({} if cached_head is not None and getattr(self,'head_handoff',False) else
                  dict(cond_mel_lengths=torch.tensor([v['voice.cache_spk_cond'].shape[-1]], device='cuda'), emo_cond_mel_lengths=torch.tensor([v['voice.cache_emo_cond'].shape[-1]], device='cuda'), emo_vec=row.prepared.emovec, use_speed=torch.zeros(1, device='cuda', dtype=torch.long)))
            live.append(s)
            code_tensors.append(current)
            ops.append(AcousticOp('latent', args, kw,
                                  cached_prefix=cached_head))
        if not live:
            return
        latents = self.phase('latent', len(live), lambda: rt._latents(ops))
        code_by_session={id(s):codes for s,codes in zip(live,code_tensors)}
        prepared_conditions=None
        if getattr(self,'config',{}).get('batch_conditions',False):
            from inspark_infer.runtime.unified_conditions import grouped_conditions
            prepared_conditions,condition_groups=grouped_conditions(
                self.tts,code_tensors,latents,[s['new_codes_count'] for s in live],phase=self.phase,
                streams=getattr(self,'condition_streams',()),
                graph_bank=getattr(self,'condition_graph_bank',None) if head else None,
                flat_projection=getattr(self,'condition_flat_projection',False) if head else False)
            self.condition_group_inventory=condition_groups
        groups = {}
        for condition_index,(s, codes, latent) in enumerate(zip(live, code_tensors, latents)):
            v = self.model.bank.get(s['case']['voice_id'])['values']

            def condition():
                if prepared_conditions is None:
                    semantic = self.tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2) + self.tts.s2mel.models['gpt_layer'](latent)
                    frames = int(s['new_codes_count'] * 1.72)
                    cond = self.tts.s2mel.models['length_regulator'](semantic, ylens=torch.tensor([frames], device='cuda'), n_quantizers=3, f0=None)[0]
                else:cond=prepared_conditions[condition_index]
                frames = cond.shape[1]
                core, horizon, decode = head_window(frames) if head else (frames, frames, frames)
                local = cond[:, :horizon]
                if local.shape[1] < decode:
                    local = torch.nn.functional.pad(local, (0, 0, 0, decode - local.shape[1]))
                mu = torch.cat((v['voice.cache_s2mel_prompt'], local), 1)
                plen = v['voice.cache_mel'].shape[-1]
                prompt = mu.new_zeros(1, 80, mu.shape[1])
                prompt[:, :, :plen] = v['voice.cache_mel']
                mask = torch.arange(mu.shape[1], device='cuda')[None, None] < plen
                length=(torch.tensor([plen+horizon],device='cuda')
                        if getattr(self,'condition_per_row_lengths',not hasattr(self,'unified_first_chunk'))
                        else plen+horizon)
                s['acoustic'] = (s['noise'][:, :, :mu.shape[1]].clone(), prompt, length, v['voice.cache_s2mel_style'], mu, mask)
                s['core'] = core
                s['horizon'] = horizon
                s['plen'] = plen
                return (mu.shape[1], plen)
            key = self.phase('condition', 1, condition)
            groups.setdefault(key, []).append(s)
        for key, members in groups.items():
            # One length-vector transfer per CFM group replaces a synchronous
            # host-to-device scalar construction for every request.
            args = tuple((torch.cat([s['acoustic'][i] for s in members])
                          if i != 2 or getattr(self,'condition_per_row_lengths',not hasattr(self,'unified_first_chunk'))
                          else torch.tensor([s['acoustic'][i] for s in members],device='cuda',dtype=torch.long))
                         for i in range(6))
            bg = len(members)
            graphs = self.head_graphs if head else None
            mel = self.phase('cfm' + str(self.steps), bg, lambda: (graphs.run_cfm(args,key[1]) if graphs is not None else self.student(*args))[:, :, key[1]:], frames=args[0].shape[-1], head=head)
            mel_for_vocoder=mel.float().contiguous()
            wave = self.phase('vocoder', bg, lambda: graphs.run_vocoder(mel_for_vocoder) if graphs is not None else self.vocoder(mel_for_vocoder), frames=mel.shape[-1], head=head).squeeze(1).clamp(-1, 1)
            if on_enqueued is not None:
                on_enqueued();on_enqueued=None
            staged=[]
            batch_pcm=(head and getattr(self,'batch_head_pcm',False)
                       and all(s['emitted']==0 and s['pending'] is None for s in members))
            if batch_pcm:
                from inspark_infer.runtime.head_pcm import stage_head_pcm
                batch_device,batch_host=stage_head_pcm(wave,[s['core']*HOP for s in members])
            for i, s in enumerate(members):
                begin = s['emitted']
                end = s['core'] * HOP
                chunk = wave[i, begin:end] if batch_pcm else wave[i, begin:end].clone()
                blend = 0
                if s['pending'] is not None:
                    blend = min(len(s['pending']), len(chunk), RIGHT * HOP)
                    if blend:
                        weight = 0.5 - 0.5 * torch.cos(torch.pi * torch.linspace(0, 1, blend, device='cuda'))
                        chunk[:blend] = s['pending'][:blend] * (1 - weight) + chunk[:blend] * weight
                s['pending'] = wave[i, s['core'] * HOP:s['horizon'] * HOP].clone() if head else None
                pcm_device=(batch_device[i,:end] if batch_pcm else (chunk*32767).to(torch.int16))
                pcm_host=(batch_host[i,:end] if batch_pcm else
                          torch.empty(pcm_device.shape,device='cpu',dtype=torch.int16,pin_memory=True))
                code_device=code_by_session[id(s)][0] if 'new_codes' not in s else None
                code_host=(torch.empty(code_device.shape,device='cpu',dtype=code_device.dtype,pin_memory=True)
                           if code_device is not None else None)
                staged.append((s,begin,end,blend,pcm_device,pcm_host,code_device,code_host))
            # One event joins the compact token metadata and already converted
            # PCM copies. A dedicated stream leaves the model stream available
            # for the next admitted group while output delivery waits.
            if not hasattr(self,'output_d2h_stream') or self.output_d2h_stream is None:
                self.output_d2h_stream=torch.cuda.Stream()
            current_stream=torch.cuda.current_stream()
            self.output_d2h_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.output_d2h_stream):
                copy_started=torch.cuda.Event(enable_timing=True)
                copy_done=torch.cuda.Event(enable_timing=True)
                copy_started.record()
                if batch_pcm:batch_host.copy_(batch_device,non_blocking=True)
                for _,_,_,_,pcm_device,pcm_host,code_device,code_host in staged:
                    if not batch_pcm:pcm_host.copy_(pcm_device,non_blocking=True)
                    if code_device is not None:code_host.copy_(code_device,non_blocking=True)
                copy_done.record()
            wait_started=time.perf_counter();copy_done.synchronize()
            self.output_d2h_wait_ms=getattr(self,'output_d2h_wait_ms',0.)+(time.perf_counter()-wait_started)*1000
            self.output_d2h_transfer_ms=getattr(self,'output_d2h_transfer_ms',0.)+copy_started.elapsed_time(copy_done)
            pcm_bytes=(batch_host.numel()*batch_host.element_size() if batch_pcm else
                       sum(pcm_host.numel()*pcm_host.element_size() for _,_,_,_,_,pcm_host,_,_ in staged))
            self.output_d2h_bytes=getattr(self,'output_d2h_bytes',0)+pcm_bytes+sum(
                code_host.numel()*code_host.element_size() if code_host is not None else 0
                for _,_,_,_,_,_,_,code_host in staged)
            for s,begin,end,blend,_,pcm_host,_,code_host in staged:
                if code_host is not None:
                    codes=code_host.tolist()
                    assert codes[:len(s['codes'])] == s['codes']
                    s['new_codes']=codes
                pcm=pcm_host.numpy().copy()
                ready = time.perf_counter()
                assert len(pcm) == end - begin
                assert len(pcm)>0 or (not head and s['eos'] and end==begin), 'Empty non-EOS speech'
                s['chunks'].append(dict(index=index, ready=ready, seconds=len(pcm) / 22050, sample_start=begin, sample_end=end, crossfade=blend, eos=s['eos'], head_before_eos=head and (not s['eos']), cfm_batch=bg, vocoder_batch=bg, pcm=pcm))
                s['emitted'] = end
                s['codes'] = s.pop('new_codes')
                s.pop('new_codes_count',None)
                s.pop('acoustic')
                if on_chunk is not None:
                    on_chunk(s)
        for row in rows:
            row.acoustic = None
        rt.events = []
