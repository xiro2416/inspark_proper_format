"""Explicit streaming acoustic protocol. No execution-backend switches."""
import time
from dataclasses import dataclass
from acc_infer_clear.models.indextts2.audio import head_ready, head_window, HOP, RIGHT
from acc_infer_clear.runtime.splitter import split

@dataclass
class AcousticOp:
    kind: str
    args: tuple
    kwargs: dict

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
        from acc_infer_clear.runtime.indextts2.core import BatchRow
        
        
        torch = self.torch
        rt = self.rt
        head = index == 0
        rows = []
        owner = {}
        b = len(sessions)
        requests = []
        for s in sessions:
            c = s['case']
            requests.append(dict(id=c['id'], text=s['text']+s['parts'][index], voice_id=c['voice_id'], seed=c['seed'], emotion=c['emotion'], arrival=s['arrival']))
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
                row.codes = [torch.tensor([c], device='cuda', dtype=torch.long) for c in s['codes']] + [rt.sample(row, logits[:, -1])]
                ctx.append(((row.cache, selected, final), {}))
            rt.context(ctx)
            for row in rows:
                row.done = int(row.codes[-1].item()) == int(rt.engine.target.gpt.stop_mel_token)
        rng_states=[s['gen'].get_state() for s in sessions]
        try:
            self.phase('prefill_rebuild' if not head else 'prefill', b, prefill)
        except Exception as error:
            from acc_infer_clear.runtime.cleanup import cleanup_all
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
        import numpy as np
        torch, rt = self.torch, self.rt
        head = index == 0
        live = []
        code_tensors = []
        ops = []
        eos = int(rt.engine.target.gpt.stop_mel_token)
        for row in rows:
            s = owner[id(row)]
            codes = (self.phase('speech_codes_d2h',1,lambda:torch.cat(row.codes).cpu().tolist())
                     if getattr(self,'trace_ranges',False) else torch.cat(row.codes).cpu().tolist())
            ended = bool(codes and codes[-1] == eos)
            if row.done and (not ended):
                s['error'] = 'No EOS before original1500 total-token cap'
                self.failures.append(dict(id=s['case']['id'], segment=index, error=s['error']))
                continue
            codes = codes[:-1] if ended else codes
            if not codes or (len(codes) == len(s['codes']) and not ended):
                s['error'] = 'Empty speech continuation'
                self.failures.append(dict(id=s['case']['id'], segment=index, error=s['error']))
                continue
            assert codes[:len(s['codes'])] == s['codes']
            s['new_codes'] = codes
            s['eos'] = ended
            s['accepted'] += row.accepted
            s['kv_head_lengths'] = row.past_length
            seg = row.prepared.prepared_segments[0]
            v = self.model.bank.get(s['case']['voice_id'])['values']
            current = torch.tensor([codes], device='cuda', dtype=torch.long)
            args = (seg.speech_latent, seg.text_ids, torch.tensor([seg.text_ids.shape[-1]], device='cuda'), current, torch.tensor([len(codes)], device='cuda'), v['voice.cache_emo_cond'])
            kw = dict(cond_mel_lengths=torch.tensor([v['voice.cache_spk_cond'].shape[-1]], device='cuda'), emo_cond_mel_lengths=torch.tensor([v['voice.cache_emo_cond'].shape[-1]], device='cuda'), emo_vec=row.prepared.emovec, use_speed=torch.zeros(1, device='cuda', dtype=torch.long))
            live.append(s)
            code_tensors.append(current)
            ops.append(AcousticOp('latent', args, kw))
        if not live:
            return
        latents = self.phase('latent', len(live), lambda: rt._latents(ops))
        groups = {}
        for s, codes, latent in zip(live, code_tensors, latents):
            v = self.model.bank.get(s['case']['voice_id'])['values']

            def condition():
                semantic = self.tts.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2) + self.tts.s2mel.models['gpt_layer'](latent)
                frames = int(len(s['new_codes']) * 1.72)
                cond = self.tts.s2mel.models['length_regulator'](semantic, ylens=torch.tensor([frames], device='cuda'), n_quantizers=3, f0=None)[0]
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
                s['acoustic'] = (s['noise'][:, :, :mu.shape[1]].clone(), prompt, torch.tensor([plen + horizon], device='cuda'), v['voice.cache_s2mel_style'], mu, mask)
                s['core'] = core
                s['horizon'] = horizon
                s['plen'] = plen
                return (mu.shape[1], plen)
            key = self.phase('condition', 1, condition)
            groups.setdefault(key, []).append(s)
        for key, members in groups.items():
            args = tuple((torch.cat([s['acoustic'][i] for s in members]) for i in range(6)))
            bg = len(members)
            graphs = self.head_graphs if head else None
            mel = self.phase('cfm' + str(self.steps), bg, lambda: (graphs.run_cfm(args,key[1]) if graphs is not None else self.student(*args))[:, :, key[1]:], frames=args[0].shape[-1], head=head)
            wave = self.phase('vocoder', bg, lambda: graphs.run_vocoder(mel.float()) if graphs is not None else self.vocoder(mel.float()), frames=mel.shape[-1], head=head).squeeze(1).clamp(-1, 1)
            if on_enqueued is not None:
                on_enqueued();on_enqueued=None
            for i, s in enumerate(members):
                begin = s['emitted']
                end = s['core'] * HOP
                chunk = wave[i, begin:end].clone()
                blend = 0
                if s['pending'] is not None:
                    blend = min(len(s['pending']), len(chunk), RIGHT * HOP)
                    if blend:
                        weight = 0.5 - 0.5 * torch.cos(torch.pi * torch.linspace(0, 1, blend, device='cuda'))
                        chunk[:blend] = s['pending'][:blend] * (1 - weight) + chunk[:blend] * weight
                s['pending'] = wave[i, s['core'] * HOP:s['horizon'] * HOP].clone() if head else None
                pcm = (self.phase('pcm_d2h',1,lambda:chunk.cpu().numpy().copy())
                       if getattr(self,'trace_ranges',False) else chunk.cpu().numpy().copy())
                ready = time.perf_counter()
                assert len(pcm) == end - begin
                assert len(pcm)>0 or (not head and s['eos'] and end==begin), 'Empty non-EOS speech'
                s['chunks'].append(dict(index=index, ready=ready, seconds=len(pcm) / 22050, sample_start=begin, sample_end=end, crossfade=blend, eos=s['eos'], head_before_eos=head and (not s['eos']), cfm_batch=bg, vocoder_batch=bg, pcm=(pcm * 32767).astype(np.int16)))
                s['emitted'] = end
                s['codes'] = s.pop('new_codes')
                s.pop('acoustic')
                if on_chunk is not None:
                    on_chunk(s)
        for row in rows:
            row.acoustic = None
        rt.events = []
