"""Device-controlled fixed-graph-batch first-head AR loop.

Acceptance decisions, residual selection, token/past/context state and ready
state remain on GPU. The host reads one all-ready scalar per round. This module
is opt-in and intentionally does not replace the generic Runtime._step.
"""
import os
import torch
from acc_infer_clear.kernels.acceptance import acceptance,prefix_plan
from acc_infer_clear.kernels.device_commit import commit,mark_keep,status

class DeviceRoundHead:
    def __init__(self,runtime,rows,max_tokens):
        if len(rows) not in (1,2,3,4,5,6,7,8,16,32):raise ValueError('Unsupported device batch')
        if not runtime.accept.fused or runtime.residual.device_failures is None:
            raise RuntimeError('Device head requires fused acceptance and device residual')
        self.rt,self.rows,self.max_tokens=runtime,rows,int(max_tokens);self.device=runtime.device;self.b=len(rows);self.context_extent=self.b*8
        if not runtime.context.graph_scatter or self.context_extent not in runtime.context.graphs:
            raise RuntimeError('Device head requires matching Context scatter graph')
        if (self.b,128) not in runtime.backbone.graphs or (self.b,128) not in runtime.target.graphs or self.b not in runtime.proposal.graphs:
            raise RuntimeError('Missing component Graph for device batch')
        self.eos=int(runtime.engine.target.gpt.stop_mel_token);self.capacity=max_tokens+64
        self.token_buffer=torch.zeros(self.b,self.capacity,device=self.device,dtype=torch.long)
        self.token_lengths=torch.tensor([len(r.codes) for r in rows],device=self.device,dtype=torch.int32)
        for i,row in enumerate(rows):self.token_buffer[i,:len(row.codes)]=torch.cat(row.codes)
        self.past=torch.tensor([r.past_length for r in rows],device=self.device,dtype=torch.int32)
        self.draft_lengths=torch.tensor([r.cache.length for r in rows],device=self.device,dtype=torch.int32)
        self.mel=torch.tensor([r.mel_length for r in rows],device=self.device,dtype=torch.int32)
        self.target_slots=torch.tensor([r.kv.slot for r in rows],device=self.device,dtype=torch.int32)
        self.draft_slots=torch.tensor([r.cache.pool_slot for r in rows],device=self.device,dtype=torch.int32)
        self.last=torch.cat([r.codes[-1] for r in rows]).long();self.done=torch.tensor([r.done for r in rows],device=self.device)
        self.ready=self.done|(self.token_lengths*172>=5200)
        self.rounds=torch.zeros(self.b,device=self.device,dtype=torch.int32)
        self.accepted=torch.full((self.b,self.capacity),-1,device=self.device,dtype=torch.int32)
        self.committed=torch.zeros(self.b,device=self.device,dtype=torch.int32)
        # Context graphs are captured once with metadata sized for max_batch,
        # even when the token bucket represents a smaller exact batch.
        meta_size=runtime.context.pool.max_slots//2
        if self.b>meta_size:raise RuntimeError('Device batch exceeds Context metadata capacity')
        self.source=torch.zeros(meta_size,device=self.device,dtype=torch.int32)
        self.source[:self.b]=torch.arange(self.b,device=self.device,dtype=torch.int32)*8
        self.context_lengths=torch.zeros(meta_size,device=self.device,dtype=torch.int32)
        self.context_slots=torch.zeros(meta_size,device=self.device,dtype=torch.int32)
        self.context_slots[:self.b].copy_(self.draft_slots)
        self.context_destinations=torch.zeros(meta_size,device=self.device,dtype=torch.int32)
        self.status=torch.zeros((),device=self.device,dtype=torch.int32)
        self.generator=torch.Generator(device=self.device).manual_seed(0xD3C0A117)
        self.step7=torch.arange(7,device=self.device)[None];self.step8=torch.arange(8,device=self.device)[None]
        bank=getattr(runtime.target,'native_full_bank',None);slot_values=[r.kv.slot for r in rows]
        self.native_target=(bank if bank is not None and
            bank.eligible(self.b,slot_values,max(r.past_length for r in rows)) else None)
        self.native_draft_eligible=self.rt.backbone.native_eligible(
            self.b,[r.cache.pool_slot for r in rows],max(r.cache.length for r in rows))

    def step(self,use_child_graphs=True):
        active=~self.ready;first=self.past+1-self.mel
        draft_args=(self.last,first[:,None]+self.step7,self.draft_slots,self.draft_lengths)
        if use_child_graphs:
            bank=getattr(self.rt.backbone,'native_full_bank',None)
            native=(self.native_draft_eligible and bank is not None and (self.b,128) in bank.graphs and
                    self.rt.backbone.graphs[self.b,128] is bank.graphs[self.b,128])
            cache_before=(self.rt.backbone.compare_native_cache(self.draft_slots,self.draft_lengths)
                          if native and os.environ.get('ACC_COMPARE_NATIVE_DRAFT')=='1' else None)
            selected_graph=self.rt.backbone.graphs[self.b,128]
            graph_is_native=bank is not None and selected_graph is bank.graphs.get((self.b,128))
            if graph_is_native and not native:
                hidden,base=self.rt.backbone.math(*draft_args,128)
            else:hidden,base=selected_graph(*draft_args)
            self.rt.backbone.device_steps=getattr(self.rt.backbone,'device_steps',0)+1
            if native:
                self.rt.backbone.native_full_steps+=1
                if os.environ.get('ACC_COMPARE_NATIVE_DRAFT')=='1':
                    self.rt.backbone.compare_native_result(*draft_args,128,hidden,base,cache_before)
        else:hidden,base=self.rt.backbone.math(*draft_args,128)
        noise=torch.empty_like(base).exponential_(generator=self.rt.proposal.batch_generator)
        if use_child_graphs:proposed,p,ll=self.rt.proposal.graphs[self.b](hidden,base,noise,self.last)
        else:proposed,p,ll=self.rt.proposal.captured_math(hidden,base,noise,self.last)
        tokens=torch.cat((self.last[:,None],proposed),1);positions=first[:,None]+self.step8
        tm=self.rt.engine.target.model;x=tm.embeddings(tokens)+tm.text_pos_embedding.emb(positions)
        mark_keep(self.rt.target.keep,self.target_slots,self.past)
        self.rt.device_target_steps=getattr(self.rt,'device_target_steps',0)+1
        if use_child_graphs and self.native_target is not None:
            self.rt.native_target_steps+=1
            logits,selected,final=self.native_target.graphs[self.b,128](x,self.target_slots,self.past)
        elif use_child_graphs:logits,selected,final=self.rt.target.graphs[self.b,128](x,self.target_slots,self.past)
        else:logits,selected,final=self.rt.target.math(x,self.target_slots,self.past,128)
        gd=torch.rand((self.b,7),device=self.device,generator=self.generator)
        ad=torch.rand((self.b,7),device=self.device,generator=self.generator)
        q,_,packed=acceptance(self.rt.engine.dense_groups,.8,logits[:,:7],p,proposed,gd,ad)
        remaining=(self.max_tokens-self.token_lengths).clamp(0,7)
        counts,ends,corrections,residual_mask=prefix_plan(packed,remaining,self.token_lengths,self.eos,self.max_tokens)
        residual_token,_=self.rt.residual.device_batch(q,p,counts,residual_mask&active)
        row=torch.arange(self.b,device=self.device);at=counts.long().clamp(0,7)
        target_probability=torch.softmax(logits[row,at].float()/.8,-1)
        target_token=torch.multinomial(target_probability,1,generator=self.generator).squeeze(1)
        correction=torch.where(residual_mask,residual_token,target_token)
        commit(proposed,correction,counts,ends,corrections,self.token_buffer,
               self.token_lengths,self.past,self.accepted,self.rounds,self.done,
               self.committed,self.last,self.eos,self.max_tokens,active)
        prepared=self.rt.engine.draft.prepare_context(selected,final).reshape(1,self.context_extent,-1)
        context_positions=(self.draft_lengths[:,None]+self.step8).reshape(1,self.context_extent)
        self.context_lengths[:self.b].copy_(self.committed)
        self.context_destinations[:self.b].copy_(self.draft_lengths)
        context_args=(prepared,context_positions,self.source,self.context_lengths,
                      self.context_slots,self.context_destinations)
        if use_child_graphs:self.rt.context.graphs[self.context_extent](*context_args)
        else:self.rt.context._project_scatter(*context_args)
        self.draft_lengths.add_(self.committed)
        self.ready.copy_(self.done|(self.token_lengths*172>=5200))

    def run(self,max_rounds=64):
        launched=0;initial_failures=int(self.rt.residual.device_failures.item());self.failed=False;self.fallback_reason=None
        while True:
            status(self.ready,self.rt.residual.device_failures,self.status,initial_failures,
                   self.past,self.draft_lengths);value=int(self.status.item())
            if value&2:self.failed=True;self.fallback_reason='residual';return 0
            if value&1:break
            if value&4:
                self.fallback_reason='kv_capacity'
                self.finish()
                return launched
            if launched>=max_rounds:raise RuntimeError('Device head exceeded round limit')
            self.step();launched+=1
        torch.cuda.synchronize();self.finish();return launched

    def finish(self):
        if self.native_target is not None:
            self.native_target.export(self.b,self.rt.target.storage)
        lengths=self.token_lengths.cpu().tolist();past=self.past.cpu().tolist();draft=self.draft_lengths.cpu().tolist()
        rounds=self.rounds.cpu().tolist();accepted=self.accepted.cpu();done=self.done.cpu().tolist()
        for i,row in enumerate(self.rows):
            row.codes=[self.token_buffer[i,j:j+1] for j in range(lengths[i])]
            row.accepted=accepted[i,:rounds[i]].tolist();row.past_length=past[i];row.kv.length=past[i]
            row.cache.length=draft[i]
            for layer in range(len(row.cache.keys)):
                row.cache.keys[layer]=row.cache.storage_keys[layer][:,:,:draft[i]]
                row.cache.values[layer]=row.cache.storage_values[layer][:,:,:draft[i]]
            row.mask=self.rt.target.keep[row.kv.slot:row.kv.slot+1,:past[i]].clone();row.done=done[i]
