"""Persistent Draft context slots; dynamic history length without bulk packing."""
import os
import torch
from contextlib import nullcontext
from acc_infer_clear.runtime.indextts2.batch_draft import BatchedDraftBackbone
from acc_infer_clear.ops.triton.draft_attention import attention
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.runtime.graph_policy import batches, FIRST_KV_LIMITS

class DraftPool:
    def __init__(self,model,max_batch,capacity=2048):
        param=next(model.parameters());a=model.layers[0]
        self.capacity=capacity;self.max_slots=max_batch*2
        self.storage=param.new_empty(len(model.layers),2,self.max_slots,a.num_heads,capacity,a.head_dim)
        self.free=list(reversed(range(self.max_slots)));self.generations=[0]*self.max_slots;self.native_bank=None
        self.leased=[False]*self.max_slots
    def attach(self,cache):
        if hasattr(cache,'pool_slot'):self.check(cache);return
        if not self.free:raise RuntimeError('Draft slots exhausted')
        if cache.length>self.capacity:raise ValueError('Draft context capacity exceeded')
        slot=self.free.pop();self.generations[slot]+=1
        self.leased[slot]=True
        names=('pool_slot','pool_generation','pool_released','pool_owner','storage_keys','storage_values')
        previous={name:getattr(cache,name) for name in names if hasattr(cache,name)}
        try:
            cache.pool_slot=slot;cache.pool_generation=self.generations[slot];cache.pool_released=False;cache.pool_owner=self
            cache.storage_keys=[self.storage[i,0,slot:slot+1] for i in range(self.storage.shape[0])]
            cache.storage_values=[self.storage[i,1,slot:slot+1] for i in range(self.storage.shape[0])]
            for dst,src in zip(cache.storage_keys,cache.keys):dst[:,:,:cache.length].copy_(src)
            for dst,src in zip(cache.storage_values,cache.values):dst[:,:,:cache.length].copy_(src)
            if self.native_bank is not None:self.native_bank.import_slot(cache,slot)
        except Exception:
            for name in names:
                if name in previous:setattr(cache,name,previous[name])
                elif hasattr(cache,name):delattr(cache,name)
            self.leased[slot]=False;self.free.append(slot);self.free.sort(reverse=True)
            raise
    def check(self,cache):
        if getattr(cache,'pool_owner',None) is not self:raise ValueError('Foreign Draft slot')
        if cache.pool_released or self.generations[cache.pool_slot]!=cache.pool_generation or not self.leased[cache.pool_slot]:raise RuntimeError('Stale Draft slot')
    def release(self,cache):
        if hasattr(cache,'pool_slot') and not cache.pool_released:
            if getattr(cache,'pool_owner',None) is not self:raise ValueError('Foreign Draft slot')
            if self.generations[cache.pool_slot]!=cache.pool_generation:raise RuntimeError('Stale Draft slot')
            cache.pool_released=True
            if self.leased[cache.pool_slot]:
                self.leased[cache.pool_slot]=False;self.free.append(cache.pool_slot);self.free.sort(reverse=True)

class SlotDraft(BatchedDraftBackbone):
    def __init__(self,model,pool,consumer_layout=False):
        super().__init__(model);self.pool=pool
        assert model.recent_context_window==0 and all(not hasattr(l,'attention_conv') for l in model.layers)
        self.consumer_layout=bool(consumer_layout)
        self.trace_subcomponents=False
        self.native_full_bank=None
        self.native_full_steps=0
        self.native_compare=[]
    def native_eligible(self,batch,slot_values,max_length):
        bank=self.native_full_bank
        return (bank is not None and (batch,128) in bank.graphs and
                slot_values==list(range(batch)) and max_length+8<=128)
    def _profile_scope(self,name):
        return torch.profiler.record_function(name) if self.trace_subcomponents else nullcontext()
    def math(self,anchors,positions,slots,lengths,limit):
        m=self.model;hidden=m._noise_embeddings(anchors,positions);context_pos=lengths[:,None]+self.step
        for index,layer in enumerate(m.layers):
            norm=layer.input_norm(hidden)
            with self._profile_scope('draft/qkv_projection'):
                if index in getattr(self,'shared_qkv',{}):qq,kk,vv=self.shared_qkv[index](norm)
                else:qq,kk,vv=layer.q_proj(norm),layer.k_proj(norm),layer.v_proj(norm)
            with self._profile_scope('draft/qkv_output_layout'):
                q=layer._heads(qq);k=layer._heads(kk);v=layer._heads(vv)
            if hasattr(layer,'rope_inv_freq'):
                with self._profile_scope('draft/rope'):
                    q=layer.rotate(q,context_pos);k=layer.rotate(k,context_pos)
            attended=attention(q,k,v,self.pool.storage[index,0],self.pool.storage[index,1],slots,lengths,limit,self.consumer_layout)
            with self._profile_scope('draft/attention_output_layout'):
                consumer=(attended.reshape_as(hidden) if self.consumer_layout else attended.transpose(1,2).contiguous().view_as(hidden))
            hidden=hidden+layer.o_proj(consumer)
            hidden=hidden+layer.mlp(layer.post_norm(hidden))
            hidden=m.apply_query_temporal(hidden,index)
        hidden=m.project_output(hidden)
        return hidden,m.base_logits(hidden)
    def prepare_graphs(self,max_batch):
        self.pool.storage.zero_();param=next(self.model.parameters())
        for b in batches(max_batch):
            anchors=torch.zeros(b,device=param.device,dtype=torch.long)
            positions=torch.arange(7,device=param.device)[None].expand(b,-1).clone()
            slots=torch.arange(b,device=param.device,dtype=torch.int32)
            for limit in FIRST_KV_LIMITS:
                lengths=torch.full((b,),limit-7,device=param.device,dtype=torch.int32)
                if self.native_full_bank is not None and (b,limit) in self.native_full_bank.graphs:
                    self.graphs[b,limit]=self.native_full_bank.graphs[b,limit]
                else:self.graphs[b,limit]=capture(lambda aa,pp,ss,ll:self.math(aa,pp,ss,ll,limit),(anchors,positions,slots,lengths))
        return dict(keys=[list(k) for k in self.graphs],history_packing=False,online_capture=False)
    def __call__(self,jobs):
        for j in jobs:self.pool.check(j['cache'])
        lengths=[j['cache'].length for j in jobs];limit=next(n for n in (64,128,256,512,1024,2048) if n>=max(lengths)+7)
        device=self.pool.storage.device
        slot_values=[j['cache'].pool_slot for j in jobs]
        slots=torch.tensor(slot_values,device=device,dtype=torch.int32)
        lens=torch.tensor(lengths,device=device,dtype=torch.int32)
        positions=torch.tensor([j['first_position'] for j in jobs],device=device)[:,None]+self.step
        anchors=torch.cat([j['anchor_token'].reshape(1) for j in jobs]);b=len(jobs)
        native_graph=(self.native_full_bank is not None and (b,limit) in self.native_full_bank.graphs)
        native=native_graph and self.native_eligible(b,slot_values,max(lengths))
        if (b,limit) in self.graphs and (not native_graph or native):
            hidden,base=self.graphs[b,limit](anchors,positions,slots,lens);self.graph_hits+=1
            if native:
                self.native_full_steps+=1
        else:hidden,base=self.math(anchors,positions,slots,lens,limit)
        if native and os.environ.get('ACC_COMPARE_NATIVE_DRAFT')=='1':
            self.compare_native_result(anchors,positions,slots,lens,limit,hidden,base)
        self.calls+=1;self.rows+=b
        return [(hidden[i:i+1].clone(),base[i:i+1].clone()) for i in range(b)]

    def compare_native_cache(self,slots,lens):
        bank=self.native_full_bank.cache;cache_rows=[]
        for row,slot in enumerate(slots.tolist()):
            length=int(lens[row]);current=self.pool.storage[:,:,slot,:,:length].float()
            mirrored=bank[:,:,slot,:,:length].float();delta=(current-mirrored).abs()
            cache_rows.append(dict(slot=slot,length=length,max_abs=float(delta.max()) if delta.numel() else 0.0,
                                   mean_abs=float(delta.mean()) if delta.numel() else 0.0,
                                   current_mean_abs=float(current.abs().mean()) if current.numel() else 0.0,
                                   mirrored_mean_abs=float(mirrored.abs().mean()) if mirrored.numel() else 0.0))
        return cache_rows

    def compare_native_result(self,anchors,positions,slots,lens,limit,hidden,base,cache_before=None):
        """Compare the actual TRT result/cache against PyTorch at one live step."""
        with torch.inference_mode():
            nh,nb=hidden.clone(),base.clone()
            rh,rb=self.math(anchors,positions,slots,lens,limit)
            cache_rows=self.compare_native_cache(slots,lens)
            def compare(x,y):
                xf=x.float().flatten();yf=y.float().flatten();delta=(xf-yf).abs()
                return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                            cosine=float(torch.nn.functional.cosine_similarity(xf,yf,dim=0)))
            self.native_compare.append(dict(batch=int(anchors.shape[0]),limit=limit,slots=slots.tolist(),lengths=lens.tolist(),
                                            cache_before=cache_before,cache=cache_rows,
                                            hidden=compare(rh,nh),base=compare(rb,nb)))
    def stats(self):return dict(calls=self.calls,rows=self.rows,graph_hits=self.graph_hits,
                                native_full_steps=self.native_full_steps,native_compare=list(self.native_compare),
                                history_packing=False)
