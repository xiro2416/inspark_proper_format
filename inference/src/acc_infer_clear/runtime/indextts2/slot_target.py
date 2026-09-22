"""Persistent Target KV with direct ragged attention and explicit offline graphs."""
import torch
from contextlib import nullcontext
from acc_infer_clear.runtime.indextts2.batch_target import BatchedTarget, RequestKV
from acc_infer_clear.ops.triton.kv_attention import append, attention
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.runtime.graph_policy import batches, FIRST_KV_LIMITS

class SlotKV(RequestKV):
    def __init__(self,pool,slot,length,generation):
        super().__init__(pool.storage[:,:,slot:slot+1],length)
        self.pool=pool;self.slot=slot;self.generation=generation;self.released=False
    def check(self):
        if self.released or self.pool.generations[self.slot]!=self.generation or not self.pool.leased[self.slot]:raise RuntimeError('Stale KV slot')
    def crop(self,length):self.check();super().crop(length)

class SlotTarget(BatchedTarget):
    def __init__(self,target,max_batch=8,capacity=2048,consumer_layout=False):
        super().__init__(target)
        body=target.model.transformer;param=next(body.parameters());attn=body.h[0].attn
        assert attn.head_dim==64 and not body.config.add_cross_attention
        self.capacity=capacity;self.max_batch=max_batch;self.max_slots=2*max_batch
        # One in-flight tail batch may be suspended while a head batch is served.
        self.storage=param.new_empty(len(body.h),2,self.max_slots,attn.num_heads,capacity,attn.head_dim)
        self.keep=torch.zeros(self.max_slots,capacity,device=param.device,dtype=torch.int32)
        self.free=list(reversed(range(self.max_slots)));self.generations=[0]*self.max_slots
        self.leased=[False]*self.max_slots
        self.pool_import=self.import_cache;self.graphs={};self.graph_hits=0;self.graph_sealed=False
        self.consumer_layout=bool(consumer_layout)
        self.trace_subcomponents=False
        self.native_attention=None;self.native_graphs={};self.native_graph_hits=0;self.native_full_bank=None
        self.native_full_steps=0
    def _profile_scope(self,name):
        return torch.profiler.record_function(name) if self.trace_subcomponents else nullcontext()
    def import_cache(self,packed,row,length):
        if not self.free:raise RuntimeError('Target KV slots exhausted')
        if length+8>self.capacity:raise ValueError('Target KV capacity exceeded')
        slot=self.free.pop();self.generations[slot]+=1
        self.leased[slot]=True
        try:
            self.storage[:,:,slot,:,:length].copy_(packed[:,:,row,:,:length])
            if self.native_attention is not None:
                self.native_attention.import_slot(packed,row,slot,length)
            if self.native_full_bank is not None:
                self.native_full_bank.import_slot(packed,row,slot,length)
        except Exception:
            self.leased[slot]=False;self.free.append(slot);self.free.sort(reverse=True)
            raise
        return SlotKV(self,slot,length,self.generations[slot])
    def prefill(self,jobs):
        outputs=super().prefill(jobs)
        try:
            for output,(_,mask) in zip(outputs,jobs):
                kv=output[1];self.keep[kv.slot].zero_();self.keep[kv.slot,:kv.length].copy_(mask[0])
        except Exception as error:
            from acc_infer_clear.runtime.cleanup import cleanup_all
            cleanup_all([('prefill Target KV',lambda kv=out[1]:self.release(kv)) for out in outputs],primary=error)
            raise
        return outputs
    def release(self,kv):
        if not isinstance(kv,SlotKV):return
        if kv.pool is not self:raise ValueError('Foreign KV')
        if kv.released:return
        if self.generations[kv.slot]!=kv.generation:raise RuntimeError('Stale KV slot')
        kv.released=True
        if self.leased[kv.slot]:
            self.leased[kv.slot]=False;self.free.append(kv.slot);self.free.sort(reverse=True)
    def math(self,x,slots,lengths,limit):
        tm=self.target.model;model=tm.transformer;selected=[];hidden=x
        # Input already contains original absolute embeddings; this model's wpe is null.
        for index,block in enumerate(model.h):
            normalized=block.ln_1(hidden);a=block.attn
            with self._profile_scope('target/qkv_projection'):
                qkv=a.c_attn(normalized)
            with self._profile_scope('target/qkv_output_layout'):
                q,k,v=qkv.split(a.split_size,dim=2)
                q=q.view(*q.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
                k=k.view(*k.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
                v=v.view(*v.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
            with self._profile_scope('target/kv_append'):
                append(k,v,self.storage[index,0],self.storage[index,1],slots,lengths)
            out=attention(q,self.storage[index,0],self.storage[index,1],self.keep,slots,lengths,limit,self.consumer_layout)
            with self._profile_scope('target/attention_output_layout'):
                out=(out.reshape(x.shape[0],8,a.embed_dim) if self.consumer_layout else
                     out.transpose(1,2).contiguous().view(x.shape[0],8,a.embed_dim))
            hidden=hidden+a.c_proj(out)
            hidden=hidden+block.mlp(block.ln_2(hidden))
            if index in self.target.target_layer_ids:selected.append(hidden)
        final=model.ln_f(hidden)
        return tm.lm_head(final),torch.cat(selected,dim=-1),final
    def prepare_graphs(self):
        if len(self.free)!=self.max_slots:raise RuntimeError('Capture before admitting requests')
        if self.graph_sealed:raise RuntimeError('Already captured')
        self.storage.zero_();self.keep.fill_(1)
        for b in batches(self.max_batch):
            slots=torch.arange(b,device=self.storage.device,dtype=torch.int32)
            x=self.storage.new_zeros(b,8,self.target.model.transformer.embed_dim)
            for limit in FIRST_KV_LIMITS:
                lengths=torch.full((b,),limit-8,device=x.device,dtype=torch.int32)
                self.graphs[b,limit]=capture(lambda xx,ss,ll:self.math(xx,ss,ll,limit),(x,slots,lengths))
        self.graph_sealed=True
    def attach_native_attention(self,backend):
        if self.graph_sealed:raise RuntimeError('Attach native attention before graph capture')
        self.native_attention=backend
    def attach_native_full_bank(self,bank):
        self.native_full_bank=bank
    def native_math(self,x,slots,lengths,limit):
        if limit!=128:raise ValueError('TRT 11.3 first-head attention is K=128 only')
        tm=self.target.model;model=tm.transformer;selected=[];hidden=x
        for index,block in enumerate(model.h):
            normalized=block.ln_1(hidden);a=block.attn;qkv=a.c_attn(normalized)
            out=self.native_attention.run(index,qkv,self.keep,slots,lengths)
            consumer=(out.reshape(x.shape[0],8,a.embed_dim) if self.consumer_layout else
                      out.transpose(1,2).contiguous().view(x.shape[0],8,a.embed_dim))
            hidden=hidden+a.c_proj(consumer)
            hidden=hidden+block.mlp(block.ln_2(hidden))
            if index in self.target.target_layer_ids:selected.append(hidden)
        final=model.ln_f(hidden)
        return tm.lm_head(final),torch.cat(selected,dim=-1),final
    def prepare_native_graphs(self,batch_values):
        if self.native_attention is None:raise RuntimeError('Native attention is not attached')
        if len(self.free)!=self.max_slots:raise RuntimeError('Capture before admitting requests')
        param=next(self.target.model.parameters())
        for b in batch_values:
            if b not in self.native_attention.engines:continue
            slots=torch.arange(b,device=param.device,dtype=torch.int32)
            lengths=torch.full((b,),120,device=param.device,dtype=torch.int32)
            x=param.new_zeros(b,8,self.target.model.transformer.embed_dim)
            self.native_graphs[b,128]=capture(lambda xx,ss,ll:self.native_math(xx,ss,ll,128),(x,slots,lengths))
        return dict(keys=[list(k) for k in self.native_graphs],backend='TensorRT 11.3 native attention')
    def __call__(self,jobs):
        if not jobs:return []
        for x,kv,mask,pos in jobs:
            if not isinstance(kv,SlotKV) or kv.pool is not self:raise ValueError('Foreign KV')
            kv.check()
            if x.shape[1]!=8 or kv.length+8>self.capacity:raise ValueError('Invalid verify extent')
            self.keep[kv.slot,kv.length:kv.length+8].copy_(mask[0,-8:])
        lengths=[kv.length for x,kv,mask,pos in jobs]
        limit=next(n for n in (64,128,256,512,1024,2048) if n>=max(lengths)+8)
        slot_values=[kv.slot for x,kv,mask,pos in jobs]
        slots=torch.tensor(slot_values,device=self.storage.device,dtype=torch.int32)
        lens=torch.tensor(lengths,device=self.storage.device,dtype=torch.int32)
        x=torch.cat([j[0] for j in jobs]);key=(len(jobs),limit)
        bank=self.native_full_bank
        if self.graph_sealed and bank is not None and bank.eligible(len(jobs),slot_values,max(lengths)):
            logits,selected,final=bank.run_with_canonical_cache(x,slots,lens,slot_values,lengths)
            self.native_full_steps+=1;self.graph_hits+=1
        elif self.graph_sealed and key in self.graphs:
            logits,selected,final=self.graphs[key](x,slots,lens);self.graph_hits+=1
        else:logits,selected,final=self.math(x,slots,lens,limit)
        result=[]
        for i,(_,kv,_,_) in enumerate(jobs):
            result.append((logits[i:i+1].clone(),SlotKV(self,kv.slot,kv.length+8,kv.generation),selected[i:i+1].clone(),final[i:i+1].clone()))
        self.calls+=1;self.rows+=len(jobs);self.shapes[str(key)]=self.shapes.get(str(key),0)+1
        return result
    def stats(self):
        return dict(super().stats(),graph_hits=self.graph_hits,graph_keys=[list(k) for k in self.graphs],
                    native_full_steps=self.native_full_steps,persistent_kv=True,
                    history_pack_bytes_per_verify=0,
                    native_cache_sync='canonical prefix import and K128 export per host-scheduled native verify')
