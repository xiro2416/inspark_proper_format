"""Persistent Draft context slots; dynamic history length without bulk packing."""
import torch
from .batch_draft import BatchedDraftBackbone
from acc_infer_clear.kernels.draft_attention import attention
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.runtime.graph_policy import batches,FIRST_KV_LIMITS

class DraftPool:
    def __init__(self,model,max_batch,capacity=2048):
        param=next(model.parameters());a=model.layers[0]
        self.capacity=capacity;self.max_slots=max_batch*2
        self.storage=param.new_empty(len(model.layers),2,self.max_slots,a.num_heads,capacity,a.head_dim)
        self.free=list(reversed(range(self.max_slots)));self.generations=[0]*self.max_slots
    def attach(self,cache):
        if hasattr(cache,'pool_slot'):self.check(cache);return
        if not self.free:raise RuntimeError('Draft slots exhausted')
        slot=self.free.pop();self.generations[slot]+=1
        cache.pool_slot=slot;cache.pool_generation=self.generations[slot];cache.pool_released=False
        cache.storage_keys=[self.storage[i,0,slot:slot+1] for i in range(self.storage.shape[0])]
        cache.storage_values=[self.storage[i,1,slot:slot+1] for i in range(self.storage.shape[0])]
        for dst,src in zip(cache.storage_keys,cache.keys):dst[:,:,:cache.length].copy_(src)
        for dst,src in zip(cache.storage_values,cache.values):dst[:,:,:cache.length].copy_(src)
    def check(self,cache):
        if cache.pool_released or self.generations[cache.pool_slot]!=cache.pool_generation:raise RuntimeError('Stale Draft slot')
    def release(self,cache):
        if hasattr(cache,'pool_slot') and not cache.pool_released:
            self.check(cache);cache.pool_released=True;self.free.append(cache.pool_slot)

class SlotDraft(BatchedDraftBackbone):
    def __init__(self,model,pool):
        super().__init__(model);self.pool=pool
        assert model.recent_context_window==0 and all(not hasattr(l,'attention_conv') for l in model.layers)
    def math(self,anchors,positions,slots,lengths,limit):
        m=self.model;hidden=m._noise_embeddings(anchors,positions);context_pos=lengths[:,None]+self.step
        for index,layer in enumerate(m.layers):
            norm=layer.input_norm(hidden)
            if index in getattr(self,'shared_qkv',{}):
                qq,kk,vv=self.shared_qkv[index](norm);q=layer._heads(qq);k=layer._heads(kk);v=layer._heads(vv)
            else:q=layer._heads(layer.q_proj(norm));k=layer._heads(layer.k_proj(norm));v=layer._heads(layer.v_proj(norm))
            if hasattr(layer,'rope_inv_freq'):q=layer.rotate(q,context_pos);k=layer.rotate(k,context_pos)
            attended=attention(q,k,v,self.pool.storage[index,0],self.pool.storage[index,1],slots,lengths,limit)
            hidden=hidden+layer.o_proj(attended.transpose(1,2).contiguous().view_as(hidden))
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
                self.graphs[b,limit]=capture(lambda aa,pp,ss,ll:self.math(aa,pp,ss,ll,limit),(anchors,positions,slots,lengths))
        return dict(keys=[list(k) for k in self.graphs],history_packing=False,online_capture=False)
    def __call__(self,jobs):
        for j in jobs:self.pool.check(j['cache'])
        lengths=[j['cache'].length for j in jobs];limit=next(n for n in (64,128,256,512,1024,2048) if n>=max(lengths)+7)
        device=self.pool.storage.device
        slots=torch.tensor([j['cache'].pool_slot for j in jobs],device=device,dtype=torch.int32)
        lens=torch.tensor(lengths,device=device,dtype=torch.int32)
        positions=torch.tensor([j['first_position'] for j in jobs],device=device)[:,None]+self.step
        anchors=torch.cat([j['anchor_token'].reshape(1) for j in jobs]);b=len(jobs)
        if (b,limit) in self.graphs:hidden,base=self.graphs[b,limit](anchors,positions,slots,lens);self.graph_hits+=1
        else:hidden,base=self.math(anchors,positions,slots,lens,limit)
        self.calls+=1;self.rows+=b
        return [(hidden[i:i+1].clone(),base[i:i+1].clone()) for i in range(b)]
    def stats(self):return dict(calls=self.calls,rows=self.rows,graph_hits=self.graph_hits,history_packing=False)
