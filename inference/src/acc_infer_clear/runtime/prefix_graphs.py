"""Bounded head prefill/latent graphs; longer tails explicitly use eager math."""
import torch
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.runtime.graph_policy import batches

class PackedKV:
    def __init__(self,packed):self.packed=packed
    def __len__(self):return self.packed.shape[0]
    def __getitem__(self,i):return self.packed[i,0],self.packed[i,1]

class PrefixGraphs:
    def __init__(self,engine):
        self.target=engine.rt.engine.target;self.gpt=engine.tts.gpt
        self.prefill_graphs={};self.latent_graphs={};self.hits={'prefill':0,'latent':0};self.tail_eager={'prefill':0,'latent':0}
    def body(self,x,keep,need_cache):
        tm=self.target.model;model=tm.transformer;hidden=x;selected=[];caches=[]
        length=x.shape[1];positions=torch.arange(length,device=x.device)
        mask=(positions[None,:]<=positions[:,None])[None,None]&keep[:,None,None,:].bool()
        for index,block in enumerate(model.h):
            a=block.attn;q,k,v=a.c_attn(block.ln_1(hidden)).split(a.split_size,dim=-1)
            q=q.view(x.shape[0],length,a.num_heads,a.head_dim).transpose(1,2)
            k=k.view(x.shape[0],length,a.num_heads,a.head_dim).transpose(1,2)
            v=v.view(x.shape[0],length,a.num_heads,a.head_dim).transpose(1,2)
            attn=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            hidden=hidden+a.c_proj(attn.transpose(1,2).contiguous().view_as(x))
            hidden=hidden+block.mlp(block.ln_2(hidden))
            if need_cache:
                caches.append(torch.stack((k,v)))
                if index in self.target.target_layer_ids:selected.append(hidden)
        final=model.ln_f(hidden)
        if not need_cache:return self.gpt.final_norm(final)
        last=(keep.long()*positions[None]).amax(-1)
        last_hidden=final.gather(1,last[:,None,None].expand(-1,1,final.shape[-1]))
        return tm.lm_head(last_hidden),PackedKV(torch.stack(caches)),torch.cat(selected,dim=-1),final
    def prepare(self,max_batch):
        param=next(self.gpt.gpt.parameters())
        for b in batches(max_batch):
            # First-chunk frontend contracts are fixed to these padded extents.
            # Longer/tail inputs are deliberately eager, never another graph bucket.
            for kind,length in (('prefill',48),('latent',80)):
                x=param.new_zeros(b,length,self.gpt.gpt.embed_dim)
                keep=torch.ones(b,length,device=x.device,dtype=torch.long)
                table=self.prefill_graphs if kind=='prefill' else self.latent_graphs
                table[b,length]=capture(lambda xx,mm,k=kind:self.body(xx,mm,k=='prefill'),(x,keep))
        return self.stats()
    def run(self,x,keep,kind):
        b,length=x.shape[:2];table=self.prefill_graphs if kind=='prefill' else self.latent_graphs
        limit=next((n for bb,n in sorted(table) if bb==b and n>=length),None)
        if limit is None:
            self.tail_eager[kind]+=1
            return self.body(x,keep,kind=='prefill')
        graph=table[b,limit];xx,mm=graph.inputs
        xx.zero_();mm.zero_();xx[:,:length].copy_(x);mm[:,:length].copy_(keep)
        graph.graph.replay();self.hits[kind]+=1
        if kind=='latent':return graph.outputs[:,:length]
        logits,kv,selected,final=graph.outputs
        return logits,kv,selected[:,:length],final[:,:length]
    def prefill(self,x,past,keep,pos):
        if past is not None:raise ValueError('Prefix graph is not verification')
        return self.run(x,keep,'prefill')
    def latent(self,x,keep):return self.run(x,keep,'latent')
    def stats(self):
        return dict(prefill_keys=[list(k) for k in self.prefill_graphs],latent_keys=[list(k) for k in self.latent_graphs],
                    hits=dict(self.hits),tail_eager=dict(self.tail_eager),online_capture=False)
