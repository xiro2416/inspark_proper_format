"""Current static verification/context/prefix/solver graphs; no new GPU kernels."""
import torch
from torch import nn
from torch.nn import functional as F

class TargetVerification(nn.Module):
    def __init__(self,target):
        super().__init__();self.blocks=target.model.transformer.h;self.final_norm=target.model.transformer.ln_f;self.head=target.model.lm_head;self.ids=target.target_layer_ids
    def forward(self,x,mask,*caches):
        hidden=x;selected=[];appended=[];b,q,_=x.shape
        for i,block in enumerate(self.blocks):
            attn=block.attn
            query,key,value=attn.c_attn(block.ln_1(hidden)).split(attn.split_size,-1)
            reshape=lambda v:v.bfloat16().reshape(b,q,attn.num_heads,attn.head_dim).transpose(1,2)
            query,key,value=map(reshape,(query,key,value));appended.extend([key.contiguous(),value.contiguous()])
            keys=torch.cat([caches[2*i],key],2);values=torch.cat([caches[2*i+1],value],2)
            scaled=query*(attn.head_dim**-.5)
            scores=torch.matmul(scaled.float(),keys.float().transpose(-2,-1))
            scores=scores.masked_fill(~mask,float('-inf'))
            context=torch.matmul(torch.softmax(scores,-1),values.float())
            context=context.transpose(1,2).contiguous().reshape_as(hidden)
            hidden=hidden+attn.c_proj(context);hidden=hidden+block.mlp(block.ln_2(hidden))
            if i in self.ids:selected.append(hidden)
        final=self.final_norm(hidden)
        return (self.head(final),torch.cat(selected,-1),final,*appended)

class ContextKV(nn.Module):
    def __init__(self,draft):super().__init__();self.layers=nn.ModuleList(draft.layers[1:])
    def forward(self,x):
        keys=[];values=[]
        for layer in self.layers:
            keys.append(layer.k_proj(x).unflatten(-1,(layer.num_heads,layer.head_dim)))
            values.append(layer.v_proj(x).unflatten(-1,(layer.num_heads,layer.head_dim)))
        return torch.stack(keys,1).contiguous(),torch.stack(values,1).contiguous()

class Prefix(nn.Module):
    def __init__(self,engine,kind):
        super().__init__();target=engine.rt.engine.target
        self.blocks=target.model.transformer.h;self.norm=target.model.transformer.ln_f
        self.final_norm=engine.tts.gpt.final_norm;self.head=target.model.lm_head;self.ids=target.target_layer_ids;self.kind=kind
    def forward(self,x,keep,past=None):
        hidden=x;b,q,_=x.shape;positions=torch.arange(q,device=x.device)
        causal=(positions[:,None]>=positions[None,:])[None,None]
        mask=causal&keep[:,None,None,:].bool() if past is None else torch.cat([keep[:,None,None,:].bool().expand(b,1,q,48),causal.expand(b,1,q,q)],-1)
        selected=[];cache=[]
        for i,block in enumerate(self.blocks):
            attn=block.attn
            query,key,value=attn.c_attn(block.ln_1(hidden)).split(attn.split_size,-1)
            shape=lambda v:v.reshape(b,q,attn.num_heads,attn.head_dim).transpose(1,2)
            query,key,value=map(shape,(query,key,value))
            keys=key if past is None else torch.cat([past[i,0],key],2)
            vals=value if past is None else torch.cat([past[i,1],value],2)
            scores=torch.matmul(query,keys.transpose(-2,-1))*(attn.head_dim**-.5)
            probabilities=torch.softmax(scores.masked_fill(~mask,float('-inf')),-1)
            context=torch.matmul(probabilities,vals).transpose(1,2).contiguous().reshape_as(hidden)
            hidden=hidden+attn.c_proj(context);hidden=hidden+block.mlp(block.ln_2(hidden))
            if self.kind=='prefill':
                cache.append(torch.stack([key,value]))
                if i in self.ids:selected.append(hidden)
        final=self.norm(hidden)
        if self.kind!='prefill':return self.final_norm(final)
        last=(keep.long()*positions[None]).amax(-1)
        last_hidden=final.gather(1,last[:,None,None].expand(-1,1,final.shape[-1]))
        return self.head(last_hidden),torch.stack(cache),torch.cat(selected,-1),final
