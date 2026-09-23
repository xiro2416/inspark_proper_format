"""Pair-only Target verification math. No shared-state handoff or upstream API edits."""
import types
import torch
from inspark_infer.ops.cuda.target_norm_quant.fused import prepare, fused
from inspark_infer.ops.triton.target_norm_quant.prequantized import Prepared, resolve, base_linear, binding
from inspark_infer.ops.triton.target_norm_quant.residual import col_linear_residual
from inspark_infer.ops.triton.target_norm_quant.residual_down import run as down_residual

class Pair:
    def __init__(self,norm,linear,config):
        self.norm=norm;self.linear=linear;self.config=config
        base=norm.base if hasattr(norm,'base') else norm
        self.gamma,self.beta,self.eps=base.weight,base.bias,base.eps
        raw=base_linear(linear);owner=linear
        while not hasattr(owner,'weight_col') and hasattr(owner,'old'):owner=owner.old
        extents=sorted({b*8 for b in config['batches']})
        self.dense=Prepared(raw.weight,raw.scales,raw.bias,{m:resolve(linear,m) for m in extents},
            col=getattr(owner,'weight_col',None),ones=getattr(owner,'ones',None),bindings={m:binding(linear,m) for m in extents})
    def __call__(self,x):
        if x.dtype!=torch.float32 or x.ndim!=3 or x.shape[1:]!=(8,1280) or x.shape[0] not in self.config['batches'] or not x.is_contiguous():
            return self.linear(self.norm(x))
        q,s,_=fused(x,self.gamma,self.beta,self.eps,False,self.config['division'],self.config['sync_strategy'])
        return self.dense(q,s,x.shape)

class ResidualPair:
    def __init__(self,linear,role,config=None):
        self.linear,self.role=linear,role;self.raw=base_linear(linear);extents=sorted({b*8 for b in (config or {}).get('batches',[8])});self.choices={m:resolve(linear,m) for m in extents};self.bindings={m:binding(linear,m)[0] for m in extents}
    def __call__(self,x,residual):
        m=x.numel()//x.shape[-1]
        choice=self.choices.get(m);col=self.bindings.get(m)
        if choice is None or x.dtype!=torch.float32 or residual.dtype!=torch.float32:
            return residual+self.linear(x)
        if self.role=='out' and choice['kind']=='compiler':
            return col_linear_residual(x,residual,col,self.raw.scales,self.raw.bias,choice['tile'])
        if self.role=='down' and choice['kind']=='explicit':
            return down_residual(x,residual,col,self.raw.scales,self.raw.bias,choice['plan'])
        return residual+self.linear(x)

def paired_math(self,x,slots,lengths,limit):
    # The arithmetic/order is the original SlotTarget.math, except two explicit
    # pairs. Prefill/latent keep using the original modules and Tensor interfaces.
    tm=self.target.model;model=tm.transformer;selected=[];hidden=x
    for index,block in enumerate(model.h):
        a=block.attn
        with self._profile_scope('target/qkv_projection'):
            qkv=self._norm_quant_pairs[index,'qkv'](hidden) if (index,'qkv') in self._norm_quant_pairs else a.c_attn(block.ln_1(hidden))
        with self._profile_scope('target/qkv_output_layout'):
            q,k,v=qkv.split(a.split_size,dim=2)
            q=q.view(*q.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
            k=k.view(*k.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
            v=v.view(*v.shape[:-1],a.num_heads,a.head_dim).transpose(1,2)
        with self._profile_scope('target/kv_append'):
            self._norm_quant_append(k,v,self.storage[index,0],self.storage[index,1],slots,lengths)
        out=self._norm_quant_attention(q,self.storage[index,0],self.storage[index,1],self.keep,slots,lengths,limit,self.consumer_layout)
        with self._profile_scope('target/attention_output_layout'):
            out=(out.reshape(x.shape[0],8,a.embed_dim) if self.consumer_layout else
                 out.transpose(1,2).contiguous().view(x.shape[0],8,a.embed_dim))
        hidden=self._residual_pairs[index,'out'](out,hidden) if (index,'out') in getattr(self,'_residual_pairs',{}) else hidden+a.c_proj(out)
        if (index,'up') in self._norm_quant_pairs:
            fc=self._norm_quant_pairs[index,'up'](hidden)
            activated=block.mlp.act(fc)
            hidden=self._residual_pairs[index,'down'](activated,hidden) if (index,'down') in getattr(self,'_residual_pairs',{}) else hidden+block.mlp.dropout(block.mlp.c_proj(activated))
        else:
            normalized=block.ln_2(hidden);activated=block.mlp.act(block.mlp.c_fc(normalized))
            hidden=self._residual_pairs[index,'down'](activated,hidden) if (index,'down') in getattr(self,'_residual_pairs',{}) else hidden+block.mlp.dropout(block.mlp.c_proj(activated))
        if index in self.target.target_layer_ids:selected.append(hidden)
    final=model.ln_f(hidden)
    return tm.lm_head(final),torch.cat(selected,dim=-1),final

def attach(engine,config):
    original=engine.prepare_target_seven
    def prepare_target_seven(path):
        result=original(path)
        with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():prepare()
        previous=engine.prepare_slot_target
        def prepare_slot_target(graphs=False):
            previous(graphs=False);target=engine.rt.target
            env=target.math.__func__.__globals__
            target._norm_quant_append=env['append'];target._norm_quant_attention=env['attention']
            pairs={}
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                for i,block in enumerate(target.target.model.transformer.h):
                    for role,norm,linear in [('qkv',block.ln_1,block.attn.c_attn),('up',block.ln_2,block.mlp.c_fc)]:
                        if role in config['roles'] and base_linear(linear).precision=='fp8':pairs[i,role]=Pair(norm,linear,config)
                target._norm_quant_pairs=pairs;target.math=types.MethodType(paired_math,target)
                residual={}
                if config.get('residual'):
                    for i,block in enumerate(target.target.model.transformer.h):
                        for role,linear in [('out',block.attn.c_proj),('down',block.mlp.c_proj)]:
                            if base_linear(linear).precision=='fp8':residual[i,role]=ResidualPair(linear,role,config)
                target._residual_pairs=residual
                if graphs:target.prepare_graphs()
            return target.stats()
        engine.prepare_slot_target=prepare_slot_target
        result['norm_quant_experiment']=config
        return result
    engine.prepare_target_seven=prepare_target_seven
