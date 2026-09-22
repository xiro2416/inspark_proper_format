"""Unified Draft QKV, mixed-precision DraftPool and FP8 QK attention."""
import types
import torch
from acc_infer_clear.ops.triton.unified_ar.attention import quantize_blocks32, draft_attention_block32
from acc_infer_clear.ops.triton.unified_ar.qkv import run_qkv_mixed
from acc_infer_clear.ops.triton.target_norm_quant.prequantized import quantize
from acc_infer_clear.ops.triton.target_norm_quant.prequantized import binding
from acc_infer_clear.ops.triton.target_seven.projections import base_linear
from acc_infer_clear.ops.triton.target_norm_quant.residual import col_linear_residual
from acc_infer_clear.ops.triton.target_norm_quant.residual_down import run as down_residual
from acc_infer_clear.ops.triton.target_full_m.tiled import run_prequantized
from acc_infer_clear.ops.triton.draft_attention import attention as attention_fp32
from acc_infer_clear.ops.triton.context_scatter import scatter_layer, scatter_layer_k8_v32
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.runtime.graph_policy import batches, FIRST_KV_LIMITS


def _attach_cache(self,cache):
    if hasattr(cache,'pool_slot'):self.check(cache);return
    if not self.free:raise RuntimeError('Draft slots exhausted')
    slot=self.free.pop();self.generations[slot]+=1;cache.pool_slot=slot;cache.pool_generation=self.generations[slot];cache.pool_released=False
    cache.storage_keys=[self.fp32_storage[0,0,slot:slot+1],self.k8_storage[0,slot:slot+1],self.k8_storage[1,slot:slot+1]]
    cache.storage_values=[self.fp32_storage[0,1,slot:slot+1],self.v32_storage[0,slot:slot+1],self.v32_storage[1,slot:slot+1]]
    if cache.length:
        cache.storage_keys[0][:,:,:cache.length].copy_(cache.keys[0]);cache.storage_values[0][:,:,:cache.length].copy_(cache.values[0])
        for index in (1,2):
            q,s=quantize_blocks32(cache.keys[index]);cache.storage_keys[index][:,:,:cache.length].copy_(q);self.k_scales[index-1,slot:slot+1,:,:cache.length].copy_(s)
            cache.storage_values[index][:,:,:cache.length].copy_(cache.values[index])


def _math(self,anchors,positions,slots,lengths,limit):
    m=self.model;hidden=m._noise_embeddings(anchors,positions);context_pos=lengths[:,None]+self.step;b=anchors.shape[0]
    for index,layer in enumerate(m.layers):
        norm=layer.input_norm(hidden)
        if index==0:
            q=layer._heads(layer.q_proj(norm));k=layer._heads(layer.k_proj(norm));v=layer._heads(layer.v_proj(norm))
            if hasattr(layer,'rope_inv_freq'):q=layer.rotate(q,context_pos);k=layer.rotate(k,context_pos)
            attended=attention_fp32(q,k,v,self.pool.fp32_storage[0,0],self.pool.fp32_storage[0,1],slots,lengths,limit,self.consumer_layout)
        else:
            if hasattr(layer,'rope_inv_freq'):raise RuntimeError('Unified Draft FP8 QKV requires offline RoPE weight folding')
            fused=self.shared_qkv[index];qa,sa=quantize(norm);qk,ss,vflat=run_qkv_mixed(qa,sa,fused.combined_col,fused.combined_scales,fused.combined_bias,32,qlen=7)
            h,d=layer.num_heads,layer.head_dim
            q8=qk[0].view(b,7,h,d).transpose(1,2);k8=qk[1].view(b,7,h,d).transpose(1,2)
            sq=ss[0];sk=ss[1];v=vflat.view(b,7,h,d).transpose(1,2)
            attended=draft_attention_block32(q8,sq,k8,sk,v,self.pool.k8_storage[index-1],self.pool.k_scales[index-1],self.pool.v32_storage[index-1],slots,lengths,limit,self.consumer_layout)
        consumer=attended.reshape_as(hidden) if self.consumer_layout else attended.transpose(1,2).contiguous().view_as(hidden)
        if index==0:hidden=hidden+layer.o_proj(consumer);hidden=hidden+layer.mlp(layer.post_norm(hidden))
        else:
            ops=self.unified_linears[index];hidden=ops['out'](consumer,hidden);normalized=layer.post_norm(hidden);qa,sa=quantize(normalized);activated=layer.mlp[1](ops['up'](qa,sa,normalized.shape));hidden=ops['down'](activated,hidden)
        hidden=m.apply_query_temporal(hidden,index)
    hidden=m.project_output(hidden);return hidden,m.base_logits(hidden)


def _prepare_graphs(self,max_batch):
    self.pool.fp32_storage.zero_();self.pool.k8_storage.zero_();self.pool.k_scales.fill_(1.);self.pool.v32_storage.zero_();param=next(self.model.parameters())
    for b in batches(max_batch):
        anchors=torch.zeros(b,device=param.device,dtype=torch.long);positions=torch.arange(7,device=param.device)[None].expand(b,-1).clone();slots=torch.arange(b,device=param.device,dtype=torch.int32)
        for limit in FIRST_KV_LIMITS:
            lengths=torch.full((b,),limit-7,device=param.device,dtype=torch.int32)
            self.graphs[b,limit]=capture(lambda aa,pp,ss,ll:self.math(aa,pp,ss,ll,limit),(anchors,positions,slots,lengths))
    return dict(keys=[list(k) for k in self.graphs],history_packing=False,online_capture=False,unified_qkv=True,fp8_k=True,fp32_v=True)


def _context_direct(self,prepared,positions,source,lengths,slots,destinations):
    m=self.model;default=None if m.context_fusion_mode=='depth_aligned' else m.project_context(prepared)
    for index,layer in enumerate(m.layers):
        context=m.project_context(prepared,index) if default is None else default
        key,value=(layer.context_kv(context,positions) if m.architecture=='official_qwen3' or m.random_rope_draft else layer.context_kv(context))
        if index==0:scatter_layer(key,value,self.pool.fp32_storage,0,source,lengths,slots,destinations)
        else:scatter_layer_k8_v32(key,value,self.pool.k8_storage[index-1],self.pool.k_scales[index-1],self.pool.v32_storage[index-1],source,lengths,slots,destinations)
    return prepared[:,:1,:1]


def attach(engine):
    backbone=engine.rt.backbone;pool=backbone.pool
    if sorted(getattr(backbone,'shared_qkv',{}))!=[1,2]:raise RuntimeError('Unified Draft QKV requires prepared deep-layer combined weights')
    old=pool.storage;_,_,slots,h,cap,d=old.shape;device=old.device
    pool.fp32_storage=torch.empty((1,2,slots,h,cap,d),device=device,dtype=torch.float32)
    pool.k8_storage=torch.empty((2,slots,h,cap,d),device=device,dtype=torch.float8_e4m3fn)
    pool.k_scales=torch.empty((2,slots,h,cap,2),device=device,dtype=torch.float32)
    pool.v32_storage=torch.empty((2,slots,h,cap,d),device=device,dtype=torch.float32)
    pool.storage=pool.fp32_storage;pool.attach=types.MethodType(_attach_cache,pool)
    class Up:
        def __init__(self,module):
            raw=base_linear(module);self.scales,self.bias=raw.scales,raw.bias;self.col=next(x for x in (binding(module,m)[0] for m in (7,14,21,28,35,42,49,56)) if x is not None)
        def __call__(self,q,s,shape):
            m=q.shape[0];bm=max(16,1<<(m-1).bit_length());warps=4 if bm<=32 else 8;plan=dict(mode='tiled',bm=bm,bn=64,bk=128,stages=6,warps=warps,wm=2,wn=warps//2,schedule='full',swizzle=True)
            return run_prequantized(q,s,shape,self.col,self.scales,self.bias,plan)
    class Residual:
        def __init__(self,module,role):
            raw=base_linear(module);self.role=role;self.scales,self.bias=raw.scales,raw.bias;self.col=next(x for x in (binding(module,m)[0] for m in (7,14,21,28,35,42,49,56)) if x is not None)
        def __call__(self,x,residual):
            if self.role=='out':return col_linear_residual(x,residual,self.col,self.scales,self.bias,dict(bm=16,bn=64,bk=128,warps=4,stages=4))
            return down_residual(x,residual,self.col,self.scales,self.bias,dict(bm=16,bn=64,bk=128,stages=5,swizzle=True,inner=False))
    backbone.unified_linears={}
    for index in (1,2):
        layer=backbone.model.layers[index];backbone.unified_linears[index]=dict(out=Residual(layer.o_proj,'out'),up=Up(layer.mlp[0]),down=Residual(layer.mlp[2],'down'))
    backbone.math=types.MethodType(_math,backbone);backbone.prepare_graphs=types.MethodType(_prepare_graphs,backbone);backbone.unified_ar=True
    engine.rt.context._project_scatter_direct=types.MethodType(_context_direct,engine.rt.context)
    return dict(layers=[1,2],backend='UnifiedQKV',qk='fp8_block32',softmax='fp32',pv='fp32',stage=2,unified_linear_roles=['out','up','down'],online_tuning=False)
