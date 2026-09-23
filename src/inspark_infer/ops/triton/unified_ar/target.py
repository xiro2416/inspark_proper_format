"""Unified Target QKV, mixed-precision slot cache and FP8 QK attention."""
import types
import math
import torch
import triton
import triton.language as tl
from inspark_infer.runtime.indextts2.slot_target import SlotKV
from inspark_infer.ops.triton.unified_ar.attention import quantize_vectors, append_fp8, append_qk_fp8, target_attention, quantize_blocks32, target_attention_block32
from inspark_infer.ops.triton.unified_ar.qkv import run_target_slots, unified_plan
from inspark_infer.ops.triton.kv_attention import append as append_fp32, attention as attention_fp32
from inspark_infer.runtime.graphs import capture
from inspark_infer.runtime.graph_policy import batches, FIRST_KV_LIMITS

@triton.jit(do_not_specialize=['ROW','SLOT','LENGTH','P0','P1','P2','P3','P4','P5'])
def _import_deep_k8_v32(PACKED,K8,SK,V32,ROW,SLOT,LENGTH,
                        P0,P1,P2,P3,P4,P5,
                        LAYERS:tl.constexpr,NSLOTS:tl.constexpr,H:tl.constexpr,CAP:tl.constexpr,D:tl.constexpr,MAX_T:tl.constexpr):
    index=tl.program_id(0);layer=index//(H*MAX_T);head=(index//MAX_T)%H;token=index%MAX_T;dim=tl.arange(0,D);valid=token<LENGTH
    source=(layer*P0+ROW*P2+head*P3+token*P4+dim*P5);k=tl.load(PACKED+source,valid,0.).to(tl.float32);v=tl.load(PACKED+source+P1,valid,0.)
    blocks=k.reshape((2,32));scale=tl.maximum(tl.max(tl.abs(blocks),1)/448.,1e-12)
    koff=((((layer*NSLOTS+SLOT)*H+head)*CAP+token)*D+dim)
    soff=(((layer*NSLOTS+SLOT)*H+head)*CAP+token)*2+tl.arange(0,2)
    voff=((((layer*NSLOTS+SLOT)*H+head)*CAP+token)*D+dim)
    tl.store(K8+koff,(blocks/scale[:,None]).reshape((D,)),valid);tl.store(SK+soff,scale,valid);tl.store(V32+voff,v,valid)

def _import_deep(packed,row,slot,length,k8,sk,v32):
    layers=k8.shape[0];h=k8.shape[-3];d=k8.shape[-1]
    _import_deep_k8_v32[(layers*h*48,)](packed,k8,sk,v32,int(row),int(slot),int(length),*packed.stride(),layers,k8.shape[2],h,k8.shape[-2],d,48,num_warps=2)


def _import_cache(self,packed,row,length):
    if not self.free:raise RuntimeError('Target KV slots exhausted')
    slot=self.free.pop();self.generations[slot]+=1
    cut=self.fp8_cutoff
    self.fp32_storage[:,:,slot,:,:length].copy_(packed[:cut,:,row,:,:length])
    if self.fp8_attention_mode==1:
        _import_deep(packed[cut:],row,slot,length,self.storage,self.fp8_scales,self.fp32_v_storage)
    else:
        source=packed[cut:,:,row,:,:length];q,s=quantize_vectors(source)
        self.storage[:,:,slot,:,:length].copy_(q);self.fp8_scales[:,:,slot,:,:length].copy_(s)
    return SlotKV(self,slot,length,self.generations[slot])


def _math(self,x,slots,lengths,limit):
    from inspark_infer.ops.cuda.target_norm_quant.fused import fused
    tm=self.target.model;model=tm.transformer;selected=[];hidden=x;b=x.shape[0]
    for index,block in enumerate(model.h):
        a=block.attn;pair=self._norm_quant_pairs.get((index,'qkv'))
        direct=False
        if index>=self.fp8_cutoff and pair is not None and self.fp8_attention_mode==1:
                m=b*8;qa,sa,_=fused(hidden,pair.gamma,pair.beta,pair.eps,False,pair.config['division'],pair.config['sync_strategy'])
                col,_=pair.dense.bindings[m]
                q8,sq=run_target_slots(qa,sa,col,pair.dense.scales,pair.dense.bias,unified_plan(m,32),
                    self.storage[index-self.fp8_cutoff,0],self.fp8_scales[index-self.fp8_cutoff,0],
                    self.fp32_v_storage[index-self.fp8_cutoff] if self.fp8_attention_mode==1 else self.storage[index-self.fp8_cutoff,1],
                    self.fp8_scales[index-self.fp8_cutoff,0 if self.fp8_attention_mode==1 else 1],slots,lengths,v_fp32=self.fp8_attention_mode==1)
                direct=True
        if not direct:
            qkv=pair(hidden) if pair is not None else a.c_attn(block.ln_1(hidden))
            q,k,v=qkv.split(a.split_size,dim=2)
            q=q.view(b,8,a.num_heads,a.head_dim).transpose(1,2);k=k.view(b,8,a.num_heads,a.head_dim).transpose(1,2);v=v.view(b,8,a.num_heads,a.head_dim).transpose(1,2)
            if index<self.fp8_cutoff:
                append_fp32(k,v,self.fp32_storage[index,0],self.fp32_storage[index,1],slots,lengths)
            else:
                q8,sq=quantize_vectors(q);k8,sk=quantize_vectors(k);j=index-self.fp8_cutoff
                if self.fp8_attention_mode==1:append_qk_fp8(k8,sk,v,self.storage[j,0],self.fp8_scales[j,0],self.fp32_v_storage[j],slots,lengths)
                else:
                    v8,sv=quantize_vectors(v);append_fp8(k8,v8,sk,sv,self.storage[j,0],self.storage[j,1],self.fp8_scales[j,0],self.fp8_scales[j,1],slots,lengths)
        if index<self.fp8_cutoff:
            out=attention_fp32(q,self.fp32_storage[index,0],self.fp32_storage[index,1],self.keep,slots,lengths,limit,self.consumer_layout)
        else:
            j=index-self.fp8_cutoff
            if self.fp8_attention_mode==1:
                out=target_attention_block32(q8,sq,self.storage[j,0],self.fp8_scales[j,0],self.fp32_v_storage[j],self.keep,slots,lengths,limit,self.consumer_layout)
            else:out=target_attention(q8,sq,self.storage[j,0],self.fp8_scales[j,0],self.storage[j,1],self.fp8_scales[j,1],self.keep,slots,lengths,limit,self.fp8_attention_mode,self.consumer_layout)
        out=out.reshape(b,8,a.embed_dim) if self.consumer_layout else out.transpose(1,2).contiguous().view(b,8,a.embed_dim)
        hidden=self._residual_pairs[index,'out'](out,hidden) if (index,'out') in self._residual_pairs else hidden+a.c_proj(out)
        if (index,'up') in self._norm_quant_pairs:
            fc=self._norm_quant_pairs[index,'up'](hidden);activated=block.mlp.act(fc)
            hidden=self._residual_pairs[index,'down'](activated,hidden) if (index,'down') in self._residual_pairs else hidden+block.mlp.dropout(block.mlp.c_proj(activated))
        else:
            activated=block.mlp.act(block.mlp.c_fc(block.ln_2(hidden)))
            hidden=self._residual_pairs[index,'down'](activated,hidden) if (index,'down') in self._residual_pairs else hidden+block.mlp.dropout(block.mlp.c_proj(activated))
        if index in self.target.target_layer_ids:selected.append(hidden)
    final=model.ln_f(hidden)
    return tm.lm_head(final),torch.cat(selected,dim=-1),final


def _prepare_graphs(self):
    if len(self.free)!=self.max_slots:raise RuntimeError('Capture before admitting requests')
    if self.graph_sealed:raise RuntimeError('Already captured')
    self.storage.zero_();self.fp8_scales.fill_(1.);self.fp32_storage.zero_();self.keep.fill_(1)
    if self.fp8_attention_mode==1:self.fp32_v_storage.zero_()
    param=next(self.target.model.transformer.parameters())
    for b in batches(self.max_batch):
        slots=torch.arange(b,device=self.storage.device,dtype=torch.int32)
        x=param.new_zeros(b,8,self.target.model.transformer.embed_dim,dtype=torch.float32)
        for limit in FIRST_KV_LIMITS:
            lengths=torch.full((b,),limit-8,device=x.device,dtype=torch.int32)
            self.graphs[b,limit]=capture(lambda xx,ss,ll:self.math(xx,ss,ll,limit),(x,slots,lengths))
    self.graph_sealed=True


def attach(engine,mode=2):
    previous=engine.prepare_slot_target
    def prepare_slot_target(graphs=False):
        previous(graphs=False);target=engine.rt.target;shape=target.storage.shape;device=target.storage.device
        target.fp8_cutoff=math.ceil(shape[0]/4);target.fp32_storage=torch.empty((target.fp8_cutoff,*shape[1:]),device=device,dtype=torch.float32)
        depth=shape[0]-target.fp8_cutoff;planes=1 if int(mode)==1 else 2
        target.storage=torch.empty((depth,planes,*shape[2:]),device=device,dtype=torch.float8_e4m3fn)
        target.fp8_scales=torch.empty((*target.storage.shape[:-1],2),device=device,dtype=torch.float32) if int(mode)==1 else torch.empty(target.storage.shape[:-1],device=device,dtype=torch.float32)
        if int(mode)==1:target.fp32_v_storage=torch.empty((depth,*shape[2:]),device=device,dtype=torch.float32)
        target.fp8_scales.fill_(1.)
        target.import_cache=types.MethodType(_import_cache,target);target.pool_import=target.import_cache
        target.math=types.MethodType(_math,target);target.unified_ar=True;target.fp8_attention_mode=int(mode)
        target.prepare_graphs=types.MethodType(_prepare_graphs,target)
        if int(mode)==1:
            # Compile the dynamic-stride importer before service. Runtime rows,
            # slots and valid lengths do not create new specializations.
            dummy=torch.zeros((depth,2,1,shape[3],48,shape[-1]),device=device,dtype=torch.float32)
            _import_deep(dummy,0,0,0,target.storage,target.fp8_scales,target.fp32_v_storage)
        with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
            if graphs:target.prepare_graphs()
        return dict(target.stats(),fp8_kv_layers=list(range(target.fp8_cutoff,len(target.target.model.transformer.h))),protected_fp32_layers=list(range(target.fp8_cutoff)),fp8_qk=True,fp8_pv=target.fp8_attention_mode==2,v_dequant=target.fp8_attention_mode==3,softmax='fp32',online_tuning=False)
    engine.prepare_slot_target=prepare_slot_target
