"""48/80 candidate routing and matched BM64 controls, offline capture only."""
import contextlib,weakref
import torch
from acc_infer_clear.runtime.prefix_graphs import PrefixGraphs
from acc_infer_clear.runtime.graphs import capture
from acc_infer_clear.target_seven.projections import Projection,base_linear,signature
from acc_infer_clear.target_norm_quant.prequantized import resolve
from acc_infer_clear.kernels.fp8 import linear
from acc_infer_clear.kernels.stage2_gemm import col_linear
from acc_infer_clear.kernels.planner import Tile
from acc_infer_clear.target_seven.pointwise import gelu

class Fixed(torch.nn.Module):
 def __init__(self,old,choice):
  super().__init__();self.old=old;self.choice=choice;self.raw=base_linear(old)
  owner=old
  while not hasattr(owner,'weight_col') and hasattr(owner,'old'):owner=owner.old
  self.col=getattr(owner,'weight_col',None)
  if self.col is None:self.col=self.raw.weight[:,:self.raw.scales.numel()].t().contiguous()
  self.ones=getattr(owner,'ones',None)
  if self.ones is None:self.ones=torch.ones(1,device=self.col.device)
 def forward(self,x):
  r=self.raw;p=self.choice;t=Tile(**p['tile'])
  if p['kind']=='row':return linear(x,r.weight,r.scales,r.bias,t)
  return col_linear(x,self.col,r.scales,r.bias,t,'triton',self.ones)

class Fused(torch.nn.Module):
 def forward(self,x):return gelu(x,512,4)

class BF16Compat(torch.nn.Module):
 def __init__(self,old,length):super().__init__();self.old=old;self.length=length
 def forward(self,x):
  n=x.shape[1]
  if n>=self.length:return self.old(x)
  padded=x.new_zeros(x.shape[0],self.length,x.shape[-1]);padded[:,:n].copy_(x)
  return self.old(padded)[:,:n].contiguous()

def choose_limit(keys,batch,length):
 return next((n for n in sorted({n for b,n in keys if b==batch}) if n>=length),None)

class Bank(PrefixGraphs):
 def __init__(self,e):super().__init__(e);self.audit=[];self.record=False;self.head_only=False;self.legacy_keys=None;self.engine_ref=weakref.ref(e)
 def run(self,x,keep,kind):
  b,length=x.shape[:2];table=self.prefill_graphs if kind=='prefill' else self.latent_graphs
  keys=table
  if self.head_only:
   engine=self.engine_ref();active=[s for s in engine.sessions.values() if not s.get('complete',False)] if engine else []
   if not active or any(s['chunks'] for s in active):
    self.tail_eager[kind]+=1;return self.body(x,keep,kind=='prefill')
  limit=choose_limit(keys,b,length)
  if self.record:
   mask=keep.bool();positions=torch.arange(length,device=x.device)[None].expand_as(keep)
   self.audit.append(dict(kind=kind,batch=b,length=length,bucket=limit,valid=mask.sum(-1).cpu().tolist(),last_valid=(torch.where(mask,positions,-1).amax(-1)+1).cpu().tolist(),x=x.detach().cpu(),keep=keep.detach().cpu()))
  if limit is None:
   self.tail_eager[kind]+=1;return self.body(x,keep,kind=='prefill')
  g=table[b,limit];xx,mm=g.inputs;xx.zero_();mm.zero_();xx[:,:length].copy_(x);mm[:,:length].copy_(keep);g.graph.replay();self.hits[kind]+=1
  if kind=='latent':return g.outputs[:,:length]
  logits,kv,selected,final=g.outputs;return logits,kv,selected[:,:length],final[:,:length]

def choices(e,b,current):
 out={}
 for kind,length in [('prefill',64),('latent',128)]:
  block=e.tts.gpt.gpt.h[6];m=b*length
  for role,mod in [('qkv',block.attn.c_attn),('out',block.attn.c_proj),('up',block.mlp.c_fc),('down',block.mlp.c_proj)]:
   r=base_linear(mod);key=signature(role,m,r.out_features,r.in_features)
   p=current['projections'].get(key) if b==8 else None
   if p is None:p=resolve(mod,m)
   if p['kind']=='explicit':
    a=p['plan'];t=dict(bm=64,bn=a['bn'],bk=a['bk'],warps=4,stages=a['stages'],split_k=1)
   else:t=dict(p['tile']);t['bm']=64;t.pop('schedule',None)
   out[kind,role]=dict(kind='row' if t.get('split_k',1)>1 else 'compiler',tile=t)
 return out

@contextlib.contextmanager
def modules(e,b,kind,length,current,fixed=None,fuse=False,compat=False):
 replaced=[];held=[]
 try:
  for block in e.tts.gpt.gpt.h:
   for owner,name,role in ((block.attn,'c_attn','qkv'),(block.attn,'c_proj','out'),(block.mlp,'c_fc','up'),(block.mlp,'c_proj','down')):
    old=getattr(owner,name);r=base_linear(old)
    if r.precision!='fp8':
     legacy_length=64 if kind=='prefill' else 128
     if compat and length<legacy_length:
      new=BF16Compat(old,legacy_length);replaced.append((owner,name,old));held.append(new);setattr(owner,name,new)
     continue
    if fixed is not None:new=Fixed(old,fixed[kind,role])
    else:
     key=signature(role,b*length,r.out_features,r.in_features);p=current['projections'].get(key) if b==8 else None
     if p is None:continue
     new=Projection(old,{b*length:p})
    replaced.append((owner,name,old));held.append(new);setattr(owner,name,new)
   old=block.mlp.act
   if fuse:
    new=Fused();replaced.append((block.mlp,'act',old));held.append(new);block.mlp.act=new
  yield held
 finally:
  for owner,name,old in reversed(replaced):setattr(owner,name,old)

def make(e,b,current,variant='current',fixed=None,compat=False):
 bank=Bank(e);bank.keepalive=[]
 with torch.cuda.stream(e.model.stream),torch.inference_mode():
  for kind,length in [('prefill',48 if variant=='tight' else 64),('latent',80 if variant=='tight' else 128)]:
   x=next(e.tts.gpt.gpt.parameters()).new_zeros(b,length,e.tts.gpt.gpt.embed_dim);mask=torch.ones(b,length,device=x.device,dtype=torch.long)
   with modules(e,b,kind,length,current,fixed,variant!='current',compat) as held:
    g=capture(lambda x,m:bank.body(x,m,kind=='prefill'),(x,mask))
   bank.keepalive.extend(held)
   (bank.prefill_graphs if kind=='prefill' else bank.latent_graphs)[b,length]=g
 return bank

def bind(e,bank):
 e.prefix_graphs=bank;e.rt.target.prefill_body=bank.prefill;e.rt.latent.body=bank.latent

def diagnose(e,b,current,fixed,audits,compat=False):
 rows=[];samples={}
 with torch.cuda.stream(e.model.stream),torch.inference_mode():
  for record in audits:
   kind=record['kind'];raw=record['x'].cuda();keep=record['keep'].cuda();length=raw.shape[1];reference={}
   for name,limit in [('old',64 if kind=='prefill' else 128),('tight',48 if kind=='prefill' else 80)]:
    if length>limit:continue
    x=raw.new_zeros(b,limit,raw.shape[-1]);x[:,:length].copy_(raw);mask=keep.new_zeros(b,limit);mask[:,:length].copy_(keep);hooks=[]
    bank=Bank(e)
    with modules(e,b,kind,limit,current,fixed,True,compat):
     for path,mod in e.tts.gpt.gpt.named_modules():
      if '.old' in path or '.base' in path or '.raw' in path or not path.endswith(('ln_1','ln_2','c_attn','c_proj','c_fc','act')):continue
      def hook(mod,args,y,path=path):
       if not isinstance(y,torch.Tensor) or y.ndim!=3:return
       v=y[:,:length]
       if name=='old':reference[path]=v.clone()
       else:
        ref=reference[path];diff=(v-ref).abs();valid=keep.bool()[:,:,None].expand_as(v)
        rows.append(dict(kind=kind,path=path,equal=bool(torch.equal(v[valid],ref[valid])),max_abs=float(diff[valid].max()),relative_l2=float((v[valid]-ref[valid]).norm()/ref[valid].norm().clamp_min(1e-12))))
       if path.startswith(('h.0.','h.6.')) and path.endswith(('c_attn','c_proj','c_fc')):
        samples[kind+'/'+name+'/'+path]=dict(x=args[0].detach().cpu(),y=y.detach().cpu())
      hooks.append(mod.register_forward_hook(hook))
     try:bank.body(x,mask,kind=='prefill');torch.cuda.synchronize()
     finally:
      for h in hooks:h.remove()
 return rows,samples
