"""Reuse one activation quantization and one output-wide GEMM for Draft Q/K/V."""
import torch,triton
from inspark_infer.runtime.graph_policy import batches
from inspark_infer.ops.triton.target_norm_quant.prequantized import Prepared, base_linear, binding, quantize, resolve
from inspark_infer.ops.triton.stage2_gemm import _gemm_col
from inspark_infer.ops.triton.target_full_m.tiled import run as full_m_linear

class SharedQKV:
    def __init__(self,layer,max_batch):
        self.shape={};self.combine_b8=False;self.full_m_b8_plan=None;self.linears=(layer.q_proj,layer.k_proj,layer.v_proj)
        self.projections=[]
        extents=[b*7 for b in batches(max_batch)]
        for linear in (layer.q_proj,layer.k_proj,layer.v_proj):
            raw=base_linear(linear)
            if getattr(raw,'precision',None)!='fp8':raise ValueError('SharedQKV requires three FP8 projections')
            owner=linear
            while not hasattr(owner,'weight_col') and hasattr(owner,'old'):owner=owner.old
            self.projections.append(Prepared(raw.weight,raw.scales,raw.bias,{m:resolve(linear,m) for m in extents},
                col=getattr(owner,'weight_col',None),ones=getattr(owner,'ones',None),bindings={m:binding(linear,m) for m in extents}))
        self.combined_col=torch.cat([p.col for p in self.projections],0).contiguous()
        self.combined_scales=torch.cat([p.scales for p in self.projections])
        self.combined_bias=torch.cat([p.bias for p in self.projections])
    def __call__(self,x):
        m=x.numel()//x.shape[-1]
        if m not in self.projections[0].choices:return tuple(linear(x) for linear in self.linears)
        if m==56 and self.full_m_b8_plan is not None:
            # This is one combined [56,1280]x[1280,3840] projection and one
            # quantize.  Do not execute the old shared quantize first.
            y=full_m_linear(x,self.combined_col,self.combined_scales,
                            self.combined_bias,self.full_m_b8_plan)
            n=y.shape[-1]
            return y.split(n//3,-1)
        q,s=quantize(x)
        if q.shape[0]==56 and self.combine_b8:
            m,k=q.shape;n=self.combined_scales.numel();y=torch.empty(m,n,device=q.device,dtype=torch.float32)
            _gemm_col[(triton.cdiv(m,16),triton.cdiv(n,32))](q,self.combined_col,s,self.combined_scales,self.combined_bias,y,m,n,k,16,32,128,True,num_warps=4,num_stages=2)
            return y.view(*x.shape[:-1],n).split(n//3,-1)
        return tuple(p(q,s,x.shape) for p in self.projections)

def prepare(engine):
    backbone=engine.rt.backbone;fused={}
    for index,layer in enumerate(backbone.model.layers):
        raw=[base_linear(getattr(layer,n)) for n in ('q_proj','k_proj','v_proj')]
        if all(getattr(x,'precision',None)=='fp8' for x in raw):fused[index]=SharedQKV(layer,engine.config['max_batch'])
    backbone.shared_qkv=fused
    return dict(layers=sorted(fused),quantize_before=3,quantize_after=1,b8_gemm_before=3,b8_gemm_after=3,combined_qkv_candidate='inactive_after_integrated_A/B',precision_unchanged=True,online_tuning=False)
