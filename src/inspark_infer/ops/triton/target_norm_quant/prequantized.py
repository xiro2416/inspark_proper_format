"""Reuse the deployed GEMM unchanged, accepting explicitly prequantized activations."""
import torch
import triton
from inspark_infer.ops.triton.fp8 import _quantize, _gemm, _reduce
from inspark_infer.ops.triton.stage2_gemm import _gemm_col, _epilogue
from inspark_infer.ops.triton.ar_pipeline.gemm import pipeline_gemm
from inspark_infer.ops.triton.target_seven.projections import Projection, base_linear
from inspark_infer.ops.triton.ar_pipeline.deploy import PipelineLinear
from inspark_infer.ops.triton.ar_refine.deploy import RefinedLinear
from inspark_infer.ops.eager.matrix import MatrixLinear

def resolve(module,m):
    from inspark_infer.ops.triton.target_full_m.deploy import FullMLinear
    if isinstance(module,FullMLinear):
        p=module.choices.get(m)
        return dict(kind='full_m',plan=dict(p)) if p is not None else resolve(module.old,m)
    if isinstance(module,Projection):
        p=module.choices.get(m)
        return dict(p) if p is not None else resolve(module.old,m)
    if isinstance(module,PipelineLinear):
        p=module.plans.get(m)
        return dict(p) if p is not None else resolve(module.old,m)
    if isinstance(module,RefinedLinear):
        p=module.plans.get(m)
        if p is None:return resolve(module.old,m)
        return dict(kind={'row':'row','col':'compiler','scaled_mm':'cublas'}[p['backend']],tile=p['tile'])
    if isinstance(module,MatrixLinear) and module.precision=='fp8':
        from dataclasses import asdict
        extent=next((n for n in module.tiles if m<=n),module._last_extent)
        return dict(kind='row',tile=asdict(module.tiles[extent]))
    raise TypeError('Unsupported deployed projection: '+type(module).__name__)

def quantize(x):
    m=x.numel()//x.shape[-1];k=x.shape[-1]
    q=torch.empty(m,k,device=x.device,dtype=torch.float8_e4m3fn);s=torch.empty(m,device=x.device,dtype=torch.float32)
    _quantize[(m,)](x,q,s,k,triton.next_power_of_2(k));return q,s

def binding(module,m):
    """Keep the exact deployed physical weight/ones buffers for each branch."""
    from inspark_infer.ops.triton.target_full_m.deploy import FullMLinear
    if isinstance(module,FullMLinear):
        owner=module if m in module.choices else None
    elif isinstance(module,Projection):
        owner=module if m in module.choices else None
    elif isinstance(module,PipelineLinear):
        owner=module.old if m in module.plans else None
    elif isinstance(module,RefinedLinear):
        owner=module if m in module.plans else None
    else:owner=module
    if owner is None:return binding(module.old,m)
    return getattr(owner,'weight_col',None),getattr(owner,'ones',None)

class Prepared:
    def __init__(self,packed,scales,bias,choices,col=None,ones=None,bindings=None):
        self.packed=packed;self.scales=scales;self.bias=bias;self.choices={int(k):v for k,v in choices.items()}
        self.k,self.n=packed.shape[0],scales.numel()
        self.col=packed[:,:self.n].t().contiguous() if col is None else col
        self.ones=torch.ones(1,device=packed.device) if ones is None else ones
        self.bindings=bindings or {}
    def full(self,x):
        """Unmodified quantize+GEMM wrappers, for independent entry-point checks."""
        from inspark_infer.ops.triton.ar_pipeline.gemm import run
        from inspark_infer.ops.triton.fp8 import linear
        from inspark_infer.ops.triton.stage2_gemm import col_linear
        from inspark_infer.ops.planning.planner import Tile
        p=self.choices[x.numel()//self.k]
        if p['kind']=='full_m':
            from inspark_infer.ops.triton.target_full_m.tiled import run as full_m
            return full_m(x,self.col,self.scales,self.bias,p['plan'])
        if p['kind']=='explicit':return run(x,self.col,self.scales,self.bias,p['plan'])
        if p['kind']=='row':return linear(x,self.packed,self.scales,self.bias,Tile(**p['tile']))
        return col_linear(x,self.col,self.scales,self.bias,Tile(**p['tile']),'scaled_mm' if p['kind']=='cublas' else 'triton',self.ones)
    def __call__(self,q,s,shape):
        m=q.shape[0];n,k=self.n,self.k;p=self.choices[m]
        col,ones=self.bindings.get(m,(self.col,self.ones))
        y=torch.empty(m,n,device=q.device,dtype=torch.float32);bias=self.bias if self.bias is not None else self.scales
        if p['kind']=='full_m':
            from inspark_infer.ops.triton.target_full_m.tiled import run_prequantized
            return run_prequantized(q,s,shape,col,self.scales,self.bias,p['plan'])
        if p['kind']=='explicit':
            a=p['plan'];bm,bn,bk=a.get('bm',16),a.get('bn',32),a.get('bk',128)
            pipeline_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](q,col,s,self.scales,bias,y,m,n,k,self.bias is not None,bm,bn,bk,a['stages'],a.get('swizzle',True),a.get('inner',False),a.get('double',False),num_warps=4)
        else:
            t=p['tile'];bm,bn,bk=t['bm'],t['bn'],t['bk']
            splits=t.get('split_k',1)
            if p['kind']=='row':
                target=y if splits==1 else torch.empty((splits,m,n),device=q.device,dtype=torch.float32)
                _gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn),splits)](q,self.packed,s,self.scales,bias,target,m,n,k,self.packed.shape[1],bm,bn,bk,self.bias is not None,splits,num_warps=t['warps'],num_stages=t['stages'])
                if splits!=1:_reduce[(triton.cdiv(m*n,256),)](target,s,self.scales,bias,y,m,n,splits,self.bias is not None,256)
            elif p['kind']=='compiler':
                if splits!=1:raise ValueError('Column kernel has no split-K reduction')
                _gemm_col[(triton.cdiv(m,bm),triton.cdiv(n,bn))](q,col,s,self.scales,bias,y,m,n,k,bm,bn,bk,self.bias is not None,num_warps=t['warps'],num_stages=t['stages'])
            elif p['kind']=='cublas':
                raw=torch._scaled_mm(q,col[:n].t(),scale_a=ones,scale_b=ones,out_dtype=torch.float32,use_fast_accum=False)
                _epilogue[(triton.cdiv(m*n,256),)](raw,s,self.scales,bias,y,m,n,self.bias is not None,256)
            else:raise ValueError(p['kind'])
        return y.view(*shape[:-1],n)
