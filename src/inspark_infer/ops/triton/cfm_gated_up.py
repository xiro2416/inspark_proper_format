"""FP8 paired GEMM with register-resident SwiGLU and static output Q/DQ."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paired(A,W1,W3,S1,S3,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            OUT_SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DEQUANT:tl.constexpr=True):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32);au=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        ki=base*BK+k
        a=tl.load(A+m[:,None]*K+ki[None,:],mask=(m[:,None]<M)&(ki[None,:]<K),other=0.0)
        wg=tl.load(W1+n[None,:]*K+ki[:,None],mask=(n[None,:]<N)&(ki[:,None]<K),other=0.0)
        wu=tl.load(W3+n[None,:]*K+ki[:,None],mask=(n[None,:]<N)&(ki[:,None]<K),other=0.0)
        ag=tl.dot(a,wg,ag,max_num_imprecise_acc=0)
        au=tl.dot(a,wu,au,max_num_imprecise_acc=0)
    gate=ag*tl.load(S1+n,mask=n<N,other=0.0)[None,:]
    up=au*tl.load(S3+n,mask=n<N,other=0.0)[None,:]
    value=(gate*tl.sigmoid(gate))*up
    q=tl.minimum(tl.maximum(value*(1.0/OUT_SCALE),-448.0),448.0).to(tl.float8e4nv)
    if DEQUANT:value=q.to(tl.float32)*OUT_SCALE
    else:value=q
    tl.store(OUT+m[:,None]*N+n[None,:],value,mask=(m[:,None]<M)&(n[None,:]<N))


class GatedUp:
    def __init__(self,weights,config):
        self.config=config
        self.input_scale=float(weights['input_scale'])
        self.output_scale=float(weights['output_scale'])
        self.weights=[];self.scales=[]
        for name in ('w1','w3'):
            w=torch.as_tensor(weights[name].copy(),device='cuda',dtype=torch.float32)
            s=torch.as_tensor(weights[name+'_scale'].copy(),device='cuda',dtype=torch.float32)
            self.weights.append((w/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous())
            self.scales.append((s*self.input_scale).contiguous())
        self.output=None
    def __call__(self,x):
        if not x.is_contiguous() or x.dtype!=torch.float32:raise ValueError('Expected contiguous FP32 CFM normalized input')
        m=x.numel()//x.shape[-1];n,k=self.weights[0].shape
        if x.shape[-1]!=k:raise ValueError('Input channels differ')
        a=(x/self.input_scale).clamp(-448,448).to(torch.float8_e4m3fn)
        if self.output is None:self.output=torch.empty((*x.shape[:-1],n),device=x.device,dtype=torch.float32)
        bm,bn,bk,warps,stages=self.config
        _paired[(triton.cdiv(m,bm),triton.cdiv(n,bn))](a,*self.weights,*self.scales,self.output,m,n,k,
            OUT_SCALE=self.output_scale,BM=bm,BN=bn,BK=bk,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        return self.output


@triton.jit
def _paired_fp32(X,W1,W3,S1,S3,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                 IN_SCALE:tl.constexpr,OUT_SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DEQUANT:tl.constexpr=True):
    W1=tl.cast(W1,tl.pointer_type(tl.float8e4nv));W3=tl.cast(W3,tl.pointer_type(tl.float8e4nv))
    m=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    ag=tl.zeros((BM,BN),tl.float32);au=tl.zeros((BM,BN),tl.float32)
    for base in range(tl.cdiv(K,BK)):
        ki=base*BK+k
        x=tl.load(X+m[:,None]*K+ki[None,:],mask=(m[:,None]<M)&(ki[None,:]<K),other=0.0)
        a=tl.minimum(tl.maximum(x*(1.0/IN_SCALE),-448.0),448.0).to(tl.float8e4nv)
        wg=tl.load(W1+n[None,:]*K+ki[:,None],mask=(n[None,:]<N)&(ki[:,None]<K),other=0.0)
        wu=tl.load(W3+n[None,:]*K+ki[:,None],mask=(n[None,:]<N)&(ki[:,None]<K),other=0.0)
        ag=tl.dot(a,wg,ag,max_num_imprecise_acc=0);au=tl.dot(a,wu,au,max_num_imprecise_acc=0)
    gate=ag*tl.load(S1+n,mask=n<N,other=0.0)[None,:];up=au*tl.load(S3+n,mask=n<N,other=0.0)[None,:]
    value=(gate*tl.sigmoid(gate))*up
    q=tl.minimum(tl.maximum(value*(1.0/OUT_SCALE),-448.0),448.0).to(tl.float8e4nv)
    tl.store(OUT+m[:,None]*N+n[None,:],q.to(tl.float32)*OUT_SCALE,mask=(m[:,None]<M)&(n[None,:]<N))


@triton.jit
def _quant_fp8_raw(X,OUT,TOTAL:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(X+i,mask=i<TOTAL,other=0.0)
    q=tl.minimum(tl.maximum(value*(1.0/SCALE),-448.0),448.0).to(tl.float8e4nv)
    tl.store(tl.cast(OUT,tl.pointer_type(tl.int8))+i,q.to(tl.int8,bitcast=True),mask=i<TOTAL)


@triton.jit
def _paired_raw(A,W1,W3,S1,S3,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                OUT_SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,DEQUANT:tl.constexpr=True):
    A=tl.cast(A,tl.pointer_type(tl.float8e4nv))
    W1=tl.cast(W1,tl.pointer_type(tl.float8e4nv));W3=tl.cast(W3,tl.pointer_type(tl.float8e4nv))
    _paired(A,W1,W3,S1,S3,OUT,M,N,K,OUT_SCALE,BM,BN,BK,DEQUANT)
