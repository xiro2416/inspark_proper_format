"""Reuse overlapping filter windows across two outputs without changing FMA order."""
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from acc_infer_clear.kernels.alias_free import _up_snake,_down

@triton.jit(do_not_specialize=['T','S0','S1','S2'])
def up_pair(X,F,A,IB,U,T,C:tl.constexpr,S0,S1,S2,BLOCK:tl.constexpr):
    bc=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    base=X+(bc//C)*S0+(bc%C)*S1
    prev=tl.load(base+tl.minimum(tl.maximum(j+3,0),T-1)*S2,j<T,0).to(tl.float32)
    even=tl.full((BLOCK,),0,tl.float32);odd=tl.full((BLOCK,),0,tl.float32)
    for tap in tl.static_range(6):
        nxt=tl.load(base+tl.minimum(tl.maximum(j+2-tap,0),T-1)*S2,j<T,0).to(tl.float32)
        even=tl.fma(nxt,tl.load(F+1+2*tap),even)
        odd=tl.fma(prev,tl.load(F+2*tap),odd);prev=nxt
    alpha=tl.load(A+bc%C);ib=tl.load(IB+bc%C)
    ve=even*2.;vo=odd*2.;se=libdevice.sin(ve*alpha);so=libdevice.sin(vo*alpha)
    tl.store(U+bc*2*T+2*j,ve+ib*(se*se),j<T)
    tl.store(U+bc*2*T+2*j+1,vo+ib*(so*so),j<T)

@triton.jit(do_not_specialize=['T'])
def down_pair(U,F,Y,T,BLOCK:tl.constexpr):
    bc=tl.program_id(0);j=2*(tl.program_id(1)*BLOCK+tl.arange(0,BLOCK))
    even=tl.full((BLOCK,),0,tl.float32);odd=tl.full((BLOCK,),0,tl.float32)
    for k in tl.static_range(14):
        pos=tl.minimum(tl.maximum(2*j-5+k,0),2*T-1)
        v=tl.load(U+bc*2*T+pos,j<T,0)
        if k<12:even=tl.fma(v,tl.load(F+k),even)
        if k>=2:odd=tl.fma(v,tl.load(F+k-2),odd)
    tl.store(Y+bc*T+j,even,j<T);tl.store(Y+bc*T+j+1,odd,j+1<T)

def run(old,x,plan):
    b,c,t=x.shape;u=torch.empty((b,c,2*t),device=x.device,dtype=x.dtype);y=torch.empty_like(x,memory_format=torch.contiguous_format)
    block=plan['block'];warps=plan['warps']
    if plan['up']:
        up_pair[(b*c,triton.cdiv(t,block))](x,old.up,old.alpha,old.inverse_beta,u,t,c,*x.stride(),block,num_warps=warps,enable_fp_fusion=False)
    else:_up_snake[(b*c,triton.cdiv(2*t,256))](x,old.up,old.alpha,old.inverse_beta,u,t,c,*x.stride(),256,enable_fp_fusion=False)
    if plan['down']:down_pair[(b*c,triton.cdiv(t,2*block))](u,old.down,y,t,block,num_warps=warps,enable_fp_fusion=False)
    else:_down[(b*c,triton.cdiv(t,256))](u,old.down,y,t,256,enable_fp_fusion=False)
    return y
