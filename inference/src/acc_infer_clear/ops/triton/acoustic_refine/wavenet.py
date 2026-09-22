"""Fuse only exact residual adds/masks; retain learned layers and gate math."""
import torch,triton
import triton.language as tl
from acc_infer_clear.models.indextts2.upstream.s2mel.modules import commons

@triton.jit
def update(X,R,O,M,Y,Z,N:tl.constexpr,C:tl.constexpr,T:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr,XS2:tl.constexpr,RS0:tl.constexpr,RS1:tl.constexpr,RS2:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,OS2:tl.constexpr,MS0:tl.constexpr,MS2:tl.constexpr,LAST:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N;b=i//(C*T);c=i//T%C;t=i%T
    o=tl.load(O+b*OS0+c*OS1+t*OS2,valid,0)
    mask=tl.load(M+b*MS0+t*MS2,valid,0).to(tl.float32)
    if LAST:
        r=tl.load(R+b*RS0+c*RS1+t*RS2,valid,0)
        tl.store(Z+i,(o+r)*mask,valid)
    else:
        x=tl.load(X+b*XS0+c*XS1+t*XS2,valid,0)
        r=tl.load(R+b*RS0+c*RS1+t*RS2,valid,0)
        skip=tl.load(R+b*RS0+(c+C)*RS1+t*RS2,valid,0)
        tl.store(Y+i,(x+r)*mask,valid);tl.store(Z+i,o+skip,valid)

class RefinedWaveNet(torch.nn.Module):
    def __init__(self,old,shapes):super().__init__();self.old=old;self.shapes=set(map(tuple,shapes))
    def forward(self,x,x_mask,g=None,**kwargs):
        if tuple(x.shape) not in self.shapes or x.dtype!=torch.float32 or x_mask.shape!=(x.shape[0],1,x.shape[2]):return self.old(x,x_mask,g,**kwargs)
        o=self.old;output=torch.zeros_like(x);b,c,t=x.shape
        if g is not None:g=o.cond_layer(g)
        for i in range(o.n_layers):
            xin=o.in_layers[i](x)
            gl=g[:,i*2*c:(i+1)*2*c,:] if g is not None else torch.zeros_like(xin)
            acts=o.drop(commons.fused_add_tanh_sigmoid_multiply(xin,gl,(c,)))
            r=o.res_skip_layers[i](acts);last=i==o.n_layers-1
            nx=torch.empty((b,c,t),device=x.device,dtype=x.dtype) if not last else x
            z=torch.empty((b,c,t),device=x.device,dtype=x.dtype)
            update[(triton.cdiv(x.numel(),256),)](x,r,output,x_mask,nx,z,x.numel(),c,t,*x.stride(),*r.stride(),*output.stride(),x_mask.stride(0),x_mask.stride(2),last,256,enable_fp_fusion=False)
            x=nx;output=z
        return output
