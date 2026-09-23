"""FP32 fixed-filter resampling + Snake, two kernels; no learned conv conversion."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=['T','S0','S1','S2'])
def _up_snake(X,F,A,IB,U,T,C:tl.constexpr,S0,S1,S2,
              BLOCK:tl.constexpr):
    bc=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    parity=(j+15)%2;acc=tl.full((BLOCK,),0,tl.float32)
    for tap in tl.static_range(6):
        k=parity+2*tap;i=tl.minimum(tl.maximum((j+15-k)//2-5,0),T-1)
        x=tl.load(X+(bc//C)*S0+(bc%C)*S1+i*S2,j<2*T,0).to(tl.float32)
        f=tl.load(F+k);acc=tl.fma(x,f,acc)
    v=acc*2.;alpha=tl.load(A+bc%C);inverse_beta=tl.load(IB+bc%C)
    sine=libdevice.sin(v*alpha);result=v+inverse_beta*(sine*sine)
    tl.store(U+bc*(2*T)+j,result,j<2*T)

@triton.jit(do_not_specialize=['T'])
def _down(U,F,Y,T,BLOCK:tl.constexpr):
    bc=tl.program_id(0);j=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK);acc=tl.full((BLOCK,),0,tl.float32)
    for k in tl.static_range(12):
        pos=tl.minimum(tl.maximum(j*2-5+k,0),2*T-1)
        v=tl.load(U+bc*2*T+pos,j<T,0);acc=tl.fma(v,tl.load(F+k),acc)
    tl.store(Y+bc*T+j,acc,j<T)

class FusedAliasFree(torch.nn.Module):
    def __init__(self,original):
        super().__init__()
        assert original.up_ratio==original.down_ratio==2
        assert original.upsample.kernel_size==original.downsample.lowpass.kernel_size==12
        a=original.act;alpha=a.alpha.detach().float();beta=getattr(a,'beta',a.alpha).detach().float()
        if a.alpha_logscale:alpha=alpha.exp();beta=beta.exp()
        self.register_buffer('alpha',alpha.contiguous());self.register_buffer('inverse_beta',(1./(beta+1e-9)).contiguous())
        self.register_buffer('up',original.upsample.filter.detach().float().contiguous())
        self.register_buffer('down',original.downsample.lowpass.filter.detach().float().contiguous())
    def forward(self,x):
        if x.dtype!=torch.float32:raise ValueError('Only tested FP32 activation interface')
        b,c,t=x.shape;u=torch.empty((b,c,2*t),device=x.device,dtype=x.dtype);y=torch.empty((b,c,t),device=x.device,dtype=x.dtype)
        _up_snake[(b*c,triton.cdiv(2*t,256))](x,self.up,self.alpha,self.inverse_beta,u,t,c,*x.stride(),256,enable_fp_fusion=False)
        _down[(b*c,triton.cdiv(t,256))](u,self.down,y,t,256,enable_fp_fusion=False)
        return y

def install(vocoder):
    changed=[]
    def walk(module,prefix):
        for name,child in list(module.named_children()):
            path=prefix+'.'+name
            if type(child).__name__=='Activation1d':module.add_module(name,FusedAliasFree(child));changed.append(path)
            else:walk(child,path)
    walk(vocoder,'vocoder');return changed
