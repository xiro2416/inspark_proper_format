"""Fused exact 12-tap replicate-padded upsample/SnakeBeta/downsample."""
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _small(X,UP,DOWN,ALPHA,BETA,OUT,C:tl.constexpr,F:tl.constexpr,
           HIGH:tl.constexpr,LOW:tl.constexpr):
    row=tl.program_id(0);channel=row%C
    u=tl.arange(0,HIGH)
    phase=(u+1)%2
    value=tl.zeros((HIGH,),tl.float32)
    for tap in tl.static_range(6):
        k=2*tap+phase
        index=tl.minimum(tl.maximum((u+5-k)//2,0),F-1)
        source=tl.load(X+row*F+index,mask=u<2*F,other=0.0)
        coefficient=tl.load(UP+k)
        value=value+source*coefficient
    value=2.0*value
    alpha=libdevice.exp(tl.load(ALPHA+channel));beta=libdevice.exp(tl.load(BETA+channel))
    sine=libdevice.sin(alpha*value)
    high=value+sine*sine/(beta+1e-9)
    o=tl.arange(0,LOW);result=tl.zeros((LOW,),tl.float32)
    for tap in tl.static_range(12):
        index=tl.minimum(tl.maximum(2*o+tap-5,0),2*F-1)
        result=result+tl.gather(high,index,axis=0)*tl.load(DOWN+tap)
    tl.store(OUT+row*F+o,result,mask=o<F)


class SmallFIR:
    def __init__(self,up,down,alpha,beta):
        self.parameters=(up,down,alpha,beta);self.output=None
    def __call__(self,x):
        if x.dtype!=torch.float32 or not x.is_contiguous() or x.shape[-1]>4096:
            raise ValueError('Expected contiguous FP32 short vocoder activation')
        b,c,f=x.shape
        if self.output is None:self.output=torch.empty_like(x)
        _small[(b*c,)](x,*self.parameters,self.output,c,f,HIGH=triton.next_power_of_2(2*f),LOW=triton.next_power_of_2(f),num_warps=8,enable_fp_fusion=False)
        return self.output
