"""FIR/Snake/FIR output tiles with exact receptive-field halos.

The halo contains every high-rate sample used by the 12-tap down filter.
Clamping is performed on the global high-rate index before evaluating the
upsample and Snake, exactly as the reference's replicate padding requires.
"""
import triton
import triton.language as tl
import torch
from triton.language.extra.cuda import libdevice


@triton.jit
def _tiled(X,UP,DOWN,ALPHA,BETA,OUT,C:tl.constexpr,F:tl.constexpr,
           BLOCK:tl.constexpr,HIGH:tl.constexpr):
    row=tl.program_id(0)
    # A power-of-two high buffer holds 2*ACTIVE+10 required samples.
    # Leave eight low-rate lanes inactive rather than double the high buffer.
    ACTIVE:tl.constexpr=BLOCK-8
    base=tl.program_id(1)*ACTIVE
    channel=row%C
    u=tl.minimum(tl.maximum(2*base-5+tl.arange(0,HIGH),0),2*F-1)
    phase=(u+1)%2
    value=tl.zeros((HIGH,),tl.float32)
    for tap in tl.static_range(6):
        k=2*tap+phase
        index=tl.minimum(tl.maximum((u+5-k)//2,0),F-1)
        source=tl.load(X+row*F+index)
        value=value+source*tl.load(UP+k)
    value=2.*value
    alpha=libdevice.exp(tl.load(ALPHA+channel))
    beta=libdevice.exp(tl.load(BETA+channel))
    sine=libdevice.sin(alpha*value)
    high=value+sine*sine/(beta+1e-9)
    o=tl.arange(0,BLOCK)
    result=tl.zeros((BLOCK,),tl.float32)
    for tap in tl.static_range(12):
        index=tl.minimum(2*o+tap,HIGH-1)
        result=result+tl.gather(high,index,axis=0)*tl.load(DOWN+tap)
    tl.store(OUT+row*F+base+o,result,mask=(o<ACTIVE)&(base+o<F))


class TiledFIR:
    def __init__(self,up,down,alpha,beta,block=256,warps=4):
        if block not in (64,128,256,512) or warps not in (4,8):
            raise ValueError('Unsupported tiled FIR schedule')
        if up.numel()!=12 or down.numel()!=12:
            raise ValueError('Tiled FIR requires complete 12-tap filters')
        self.parameters=(up,down,alpha,beta)
        self.block,self.warps=block,warps
        self.output=None
    def __call__(self,x):
        if x.ndim!=3 or x.dtype!=torch.float32 or not x.is_cuda or not x.is_contiguous():
            raise ValueError('Tiled FIR expects contiguous CUDA FP32 B,C,F')
        b,c,f=x.shape
        if f<=0 or self.parameters[2].numel()!=c or self.parameters[3].numel()!=c:
            raise ValueError('Tiled FIR channel/length mismatch')
        if self.output is None:self.output=torch.empty_like(x)
        if self.output.shape!=x.shape:raise ValueError('Tiled FIR instance has fixed shape')
        _tiled[(b*c,triton.cdiv(f,self.block-8))](x,*self.parameters,self.output,c,f,
            BLOCK=self.block,HIGH=2*self.block,num_warps=self.warps,enable_fp_fusion=False)
        return self.output
