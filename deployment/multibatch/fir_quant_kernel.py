"""Exact source FP32 FIR/Snake/FIR followed by the unchanged signed INT8 rule.

Only the FP32 store/reload and a separate quantization launch are eliminated.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def fir_quant(X,UP,DOWN,ALPHA,BETA,SMOOTH,SCALE,OUT,C:tl.constexpr,F:tl.constexpr,
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
    normalized=libdevice.div_rn(result,tl.load(SMOOTH+channel))
    scaled=libdevice.div_rn(normalized,tl.load(SCALE))
    quantized=tl.minimum(tl.maximum(libdevice.rint(scaled),-128.),127.).to(tl.int8)
    tl.store(OUT+row*F+base+o,quantized,mask=(o<ACTIVE)&(base+o<F))
