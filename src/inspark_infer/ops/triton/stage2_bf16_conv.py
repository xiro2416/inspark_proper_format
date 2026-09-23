"""Native BF16 implicit learned Conv1d, preserving BF16 input/weight/output rounding."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['T','TO','M'])
def _bf16_conv(X,W,BIAS,Y,T,TO,M,CI:tl.constexpr,CO:tl.constexpr,KW:tl.constexpr,STRIDE:tl.constexpr,PAD:tl.constexpr,DIL:tl.constexpr,
               HAS_BIAS:tl.constexpr,BIAS_AFTER_ROUND:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM);ni=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
    batch=mi//TO;out_t=mi%TO;acc=tl.zeros((BM,BN),tl.float32)
    for step in range(tl.cdiv(CI*KW,BK)):
        red=step*BK+kk;channel=red%CI;tap=red//CI;time=out_t[:,None]*STRIDE-PAD+tap[None,:]*DIL
        x=tl.load(X+(batch[:,None]*T+time)*CI+channel[None,:],(mi[:,None]<M)&(red[None,:]<CI*KW)&(time>=0)&(time<T),0.)
        w=tl.load(W+red[:,None]*CO+ni[None,:],(red[:,None]<CI*KW)&(ni[None,:]<CO),0.)
        acc=tl.dot(x,w,acc)
    if BIAS_AFTER_ROUND:acc=acc.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:acc+=tl.load(BIAS+ni,ni<CO,0.).to(tl.float32)[None,:]
    y=acc.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+(batch[:,None]*CO+ni[None,:])*TO+out_t[:,None],y,(mi[:,None]<M)&(ni[None,:]<CO))

def bf16_conv(x,weight,bias,ci,co,kw,stride,padding,dilation,tile,bias_after_round=True):
    b,_,t=x.shape;to=(t+2*padding-dilation*(kw-1)-1)//stride+1
    xx=x.transpose(1,2).to(dtype=torch.bfloat16,memory_format=torch.contiguous_format)
    y=torch.empty((b,co,to),device=x.device,dtype=x.dtype)
    _bf16_conv[(triton.cdiv(b*to,tile.bm),triton.cdiv(co,tile.bn))](xx,weight,bias if bias is not None else weight,y,t,to,b*to,ci,co,kw,stride,padding,dilation,
        bias is not None,bias_after_round,tile.bm,tile.bn,tile.bk,num_warps=tile.warps,num_stages=tile.stages)
    return y
