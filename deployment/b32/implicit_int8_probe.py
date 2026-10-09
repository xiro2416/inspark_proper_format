"""Two-stage INT8 implicit convolution probe: avoid expanded column storage."""
import json,statistics
from pathlib import Path
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ROOT=Path(__file__).resolve().parents[2]


@triton.jit
def quantize_nc(X,SCALE,SMOOTH,Q,N:tl.constexpr,C:tl.constexpr,F:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    channel=(i//F)%C
    x=tl.load(X+i,i<N,0);smooth=tl.load(SMOOTH+channel)
    scale=tl.load(SCALE)
    value=libdevice.div_rn(libdevice.div_rn(x,smooth),scale)
    q=tl.minimum(tl.maximum(libdevice.rint(value),-128.),127.).to(tl.int8)
    tl.store(Q+i,q,i<N)


@triton.jit
def implicit_conv(Q,W,ACT,WS,BIAS,Y,B:tl.constexpr,C:tl.constexpr,F:tl.constexpr,
                  O:tl.constexpr,K:tl.constexpr,L:tl.constexpr,STRIDE:tl.constexpr,
                  PAD:tl.constexpr,DILATION:tl.constexpr,EXPAND:tl.constexpr,
                  BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);outs=tl.program_id(1)*BN+tl.arange(0,BN)
    b=rows//L;t=rows%L
    ks=tl.arange(0,BK);acc=tl.zeros((BM,BN),tl.int32)
    for start in range(tl.cdiv(C*K,BK)):
        k=start*BK+ks;channel=k//K;tap=k%K
        pos=t[:,None]*STRIDE+tap[None,:]*DILATION-PAD
        valid=(pos>=0)&(pos%EXPAND==0)&(pos//EXPAND<F)
        x=tl.load(Q+(b[:,None]*C+channel[None,:])*F+pos//EXPAND,
                  (rows[:,None]<B*L)&(k[None,:]<C*K)&valid,0)
        w=tl.load(W+outs[None,:]*(C*K)+k[:,None],(k[:,None]<C*K)&(outs[None,:]<O),0)
        acc=tl.dot(x,w,acc,out_dtype=tl.int32)
    factor=tl.load(ACT)*tl.load(WS+outs,outs<O,0)
    bias=tl.load(BIAS+outs,outs<O,0)
    value=acc.to(tl.float32)*factor[None,:]+bias[None,:]
    tl.store(Y+(b[:,None]*O+outs[None,:])*L+t[:,None],value,
             (rows[:,None]<B*L)&(outs[None,:]<O))


class ImplicitConv:
    def __init__(self,weight,act_scale,weight_scale,bias,smooth,*,stride=1,padding=0,dilation=1,expand=1,bm=32,bn=32,bk=64):
        self.weight,self.act,self.ws,self.bias,self.smooth=weight,act_scale,weight_scale,bias,smooth
        self.stride,self.padding,self.dilation,self.expand=stride,padding,dilation,expand
        self.bm,self.bn,self.bk=bm,bn,bk;self.q=self.y=None;self.compiled=None
    def __call__(self,x):
        b,c,f=x.shape;o,_,k=self.weight.shape
        l=((f-1)*self.expand+1+2*self.padding+getattr(self,"output_padding",0)-self.dilation*(k-1)-1)//self.stride+1
        if self.q is None:self.q=torch.empty_like(x,dtype=torch.int8);self.y=torch.empty(b,o,l,device=x.device)
        quantize_nc[(triton.cdiv(x.numel(),256),)](x,self.act,self.smooth,self.q,x.numel(),c,f,256,num_warps=4,enable_fp_fusion=False)
        self.compiled=implicit_conv[(triton.cdiv(b*l,self.bm),triton.cdiv(o,self.bn))](
            self.q,self.weight,self.act,self.ws,self.bias,self.y,b,c,f,o,k,l,
            self.stride,self.padding,self.dilation,self.expand,self.bm,self.bn,self.bk,
            num_warps=4,num_stages=3,enable_fp_fusion=False)
        return self.y


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    import onnx
    from onnx import numpy_helper
    from deployment.b32.probe_fir import graph_measure
    source=ROOT/'.cache/b32-columns-probe/floating'
    model=onnx.load(source/'model.onnx');initializers={v.name:numpy_helper.to_array(v) for v in model.graph.initializer}
    # Read the exact same weights/scales/bias from the already-built matched control.
    print([(k,v.shape) for k,v in initializers.items()],flush=True)
    weight_name=next(n.input[0] for n in model.graph.node if n.op_type=='QuantizeLinear' and n.input[1]=='weight_scale')
    weight=initializers[weight_name];act=initializers['input_scale'];ws=initializers['weight_scale'];bias=initializers['bias']
    weight=weight.reshape(24,24,11)
    with GPULease(1):
        select_gpu(1);trt=_import_trt113();logger=trt.Logger(trt.Logger.ERROR)
        runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine((source/'model.engine').read_bytes());context=engine.create_execution_context()
        torch.manual_seed(1032);x=torch.randn(32,24,13312,device='cuda');y=torch.empty_like(x)
        context.set_tensor_address('x',x.data_ptr());context.set_tensor_address('y',y.data_ptr())
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        def control():
            if not context.execute_async_v3(stream.cuda_stream):raise RuntimeError('Control enqueue failed')
            return y
        base,expected=graph_measure(control,stream)
        tensor=lambda a:torch.tensor(a,device='cuda')
        wq=tensor((weight/ws[:,None,None]).round().clip(-128,127)).to(torch.int8).contiguous()
        rows=[]
        for bm,bn,bk in [(32,32,64),(64,32,64),(32,32,128),(64,32,128),(32,16,64)]:
            op=ImplicitConv(wq,tensor(act),tensor(ws),tensor(bias),torch.ones(24,device='cuda'),padding=5,bm=bm,bn=bn,bk=bk)
            measured,observed=graph_measure(lambda:op(x),stream);diff=observed-expected
            if not torch.isfinite(observed).all():raise RuntimeError('Nonfinite implicit convolution')
            ptx=op.compiled.asm['ptx'];mma='mma.sync' in ptx and '.s8.s8.s32' in ptx
            if not mma:raise RuntimeError('No actual signed INT8 MMA instruction')
            rows.append(dict(tile=[bm,bn,bk],p50_ms=measured['p50_ms'],max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/expected.norm()),
                             actual_int8_mma=True,registers=op.compiled.n_regs,shared_memory=op.compiled.metadata.shared,
                             gain_pct=100*(base['p50_ms']-measured['p50_ms'])/base['p50_ms']))
            print(json.dumps(rows[-1]),flush=True)
        (ROOT/'deployment/b32/history/implicit-int8-probe.json').write_text(json.dumps(dict(scope='standalone real-shape convolution, same exact synthetic-control weights/scales/input, complete quantize+implicit-conv Graph; not E2E',baseline=base,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
