from typing import Tuple,Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
import triton.language as tl
from .protected_ffn_kernel import linear_f32_activation

@triton.jit
def protected_aot(X,W,Bias,M,Out,N:tl.constexpr,BM:tl.constexpr):
    linear_f32_activation(X,W,Bias,Out,M,512,N,BM,64,32,4.,0.07999999821186066,0.03500000014901161,'tf32')

@trtp.register('zipvoice_fp8_sm120_b128_protected::F32SwooshL')
def desc(x:trtp.TensorDesc,w:trtp.TensorDesc,bias:trtp.TensorDesc,width:int,bm:int)->trtp.TensorDesc:
    assert x.dtype==w.dtype==bias.dtype==trt.float32
    out=x.like();dims=trtp.ShapeExprs(3);dims[0]=x.shape_expr[0];dims[1]=x.shape_expr[1];dims[2]=width;out.shape_expr=dims
    return out

@trtp.aot_impl('zipvoice_fp8_sm120_b128_protected::F32SwooshL')
def aot(x:trtp.TensorDesc,w:trtp.TensorDesc,bias:trtp.TensorDesc,width:int,bm:int,outputs:Tuple[trtp.TensorDesc],tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
    compiled=triton.compile(triton.compiler.ASTSource(fn=protected_aot,signature={'X':'*fp32','W':'*fp32','Bias':'*fp32','M':'i32','Out':'*fp32'},constexprs={'N':width,'BM':bm}),options={'num_warps':4,'num_stages':1,'enable_fp_fusion':False})
    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx'] and not compiled.metadata.global_scratch_size
    rows=x.shape_expr[0]*x.shape_expr[1];launch=trtp.KernelLaunchParams();launch.grid_x=((rows+bm-1)//bm)*((width+63)//64);launch.grid_y=1;launch.grid_z=1;launch.block_x=128;launch.shared_mem=compiled.metadata.shared
    extra=trtp.SymIntExprs(1);extra[0]=trtp.SymInt32(rows)
    return compiled.metadata.name,compiled.asm['ptx'],launch,extra
