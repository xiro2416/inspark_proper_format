"""Explicit original E4M3 projection, FP32 bias/residual and next quantization."""
from typing import Tuple,Union
import numpy as np
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .residual_kernel import residual_aot

@trtp.register('zipvoice_fp8_sm120_b128_residual::FusedProjectionResidual')
def desc(a:trtp.TensorDesc,w:trtp.TensorDesc,bias:trtp.TensorDesc,res:trtp.TensorDesc,alpha:float,inv:float)->Tuple[trtp.TensorDesc,trtp.TensorDesc]:
    assert a.dtype==w.dtype==trt.fp8 and bias.dtype==res.dtype==trt.float32
    original=res.like();quantized=res.like();quantized.dtype=trt.fp8
    return original,quantized

@trtp.aot_impl('zipvoice_fp8_sm120_b128_residual::FusedProjectionResidual')
def aot(a:trtp.TensorDesc,w:trtp.TensorDesc,bias:trtp.TensorDesc,res:trtp.TensorDesc,alpha:float,inv:float,outputs:Tuple[trtp.TensorDesc],tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
    n,k=w.shape[0],w.shape[1]
    compiled=triton.compile(triton.compiler.ASTSource(fn=residual_aot,signature={'A':'*fp8e4nv','W':'*fp8e4nv','Bias':'*fp32','R':'*fp32','M':'i32','Y':'*fp32','Q':'*fp8e4nv'},constexprs={'N':n,'K':k,'Alpha':float(np.float32(alpha)),'Inv':float(np.float32(inv))}),options={'num_warps':4,'num_stages':3,'enable_fp_fusion':False})
    assert 'e4m3' in compiled.asm['ptx'] and not compiled.metadata.global_scratch_size and not compiled.metadata.profile_scratch_size
    rows=res.shape_expr[0]*res.shape_expr[1]
    launch=trtp.KernelLaunchParams();launch.grid_x=((rows+63)//64)*((n+63)//64);launch.grid_y=1;launch.grid_z=1;launch.block_x=128;launch.shared_mem=compiled.metadata.shared
    extra=trtp.SymIntExprs(1);extra[0]=trtp.SymInt32(rows)
    return compiled.metadata.name,compiled.asm['ptx'],launch,extra
