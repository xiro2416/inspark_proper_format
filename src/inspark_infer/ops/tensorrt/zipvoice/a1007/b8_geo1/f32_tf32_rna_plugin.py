from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .f32_tf32_rna_aot import f32_tf32_rna_aot

@trtp.register('zipvoice_int8_b8_geo1::F32TF32RNAFF1')
def desc_ff1(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc) -> trtp.TensorDesc:
    assert x.dtype == w.dtype == bias.dtype == trt.float32
    dims = trtp.ShapeExprs(3)
    dims[0] = x.shape_expr[0]
    dims[1] = x.shape_expr[1]
    dims[2] = 1152
    out = x.like()
    out.shape_expr = dims
    return out

@trtp.aot_impl('zipvoice_int8_b8_geo1::F32TF32RNAFF1')
def aot_ff1(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=f32_tf32_rna_aot, signature={'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'M': 'i32', 'Out': '*fp32'}, constexprs={'N': 1152}), options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})
    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx'] and compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size)
    m = x.shape_expr[0] * x.shape_expr[1]
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (m + 63) // 64 * 18
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(m)
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)

@trtp.register('zipvoice_int8_b8_geo1::F32TF32RNAFF2')
def desc_ff2(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc) -> trtp.TensorDesc:
    assert x.dtype == w.dtype == bias.dtype == trt.float32
    dims = trtp.ShapeExprs(3)
    dims[0] = x.shape_expr[0]
    dims[1] = x.shape_expr[1]
    dims[2] = 1536
    out = x.like()
    out.shape_expr = dims
    return out

@trtp.aot_impl('zipvoice_int8_b8_geo1::F32TF32RNAFF2')
def aot_ff2(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=f32_tf32_rna_aot, signature={'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'M': 'i32', 'Out': '*fp32'}, constexprs={'N': 1536}), options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})
    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx'] and compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size)
    m = x.shape_expr[0] * x.shape_expr[1]
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (m + 63) // 64 * 24
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(m)
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)

@trtp.register('zipvoice_int8_b8_geo1::F32TF32RNAFF3')
def desc_ff3(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc) -> trtp.TensorDesc:
    assert x.dtype == w.dtype == bias.dtype == trt.float32
    dims = trtp.ShapeExprs(3)
    dims[0] = x.shape_expr[0]
    dims[1] = x.shape_expr[1]
    dims[2] = 1920
    out = x.like()
    out.shape_expr = dims
    return out

@trtp.aot_impl('zipvoice_int8_b8_geo1::F32TF32RNAFF3')
def aot_ff3(x: trtp.TensorDesc, w: trtp.TensorDesc, bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=f32_tf32_rna_aot, signature={'X': '*fp32', 'W': '*fp32', 'Bias': '*fp32', 'M': 'i32', 'Out': '*fp32'}, constexprs={'N': 1920}), options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})
    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx'] and compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size)
    m = x.shape_expr[0] * x.shape_expr[1]
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (m + 63) // 64 * 30
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(m)
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)
