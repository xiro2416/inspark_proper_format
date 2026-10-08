"""Original INT8 activation/weight boundary, Float32 scales/bias/output."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .dw_int8_aot_kernel import dw_int8_aot

@trtp.register('zipvoice_int8_b16::DepthwiseOriginalInt8')
def desc(x: trtp.TensorDesc, w: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, input_scale: float, kernel: int) -> trtp.TensorDesc:
    assert x.dtype == w.dtype == trt.int8 and ws.dtype == bias.dtype == trt.float32
    assert kernel in (7, 15, 31) and input_scale > 0
    out = x.like()
    out.dtype = trt.float32
    return out

@trtp.aot_impl('zipvoice_int8_b16::DepthwiseOriginalInt8')
def aot(x: trtp.TensorDesc, w: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, input_scale: float, kernel: int, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=dw_int8_aot, signature={'X': '*i8', 'W': '*i8', 'WS': '*fp32', 'Bias': '*fp32', 'T': 'i32', 'Out': '*fp32'}, constexprs={'XS': input_scale, 'K': kernel}), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    assert compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size)
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (x.shape_expr[3] + 255) // 256
    launch.grid_y = 16 * 512
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(x.shape_expr[3])
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)
