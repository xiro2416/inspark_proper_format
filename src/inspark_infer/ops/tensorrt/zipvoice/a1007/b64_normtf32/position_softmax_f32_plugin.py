"""Float32-only AOT plugin; runtime T is supplied from the actual binding shape."""
from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .position_softmax_f32_kernel import position_softmax_f32

@trtp.register('zipvoice_int8_b64_normtf32::PositionSoftmaxFloat32')
def desc(query: trtp.TensorDesc, emb: trtp.TensorDesc, scores: trtp.TensorDesc, mask: trtp.TensorDesc, capacity: int, batch: int) -> trtp.TensorDesc:
    if any((x.dtype != trt.float32 for x in (query, emb, scores))) or mask.dtype != trt.bool:
        raise ValueError('Original Float32 boundaries and Bool mask required')
    if batch != 64 or capacity < 1 or capacity > 1750:
        raise ValueError('Verified complete B64 profile capacity required')
    return scores.like()

@trtp.aot_impl('zipvoice_int8_b64_normtf32::PositionSoftmaxFloat32')
def aot(query: trtp.TensorDesc, emb: trtp.TensorDesc, scores: trtp.TensorDesc, mask: trtp.TensorDesc, capacity: int, batch: int, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=position_softmax_f32, signature={'Query': '*fp32', 'Emb': '*fp32', 'Scores': '*fp32', 'Mask': '*i1', 'T': 'i32', 'Out': '*fp32'}, constexprs={'B': batch, 'BLOCK': triton.next_power_of_2(capacity)}), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    if compiled.metadata.shared > 49152 or compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
        raise RuntimeError('Unexpected scratch/shared-memory requirement')
    launch = trtp.KernelLaunchParams()
    launch.grid_x = scores.shape_expr[2]
    launch.grid_y = batch
    launch.grid_z = 4
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(scores.shape_expr[2])
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)
