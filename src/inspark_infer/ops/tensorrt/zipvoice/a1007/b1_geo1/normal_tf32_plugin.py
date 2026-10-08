from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .normal_tf32_aot import normal_tile_aot, normal_stats_write_aot, normal_stats_read_aot

def compile_launch(fn, signature, q):
    compiled = triton.compile(triton.compiler.ASTSource(fn=fn, signature=signature, constexprs={}), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    assert compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size) and ('mma.sync' in compiled.asm['ptx']) and ('.tf32.' in compiled.asm['ptx'])
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (q.shape_expr[2] + 63) // 64
    launch.grid_y = 1
    launch.grid_z = 4
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(q.shape_expr[2])
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)

@trtp.register('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideTiledFloat32')
def desc_0(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc) -> trtp.TensorDesc:
    assert all((x.dtype == trt.float32 for x in (q, k, pq, e, v))) and mask.dtype == trt.bool
    return v.like()

@trtp.aot_impl('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideTiledFloat32')
def aot_0(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return compile_launch(normal_tile_aot, {'Q': '*fp32', 'K': '*fp32', 'PQ': '*fp32', 'E': '*fp32', 'Mask': '*i1', 'V': '*fp32', 'T': 'i32', 'O': '*fp32'}, q)

@trtp.register('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideStatsWriteFloat32')
def desc_1(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    assert all((x.dtype == trt.float32 for x in (q, k, pq, e, v))) and mask.dtype == trt.bool
    dims = trtp.ShapeExprs(4)
    dims[0] = v.shape_expr[0]
    dims[1] = v.shape_expr[1]
    dims[2] = v.shape_expr[2]
    dims[3] = 2
    stats = v.like()
    stats.shape_expr = dims
    return (v.like(), stats)

@trtp.aot_impl('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideStatsWriteFloat32')
def aot_1(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return compile_launch(normal_stats_write_aot, {'Q': '*fp32', 'K': '*fp32', 'PQ': '*fp32', 'E': '*fp32', 'Mask': '*i1', 'V': '*fp32', 'T': 'i32', 'O': '*fp32', 'Stats': '*fp32'}, q)

@trtp.register('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideStatsReadFloat32')
def desc_2(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc, stats: trtp.TensorDesc) -> trtp.TensorDesc:
    assert all((x.dtype == trt.float32 for x in (q, k, pq, e, v))) and mask.dtype == trt.bool
    return v.like()

@trtp.aot_impl('zipvoice_int8_b1_geo1_normal_tf32::OnlineNormalWideStatsReadFloat32')
def aot_2(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc, stats: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return compile_launch(normal_stats_read_aot, {'Q': '*fp32', 'K': '*fp32', 'PQ': '*fp32', 'E': '*fp32', 'Mask': '*i1', 'V': '*fp32', 'Stats': '*fp32', 'T': 'i32', 'O': '*fp32'}, q)
