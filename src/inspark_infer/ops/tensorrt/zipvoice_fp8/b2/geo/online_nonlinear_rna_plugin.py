from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .online_nonlinear_rna_aot import nonlinear_rna_aot

@trtp.register('zipvoice_fp8_sm120_b2_geo::OnlineNonlinearRNAFloat32')
def desc_OnlineNonlinearRNAFloat32(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc) -> trtp.TensorDesc:
    assert all((x.dtype == trt.float32 for x in (q, k, pq, e, v))) and mask.dtype == trt.bool
    return v.like()

@trtp.aot_impl('zipvoice_fp8_sm120_b2_geo::OnlineNonlinearRNAFloat32')
def aot_OnlineNonlinearRNAFloat32(q: trtp.TensorDesc, k: trtp.TensorDesc, pq: trtp.TensorDesc, e: trtp.TensorDesc, mask: trtp.TensorDesc, v: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    compiled = triton.compile(triton.compiler.ASTSource(fn=nonlinear_rna_aot, signature={'Q': '*fp32', 'K': '*fp32', 'PQ': '*fp32', 'E': '*fp32', 'Mask': '*i1', 'V': '*fp32', 'T': 'i32', 'O': '*fp32'}, constexprs={}), options={'num_warps': 4, 'num_stages': 1, 'enable_fp_fusion': False})
    assert compiled.metadata.shared <= 49152 and (not compiled.metadata.global_scratch_size) and (not compiled.metadata.profile_scratch_size)
    assert 'cvt.rna.tf32.f32' in compiled.asm['ptx']
    launch = trtp.KernelLaunchParams()
    launch.grid_x = (q.shape_expr[2] + 15) // 16
    launch.grid_y = 2
    launch.grid_z = 1
    launch.block_x = 128
    launch.shared_mem = compiled.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(q.shape_expr[2])
    return (compiled.metadata.name, compiled.asm['ptx'], launch, extra)
