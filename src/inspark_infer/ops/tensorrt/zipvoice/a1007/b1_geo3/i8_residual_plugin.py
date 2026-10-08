from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .i8_residual_aot import residual_64x64, residual_32x64

def descriptors(q, w, as_, ws, bias, r):
    assert q.dtype == w.dtype == trt.int8 and all((x.dtype == trt.float32 for x in (as_, ws, bias, r)))
    assert q.ndim == r.ndim == 3 and w.ndim == 2 and (ws.ndim == bias.ndim == 1)
    if q.has_shape:
        assert tuple(q.shape[1:]) == (1, 48)
    if w.has_shape:
        assert tuple(w.shape) == (512, 48)
    if r.has_shape:
        assert tuple(r.shape[1:]) == (1, 512)
    if as_.has_shape:
        assert as_.numel() == 1
    if ws.has_shape:
        assert tuple(ws.shape) == (512,)
    if bias.has_shape:
        assert tuple(bias.shape) == (512,)
    shape = trtp.ShapeExprs(3)
    shape[0] = q.shape_expr[0]
    shape[1] = 1
    shape[2] = 512
    y = r.like()
    y.shape_expr = shape
    return y

def launch(fn, q, bm):
    c = triton.compile(triton.compiler.ASTSource(fn=fn, signature={'Q': '*i8', 'W': '*i8', 'AS': '*fp32', 'WS': '*fp32', 'Bias': '*fp32', 'R': '*fp32', 'T': 'i32', 'Y': '*fp32'}, constexprs={}), options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})
    assert c.metadata.shared <= 49152 and (not c.metadata.global_scratch_size) and (not c.metadata.profile_scratch_size)
    assert '.s32.s8.s8.s32' in c.asm['ptx'] and 'tf32' not in c.asm['ptx'].lower()
    p = trtp.KernelLaunchParams()
    p.grid_x = (q.shape_expr[0] * 1 + bm - 1) // bm
    p.grid_y = 8
    p.block_x = 128
    p.shared_mem = c.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(q.shape_expr[0])
    return (c.metadata.name, c.asm['ptx'], p, extra)

@trtp.register('zipvoice_int8_b1_geo3::OriginalInt8Residual64x64')
def desc64(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, r: trtp.TensorDesc) -> trtp.TensorDesc:
    return descriptors(q, w, as_, ws, bias, r)

@trtp.aot_impl('zipvoice_int8_b1_geo3::OriginalInt8Residual64x64')
def aot64(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, r: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return launch(residual_64x64, q, 64)

@trtp.register('zipvoice_int8_b1_geo3::OriginalInt8Residual32x64')
def desc128(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, r: trtp.TensorDesc) -> trtp.TensorDesc:
    return descriptors(q, w, as_, ws, bias, r)

@trtp.aot_impl('zipvoice_int8_b1_geo3::OriginalInt8Residual32x64')
def aot128(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, r: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return launch(residual_32x64, q, 32)
