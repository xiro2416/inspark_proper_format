from typing import Tuple, Union
import tensorrt as trt
import tensorrt.plugin as trtp
import triton
from .int8_nonlinear_value_aot import nonlinear_value_128x32, nonlinear_value_64x64

def descriptors(q, w, as_, ws, bias):
    assert q.dtype == w.dtype == trt.int8 and all((t.dtype == trt.float32 for t in (as_, ws, bias)))
    assert q.ndim == 3 and w.ndim == 2 and (ws.ndim == bias.ndim == 1)
    if q.has_shape:
        assert tuple(q.shape[1:]) == (4, 512)
    if w.has_shape:
        assert tuple(w.shape) == (1152, 512)
    if as_.has_shape:
        assert as_.numel() == 1
    if ws.has_shape:
        assert tuple(ws.shape) == (1152,)
    if bias.has_shape:
        assert tuple(bias.shape) == (1152,)
    vd = trtp.ShapeExprs(4)
    vd[0] = 1
    vd[1] = 4
    vd[2] = q.shape_expr[0]
    vd[3] = 384
    gd = trtp.ShapeExprs(3)
    gd[0] = q.shape_expr[0]
    gd[1] = 4
    gd[2] = 384
    v = q.like()
    v.dtype = trt.float32
    v.shape_expr = vd
    g = q.like()
    g.dtype = trt.float32
    g.shape_expr = gd
    return (v, g)

def launch(fn, q, bm, bn):
    c = triton.compile(triton.compiler.ASTSource(fn=fn, signature={'Q': '*i8', 'W': '*i8', 'AS': '*fp32', 'WS': '*fp32', 'Bias': '*fp32', 'T': 'i32', 'V': '*fp32', 'G': '*fp32'}, constexprs={}), options={'num_warps': 4, 'num_stages': 2, 'enable_fp_fusion': False})
    assert c.metadata.shared <= 49152 and (not c.metadata.global_scratch_size) and (not c.metadata.profile_scratch_size)
    assert '.s32.s8.s8.s32' in c.asm['ptx'] and 'tf32' not in c.asm['ptx'].lower()
    p = trtp.KernelLaunchParams()
    p.grid_x = (q.shape_expr[0] * 4 + bm - 1) // bm
    p.grid_y = (384 + bn - 1) // bn
    p.block_x = 128
    p.shared_mem = c.metadata.shared
    extra = trtp.SymIntExprs(1)
    extra[0] = trtp.SymInt32(q.shape_expr[0])
    return (c.metadata.name, c.asm['ptx'], p, extra)

@trtp.register('zipvoice_int8_b4::OriginalInt8NonlinearValue128x32')
def desc_128(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    return descriptors(q, w, as_, ws, bias)

@trtp.aot_impl('zipvoice_int8_b4::OriginalInt8NonlinearValue128x32')
def aot_128(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return launch(nonlinear_value_128x32, q, 128, 32)

@trtp.register('zipvoice_int8_b4::OriginalInt8NonlinearValue64x64')
def desc_64(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    return descriptors(q, w, as_, ws, bias)

@trtp.aot_impl('zipvoice_int8_b4::OriginalInt8NonlinearValue64x64')
def aot_64(q: trtp.TensorDesc, w: trtp.TensorDesc, as_: trtp.TensorDesc, ws: trtp.TensorDesc, bias: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:
    return launch(nonlinear_value_64x64, q, 64, 64)
