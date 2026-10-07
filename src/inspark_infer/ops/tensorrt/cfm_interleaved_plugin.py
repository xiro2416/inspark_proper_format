"""AOT plugin for immutable interleaved gate/up weights, FP8 in and out."""
from typing import Tuple, Union
NAME = 'inspark_custom::fp8_gated_up_interleaved'


def register():
    import tensorrt as trt
    import tensorrt.plugin as trtp
    try:
        getattr(trtp.op.inspark_custom, 'fp8_gated_up_interleaved')
        return
    except AttributeError:
        pass

    @trtp.register(NAME)
    def desc(x: trtp.TensorDesc, w: trtp.TensorDesc, s: trtp.TensorDesc,
             input_scale: float, output_scale: float) -> trtp.TensorDesc:
        output = x.like()
        output.dtype = trt.fp8
        output.shape_expr[-1] = w.shape_expr[0] // 2
        return output

    @trtp.aot_impl(NAME)
    def aot(x, w, s, input_scale: float, output_scale: float, outputs, tactic: int) -> Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from inspark_infer.ops.triton.cfm_interleaved import _interleaved_raw
        m = int(x.shape[0]) * int(x.shape[1])
        n = int(w.shape[0]) // 2
        k = int(w.shape[1]) * 4
        if int(w.shape[0]) % 2 or int(s.shape[0]) != 2*n or int(x.shape[-1])*4 != k:
            raise ValueError('Interleaved gate/up geometry mismatch')
        src = triton.compiler.ASTSource(fn=_interleaved_raw,
            signature={'A':'*i32','W':'*i32','S':'*fp32','OUT':'*fp8e4nv'},
            constexprs={'M':m,'N':n,'K':k,'SCALE':output_scale,'BM':128,'BN':128,'BK':128},
            attrs={(i,):[['tt.divisibility',16]] for i in range(4)})
        c = triton.compile(src, options={'num_warps':8,'num_stages':2,'enable_fp_fusion':False})
        p = trtp.KernelLaunchParams()
        p.grid_x = trtp.cdiv(x.shape_expr[0]*x.shape_expr[1],128)*trtp.cdiv(w.shape_expr[0],128)
        p.block_x = c.metadata.num_warps*32
        p.shared_mem = c.metadata.shared
        return c.metadata.name, c.asm['ptx'], p, trtp.SymIntExprs.from_tuple([])
