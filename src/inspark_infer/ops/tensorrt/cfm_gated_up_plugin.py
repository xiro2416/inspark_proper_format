"""AOT TensorRT adapters: opaque FP8 bytes and register-resident SwiGLU."""
from typing import Tuple,Union
NAME='inspark_custom::fp8_gated_up'
QUANT='inspark_custom::fp8_quantize'


def register():
    from inspark_infer.ops.tensorrt.cfm_interleaved_plugin import register as register_interleaved
    register_interleaved()
    import tensorrt as trt
    import tensorrt.plugin as trtp
    try:
        getattr(trtp.op.inspark_custom,'fp8_gated_up');getattr(trtp.op.inspark_custom,'fp8_quantize')
        return
    except AttributeError:pass

    @trtp.register(QUANT)
    def quant_desc(x:trtp.TensorDesc,input_scale:float)->trtp.TensorDesc:
        output=x.like();output.dtype=trt.int32;output.shape_expr[-1]=trtp.cdiv(x.shape_expr[-1],4)
        return output

    @trtp.aot_impl(QUANT)
    def quant_aot(x,input_scale:float,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from inspark_infer.ops.triton.cfm_gated_up import _quant_fp8_raw
        total=1
        for v in x.shape:total*=int(v)
        src=triton.compiler.ASTSource(fn=_quant_fp8_raw,signature={'X':'*fp32','OUT':'*i32'},constexprs={'TOTAL':total,'SCALE':input_scale,'BLOCK':1024},attrs={(i,):[['tt.divisibility',16]] for i in range(2)})
        c=triton.compile(src,options={'num_warps':4,'enable_fp_fusion':False})
        p=trtp.KernelLaunchParams();p.grid_x=trtp.cdiv(x.shape_expr.numel(),1024);p.block_x=c.metadata.num_warps*32;p.shared_mem=c.metadata.shared
        return c.metadata.name,c.asm['ptx'],p,trtp.SymIntExprs.from_tuple([])

    @trtp.register(NAME)
    def desc(x:trtp.TensorDesc,w1:trtp.TensorDesc,w3:trtp.TensorDesc,
             s1:trtp.TensorDesc,s3:trtp.TensorDesc,input_scale:float,output_scale:float)->trtp.TensorDesc:
        output=x.like();output.dtype=trt.fp8;output.shape_expr[-1]=w1.shape_expr[0]
        return output

    @trtp.aot_impl(NAME)
    def aot(x,w1,w3,s1,s3,input_scale:float,output_scale:float,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from inspark_infer.ops.triton.cfm_gated_up import _paired_raw
        m=int(x.shape[0])*int(x.shape[1]);n=int(w1.shape[0]);k=int(w1.shape[1])*4
        src=triton.compiler.ASTSource(fn=_paired_raw,signature={'A':'*i32','W1':'*i32','W3':'*i32','S1':'*fp32','S3':'*fp32','OUT':'*fp8e4nv'},
            constexprs={'M':m,'N':n,'K':k,'OUT_SCALE':output_scale,'BM':64,'BN':128,'BK':64,'DEQUANT':False},attrs={(i,):[['tt.divisibility',16]] for i in range(6)})
        c=triton.compile(src,options={'num_warps':8,'num_stages':2,'enable_fp_fusion':False})
        p=trtp.KernelLaunchParams();p.grid_x=trtp.cdiv(x.shape_expr[0]*x.shape_expr[1],64);p.grid_y=trtp.cdiv(w1.shape_expr[0],128)
        p.block_x=c.metadata.num_warps*32;p.shared_mem=c.metadata.shared
        return c.metadata.name,c.asm['ptx'],p,trtp.SymIntExprs.from_tuple([])
