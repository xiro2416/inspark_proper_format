"""AOT fusion of protected FP32 FIR math and original terminal INT8 quantization."""
from typing import Tuple,Union
NAME='inspark_custom::small_fir_activation_quantized'


def register():
    import tensorrt as trt
    import tensorrt.plugin as trtp
    try:getattr(trtp.op.inspark_custom,NAME.split('::')[1]);return
    except AttributeError:pass
    @trtp.register(NAME)
    def desc(x:trtp.TensorDesc,up:trtp.TensorDesc,down:trtp.TensorDesc,alpha:trtp.TensorDesc,beta:trtp.TensorDesc,smooth:trtp.TensorDesc,scale:trtp.TensorDesc)->trtp.TensorDesc:
        if x.dtype!=trt.float32 or any(t.dtype!=trt.float32 for t in (up,down,alpha,beta,smooth,scale)):raise ValueError('Source floating FIR math must stay FP32')
        out=x.like();out.dtype=trt.int8;return out
    @trtp.aot_impl(NAME)
    def aot(x,up,down,alpha,beta,smooth,scale,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from deployment.multibatch.fir_quant_kernel import fir_quant
        b,c,f=map(int,x.shape)
        if b not in (1,2,4,8,16,64,128) or int(up.shape[0])!=12 or int(down.shape[0])!=12 or int(alpha.shape[0])!=c or int(beta.shape[0])!=c or int(smooth.shape[0])!=c:raise ValueError('Unchanged FIR/source scale geometry required')
        source=triton.compiler.ASTSource(fn=fir_quant,signature={name:('*i8' if name=='OUT' else '*fp32') for name in ('X','UP','DOWN','ALPHA','BETA','SMOOTH','SCALE','OUT')},
            constexprs=dict(C=c,F=f,BLOCK=256,HIGH=512),attrs={(i,):[['tt.divisibility',16]] for i in range(8)})
        compiled=triton.compile(source,options=dict(num_warps=4,enable_fp_fusion=False))
        launch=trtp.KernelLaunchParams();launch.grid_x=x.shape_expr[0]*x.shape_expr[1];launch.grid_y=trtp.cdiv(x.shape_expr[2],248)
        launch.block_x=compiled.metadata.num_warps*32;launch.shared_mem=compiled.metadata.shared
        return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple([])
