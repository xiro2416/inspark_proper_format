"""AOT TensorRT adapter for the exact short-row FIR/Snake/FIR kernel."""
from typing import Tuple,Union
NAME='inspark_custom::small_fir_activation'


def register():
    from inspark_infer.ops.tensorrt.vocoder_tiled_fir_plugin import register as register_tiled
    register_tiled()
    import tensorrt.plugin as trtp
    try:getattr(trtp.op.inspark_custom,'small_fir_activation');return
    except AttributeError:pass
    @trtp.register(NAME)
    def desc(x:trtp.TensorDesc,up:trtp.TensorDesc,down:trtp.TensorDesc,alpha:trtp.TensorDesc,beta:trtp.TensorDesc)->trtp.TensorDesc:
        return x.like()
    @trtp.aot_impl(NAME)
    def aot(x,up,down,alpha,beta,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from inspark_infer.ops.triton.vocoder_small_fir import _small
        c=int(x.shape[1]);f=int(x.shape[2])
        if f>2048:raise ValueError('Only short vocoder rows selected')
        src=triton.compiler.ASTSource(fn=_small,signature={k:'*fp32' for k in ('X','UP','DOWN','ALPHA','BETA','OUT')},
            constexprs={'C':c,'F':f,'HIGH':triton.next_power_of_2(2*f),'LOW':triton.next_power_of_2(f)},attrs={(i,):[['tt.divisibility',16]] for i in range(6)})
        compiled=triton.compile(src,options={'num_warps':8,'enable_fp_fusion':False})
        launch=trtp.KernelLaunchParams();launch.grid_x=x.shape_expr[0]*x.shape_expr[1];launch.block_x=compiled.metadata.num_warps*32;launch.shared_mem=compiled.metadata.shared
        return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple([])
