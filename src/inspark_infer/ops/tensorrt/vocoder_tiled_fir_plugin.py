"""AOT adapter for complete-halo, low-resource tiled FIR/Snake/FIR."""
from typing import Tuple,Union
NAME='inspark_custom::small_fir_activation_tiled'


def register():
    import tensorrt.plugin as trtp
    try:getattr(trtp.op.inspark_custom,'small_fir_activation_tiled');return
    except AttributeError:pass
    @trtp.register(NAME)
    def desc(x:trtp.TensorDesc,up:trtp.TensorDesc,down:trtp.TensorDesc,alpha:trtp.TensorDesc,beta:trtp.TensorDesc)->trtp.TensorDesc:
        return x.like()
    @trtp.aot_impl(NAME)
    def aot(x,up,down,alpha,beta,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from inspark_infer.ops.triton.vocoder_tiled_fir import _tiled
        c,f=int(x.shape[1]),int(x.shape[2])
        if int(up.shape[0])!=12 or int(down.shape[0])!=12 or int(alpha.shape[0])!=c or int(beta.shape[0])!=c:
            raise ValueError('Tiled FIR filter/channel geometry mismatch')
        source=triton.compiler.ASTSource(fn=_tiled,signature={key:'*fp32' for key in ('X','UP','DOWN','ALPHA','BETA','OUT')},
            constexprs={'C':c,'F':f,'BLOCK':256,'HIGH':512},attrs={(i,):[['tt.divisibility',16]] for i in range(6)})
        compiled=triton.compile(source,options={'num_warps':4,'enable_fp_fusion':False})
        launch=trtp.KernelLaunchParams()
        launch.grid_x=x.shape_expr[0]*x.shape_expr[1]
        launch.grid_y=trtp.cdiv(x.shape_expr[2],248)
        launch.block_x=compiled.metadata.num_warps*32
        launch.shared_mem=compiled.metadata.shared
        return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple([])
