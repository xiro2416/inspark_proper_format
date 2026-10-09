"""Explicit target-measured schedules; leaves migrated serialized APIs intact."""
from typing import Tuple,Union
NAME='inspark_custom::implicit_int8_conv_1d_schedule'
TILES=((32,32,64),(32,64,64),(32,128,64),(64,64,64),(64,128,64),(64,64,128),(128,64,64))


def register():
    import tensorrt as trt
    import tensorrt.plugin as trtp
    try:getattr(trtp.op.inspark_custom,NAME.split('::')[1]);return
    except AttributeError:pass
    @trtp.register(NAME)
    def desc(q:trtp.TensorDesc,w:trtp.TensorDesc,act:trtp.TensorDesc,ws:trtp.TensorDesc,bias:trtp.TensorDesc,
             stride:int,pad:int,dilation:int,expand:int,output_length:int,bm:int,bn:int,bk:int)->trtp.TensorDesc:
        if q.dtype!=trt.int8 or w.dtype!=trt.int8 or (bm,bn,bk) not in TILES:raise ValueError('Invalid signed INT8 schedule')
        output=q.like();output.dtype=trt.float32
        output.shape_expr=trtp.ShapeExprs.from_tuple((q.shape_expr[0],w.shape_expr[0],output_length))
        return output
    @trtp.aot_impl(NAME)
    def aot(q,w,act,ws,bias,stride:int,pad:int,dilation:int,expand:int,output_length:int,bm:int,bn:int,bk:int,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from deployment.b32.implicit_int8_probe import implicit_conv
        b,c,f=map(int,q.shape);o,ck=map(int,w.shape)
        if (bm,bn,bk) not in TILES or b not in (1,2,4,8,16,64,128) or c<=0 or o<=0 or ck%c or ck//c>64 or int(ws.shape[0])!=o or int(bias.shape[0])!=o:
            raise ValueError('Only target-probed original source-recipe convolution regions')
        source=triton.compiler.ASTSource(fn=implicit_conv,
            signature={name:('*i8' if name in ('Q','W') else '*fp32') for name in ('Q','W','ACT','WS','BIAS','Y')},
            constexprs=dict(B=b,C=c,F=f,O=o,K=ck//c,L=output_length,STRIDE=stride,PAD=pad,DILATION=dilation,EXPAND=expand,BM=bm,BN=bn,BK=bk),
            attrs={(i,):[['tt.divisibility',16]] for i in range(6)})
        compiled=triton.compile(source,options=dict(num_warps=4,num_stages=3,enable_fp_fusion=False))
        if 'mma.sync' not in compiled.asm['ptx'] or '.s8.s8.s32' not in compiled.asm['ptx']:raise RuntimeError('Lost signed INT8 MMA')
        launch=trtp.KernelLaunchParams();launch.grid_x=trtp.cdiv(q.shape_expr[0]*output_length,bm);launch.grid_y=trtp.cdiv(w.shape_expr[0],bn)
        launch.block_x=compiled.metadata.num_warps*32;launch.shared_mem=compiled.metadata.shared
        return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple([])
