"""AOT TensorRT integration for source-recipe signed INT8 implicit convolution."""
from typing import Tuple,Union
NAME='inspark_custom::implicit_int8_conv_1d'
TUNED_NAME='inspark_custom::implicit_int8_conv_1d_tuned'
MIGRATED_NAME='inspark_custom::implicit_int8_conv_1d_migrated'


def register(tuned=False,migrated=False):
    import tensorrt as trt
    import tensorrt.plugin as trtp
    name=MIGRATED_NAME if migrated else TUNED_NAME if tuned else NAME
    try:getattr(trtp.op.inspark_custom,name.split('::')[1]);return
    except AttributeError:pass
    @trtp.register(name)
    def desc(q:trtp.TensorDesc,w:trtp.TensorDesc,act:trtp.TensorDesc,ws:trtp.TensorDesc,bias:trtp.TensorDesc,
             stride:int,pad:int,dilation:int,expand:int,output_length:int)->trtp.TensorDesc:
        if q.dtype!=trt.int8 or w.dtype!=trt.int8:raise ValueError('Implicit convolution requires actual signed INT8 operands')
        output=q.like();output.dtype=trt.float32
        output.shape_expr=trtp.ShapeExprs.from_tuple((q.shape_expr[0],w.shape_expr[0],output_length))
        return output
    @trtp.aot_impl(name)
    def aot(q,w,act,ws,bias,stride:int,pad:int,dilation:int,expand:int,output_length:int,outputs,tactic:int)->Tuple[Union[str,bytes],Union[str,bytes],trtp.KernelLaunchParams,trtp.SymExprs]:
        import triton
        from deployment.b32.implicit_int8_probe import implicit_conv
        b,c,f=map(int,q.shape);o,ck=map(int,w.shape)
        if ck%c or int(ws.shape[0])!=o or int(bias.shape[0])!=o:raise ValueError('Implicit convolution source geometry mismatch')
        k=ck//c
        bm,bn,bk=(64,64,64) if tuned or migrated else (32,32,64)
        if tuned and (b,c,f,o,k)!=(32,192,1664,192,11):raise ValueError('Tuned schedule restricted to the measured source region')
        if migrated and (b not in (1,2,4,8,16,64,128) or (c,f,o,k)!=(192,1664,192,11)):raise ValueError('Migrated source schedule requires authorized batch and exact original region')
        source=triton.compiler.ASTSource(fn=implicit_conv,
            signature={name:('*i8' if name in ('Q','W') else '*fp32') for name in ('Q','W','ACT','WS','BIAS','Y')},
            constexprs=dict(B=b,C=c,F=f,O=o,K=k,L=output_length,STRIDE=stride,PAD=pad,DILATION=dilation,EXPAND=expand,BM=bm,BN=bn,BK=bk),
            attrs={(i,):[['tt.divisibility',16]] for i in range(6)})
        compiled=triton.compile(source,options=dict(num_warps=4,num_stages=3,enable_fp_fusion=False))
        if 'mma.sync' not in compiled.asm['ptx'] or '.s8.s8.s32' not in compiled.asm['ptx']:raise RuntimeError('Implicit convolution lost actual INT8 MMA')
        launch=trtp.KernelLaunchParams();launch.grid_x=trtp.cdiv(q.shape_expr[0]*output_length,bm);launch.grid_y=trtp.cdiv(w.shape_expr[0],bn)
        launch.block_x=compiled.metadata.num_warps*32;launch.shared_mem=compiled.metadata.shared
        return compiled.metadata.name,compiled.asm['ptx'],launch,trtp.SymIntExprs.from_tuple([])
