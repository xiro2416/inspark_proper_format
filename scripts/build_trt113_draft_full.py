#!/usr/bin/env python3
"""Build a fixed-batch full three-layer Draft backbone with native TRT 11.3."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,default=6)
    p.add_argument("--batch",type=int,required=True);p.add_argument("--config",default="configs/runtime.yaml")
    p.add_argument("--deployment",default="configs/sm89_bf16_target_trt113_lab.json")
    p.add_argument("--out-dir",default="artifacts/trt113_draft_full")
    p.add_argument("--optimization-level",type=int,default=5,choices=range(6))
    p.add_argument("--stable-block-linears",action="store_true")
    p.add_argument("--debug-outputs",action="store_true");args=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.tensorrt_backend.native113 import _import_trt113
    trt=_import_trt113();config=load_config(args.config);config["max_batch"]=args.batch
    with GPULease(args.gpu):
        engine=Engine(config)
        try:
            engine.prepare_deployment(load_deployment(args.deployment));m=engine.rt.engine.draft;b=args.batch
            if len(m.layers)!=3 or m.hidden_size!=1280 or m.block_size!=7:
                raise ValueError("Builder is specialized for the deployed three-layer H1280/Q7 Draft")
            logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
            network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED));keep=[]

            def constant(tensor,shape=None,dtype=None):
                value=tensor.detach().cpu().contiguous()
                if dtype==trt.bfloat16 or value.dtype==torch.bfloat16:
                    value=value.bfloat16().view(torch.uint16).numpy().copy();keep.append(value)
                    weights=trt.Weights(trt.bfloat16,value.ctypes.data,value.size)
                else:
                    value=value.float().numpy().copy();keep.append(value);weights=trt.Weights(value)
                return network.add_constant(shape or tuple(tensor.shape),weights).get_output(0)
            def cast(x,dtype):return network.add_cast(x,dtype).get_output(0)
            def linear(x,module):
                xb=cast(x,trt.bfloat16);weight=constant(module.weight,(1,*tuple(module.weight.shape)),trt.bfloat16)
                y=network.add_matrix_multiply(xb,trt.MatrixOperation.NONE,weight,trt.MatrixOperation.TRANSPOSE).get_output(0)
                if module.bias is not None:
                    bias=constant(module.bias,(1,1,module.bias.numel()),trt.bfloat16)
                    y=network.add_elementwise(y,bias,trt.ElementWiseOperation.SUM).get_output(0)
                return cast(y,trt.float32)
            def linear_fp32(x,module):
                # The deployed precision contract converts only Draft block
                # linears to BF16.  lm_head remains FP32 because its logits feed
                # discrete proposal sampling and are especially error-sensitive.
                weight=constant(module.weight,(1,*tuple(module.weight.shape)),trt.float32)
                y=network.add_matrix_multiply(x,trt.MatrixOperation.NONE,weight,trt.MatrixOperation.TRANSPOSE).get_output(0)
                if module.bias is not None:
                    bias=constant(module.bias,(1,1,module.bias.numel()),trt.float32)
                    y=network.add_elementwise(y,bias,trt.ElementWiseOperation.SUM).get_output(0)
                return y
            def linear_stable_bf16(x,module):
                # Match MatrixLinear's BF16 input/weight/output boundaries while
                # preventing a numerically unstable native BF16 reduction tactic.
                xb=cast(x,trt.bfloat16);xf=cast(xb,trt.float32)
                weight=constant(module.weight.float(),(1,*tuple(module.weight.shape)),trt.float32)
                y=network.add_matrix_multiply(xf,trt.MatrixOperation.NONE,weight,trt.MatrixOperation.TRANSPOSE).get_output(0)
                if module.bias is not None:
                    bias=constant(module.bias.float(),(1,1,module.bias.numel()),trt.float32)
                    y=network.add_elementwise(y,bias,trt.ElementWiseOperation.SUM).get_output(0)
                return cast(cast(y,trt.bfloat16),trt.float32)
            def rmsnorm(x,module):
                axes=1 << 2
                square=network.add_elementwise(x,x,trt.ElementWiseOperation.PROD).get_output(0)
                mean=network.add_reduce(square,trt.ReduceOperation.AVG,axes,True).get_output(0)
                eps=constant(torch.tensor([module.eps]),(1,1,1),trt.float32)
                denom=network.add_unary(network.add_elementwise(mean,eps,trt.ElementWiseOperation.SUM).get_output(0),trt.UnaryOperation.SQRT).get_output(0)
                normalized=network.add_elementwise(x,denom,trt.ElementWiseOperation.DIV).get_output(0)
                scale=constant(module.weight,(1,1,module.weight.numel()),trt.float32)
                return network.add_elementwise(normalized,scale,trt.ElementWiseOperation.PROD).get_output(0)

            hidden=network.add_input("x",trt.float32,(b,7,1280));mask=network.add_input("mask",trt.bool,(b,1,7,135))
            scale=constant(torch.tensor([0.125]),(1,1,1,1),trt.float32)
            for index,layer in enumerate(m.layers):
                normalized=rmsnorm(hidden,layer.input_norm);parts=[]
                if args.debug_outputs and index==0:
                    normalized.name="layer0_input_norm";network.mark_output(normalized)
                for module in (layer.q_proj,layer.k_proj,layer.v_proj):
                    part=linear(normalized,module)
                    shuffle=network.add_shuffle(part);shuffle.reshape_dims=(b,7,20,64);shuffle.second_transpose=trt.Permutation((0,2,1,3));parts.append(shuffle.get_output(0))
                q,k,v=parts
                if args.debug_outputs and index==0:
                    for name,tensor in (("layer0_q",q),("layer0_k",k),("layer0_v",v)):
                        tensor.name=name;network.mark_output(tensor)
                ki=network.add_input(f"k_cache_{index}",trt.float32,(b,20,128,64))
                vi=network.add_input(f"v_cache_{index}",trt.float32,(b,20,128,64))
                keys=network.add_concatenation((ki,k));keys.axis=2;values=network.add_concatenation((vi,v));values.axis=2
                qs=network.add_elementwise(q,scale,trt.ElementWiseOperation.PROD).get_output(0)
                attn=network.add_attention_v2(qs,keys.get_output(0),values.get_output(0),trt.AttentionNormalizationOp.SOFTMAX,trt.CausalMaskKind.NONE)
                attn.mask=mask;attn.decomposable=True;context=cast(attn.get_output(0),trt.float32)
                if args.debug_outputs and index==0:
                    context.name="layer0_context_heads";network.mark_output(context)
                shuffle=network.add_shuffle(context);shuffle.first_transpose=trt.Permutation((0,2,1,3));shuffle.reshape_dims=(b,7,1280)
                block_linear=linear_stable_bf16 if args.stable_block_linears else linear
                projection=block_linear(shuffle.get_output(0),layer.o_proj)
                if args.debug_outputs and index==0:
                    projection.name="layer0_attention_projection";network.mark_output(projection)
                hidden=network.add_elementwise(hidden,projection,trt.ElementWiseOperation.SUM).get_output(0)
                if args.debug_outputs:
                    hidden.name=f"layer{index}_attention_residual";network.mark_output(hidden)
                postnorm=rmsnorm(hidden,layer.post_norm);ff_up=block_linear(postnorm,layer.mlp[0]);ff_act=network.add_activation(ff_up,trt.ActivationType.GELU_TANH).get_output(0);ff=block_linear(ff_act,layer.mlp[2])
                if args.debug_outputs and index==0:
                    for name,tensor in (("layer0_post_norm",postnorm),("layer0_ff_up",ff_up),("layer0_ff_act",ff_act),("layer0_ff_down",ff)):
                        tensor.name=name;network.mark_output(tensor)
                hidden=network.add_elementwise(hidden,ff,trt.ElementWiseOperation.SUM).get_output(0)
                if args.debug_outputs:
                    hidden.name=f"layer{index}_mlp_residual";network.mark_output(hidden)
            output=rmsnorm(hidden,m.output_norm);output.name="hidden";network.mark_output(output)
            base=linear_fp32(output,m.lm_head);base.name="base";network.mark_output(base)
            build=builder.create_builder_config();build.builder_optimization_level=args.optimization_level
            # Current Draft attention deliberately uses Triton tf32x3, which is
            # much closer to FP32 than a single-TF32 TensorRT tactic.  Keep the
            # explicitly BF16 block GEMMs, but forbid TF32 contraction for the
            # FP32 attention and FP32 lm_head portions of this engine.
            build.clear_flag(trt.BuilderFlag.TF32)
            build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,8<<30)
            started=time.time();serialized=builder.build_serialized_network(network,build)
            if serialized is None:raise RuntimeError("TensorRT full Draft build failed")
            out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True);path=out/f"draft_full_b{b}.engine";path.write_bytes(bytes(serialized))
            report={"batch":b,"engine":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"build_seconds":time.time()-started,"trt":trt.__version__,"optimization_level":args.optimization_level,"workspace_bytes":8<<30,"precision":"BF16 block Linear with FP32 attention/KV/norms/residuals/lm_head/interfaces","tf32":False,"debug_outputs":args.debug_outputs,"stable_block_linears":args.stable_block_linears,"qkv":"three eager-origin projections"}
            (out/f"draft_full_b{b}.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
        finally:engine.close()


if __name__=="__main__":main()
