#!/usr/bin/env python3
"""Locate the first numerical divergence in the native TRT 11.3 Draft graph."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,default=6)
    p.add_argument("--engine",default='../artifacts/trt113_draft_debug/draft_full_b1.engine')
    p.add_argument("--config",default="configs/runtime.yaml")
    p.add_argument("--deployment",default="configs/sm89_bf16_current_fair.json")
    p.add_argument("--ref-audio",required=True);p.add_argument("--out",required=True);args=p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    import torch
    from acc_infer_clear.runtime.config import load as load_config
    from acc_infer_clear.ops.triton.draft_attention import attention
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.engine import Engine
    from acc_infer_clear.ops.tensorrt.native113 import _import_trt113, _make_draft_mask

    config=load_config(args.config);config["max_batch"]=1;engine=Engine(config)
    try:
        engine.prepare_reference("reference",args.ref_audio)
        engine.prepare_deployment(load_deployment(args.deployment));draft=engine.rt.backbone;m=draft.model
        param=next(m.parameters());anchors=torch.zeros(1,device=param.device,dtype=torch.long)
        positions=torch.arange(7,device=param.device)[None];slots=torch.zeros(1,device=param.device,dtype=torch.int32)
        lengths=torch.full((1,),121,device=param.device,dtype=torch.int32)
        with torch.inference_mode():
            draft.pool.storage.zero_();x=m._noise_embeddings(anchors,positions);hidden=x.clone();reference={}
            for index,layer in enumerate(m.layers):
                norm=layer.input_norm(hidden);q=layer._heads(layer.q_proj(norm));k=layer._heads(layer.k_proj(norm));v=layer._heads(layer.v_proj(norm))
                if index==0:
                    reference.update(layer0_input_norm=norm.clone(),layer0_q=q.clone(),layer0_k=k.clone(),layer0_v=v.clone())
                attended=attention(q,k,v,draft.pool.storage[index,0],draft.pool.storage[index,1],slots,lengths,128,draft.consumer_layout)
                if index==0:reference["layer0_context_heads"]=attended.clone()
                consumer=attended.reshape_as(hidden) if draft.consumer_layout else attended.transpose(1,2).contiguous().view_as(hidden)
                projection=layer.o_proj(consumer)
                if index==0:reference["layer0_attention_projection"]=projection.clone()
                hidden=hidden+projection;reference[f"layer{index}_attention_residual"]=hidden.clone()
                postnorm=layer.post_norm(hidden);ff_up=layer.mlp[0](postnorm);ff_act=layer.mlp[1](ff_up);ff=layer.mlp[2](ff_act)
                if index==0:reference.update(layer0_post_norm=postnorm.clone(),layer0_ff_up=ff_up.clone(),layer0_ff_act=ff_act.clone(),layer0_ff_down=ff.clone())
                hidden=hidden+ff;reference[f"layer{index}_mlp_residual"]=hidden.clone()
                hidden=m.apply_query_temporal(hidden,index)
            reference["hidden"]=m.project_output(hidden);reference["base"]=m.base_logits(reference["hidden"])

        trt=_import_trt113();runtime=trt.Runtime(trt.Logger(trt.Logger.ERROR))
        trt_engine=runtime.deserialize_cuda_engine(Path(args.engine).read_bytes());context=trt_engine.create_execution_context()
        mask=torch.empty(1,1,7,135,device=param.device,dtype=torch.bool)
        _make_draft_mask[(1,)](lengths,mask,135,num_warps=4,num_stages=1)
        compact=draft.pool.storage[:,:,:1,:,:128].contiguous();buffers={"x":x,"mask":mask}
        for layer in range(3):
            buffers[f"k_cache_{layer}"]=compact[layer,0]
            buffers[f"v_cache_{layer}"]=compact[layer,1]
        dtype_map={trt.float32:torch.float32,trt.bfloat16:torch.bfloat16,trt.bool:torch.bool,trt.int32:torch.int32}
        for i in range(trt_engine.num_io_tensors):
            name=trt_engine.get_tensor_name(i)
            if trt_engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT:
                buffers[name]=torch.empty(tuple(trt_engine.get_tensor_shape(name)),device=param.device,dtype=dtype_map[trt_engine.get_tensor_dtype(name)])
        for name,value in buffers.items():
            if not context.set_tensor_address(name,value.data_ptr()):raise RuntimeError(f"binding rejected: {name}")
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError("TRT enqueue failed")
        torch.cuda.synchronize()

        def delta(a,b):
            a=a.float();b=b.float();d=(a-b).abs()
            return {"max_abs":float(d.max()),"mean_abs":float(d.mean()),
                    "cosine":float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}
        result={name:delta(reference[name],buffers[name]) for name in reference}
        with torch.inference_mode():
            ref_context=reference["layer0_context_heads"]
            trt_context=buffers["layer0_context_heads"]
            trt_consumer=trt_context.transpose(1,2).contiguous().view(1,7,1280)
            cross_projection=m.layers[0].o_proj(trt_consumer)
            result["layer0_context_bf16_mismatch"]={
                "elements":ref_context.numel(),
                "different":int((ref_context.bfloat16()!=trt_context.bfloat16()).sum()),
            }
            result["layer0_current_gemm_on_trt_context_vs_reference"]=delta(
                reference["layer0_attention_projection"],cross_projection)
            result["layer0_current_gemm_on_trt_context_vs_trt_gemm"]=delta(
                cross_projection,buffers["layer0_attention_projection"])
        Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        engine.close()


if __name__=="__main__":main()
