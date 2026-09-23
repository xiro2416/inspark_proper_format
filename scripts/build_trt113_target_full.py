#!/usr/bin/env python3
"""Build a fixed-batch full 24-layer Target engine with native TRT 11.3."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--batch", type=int, required=True); p.add_argument("--config", default="configs/common/runtime.yaml")
    p.add_argument("--deployment", default="configs/hardware/sm89/sm89_bf16_target_trt113_lab.json")
    p.add_argument("--out-dir", default='artifacts/trt113_target_full')
    p.add_argument("--plan", help="Write a single-batch plan with engine hash and build provenance")
    p.add_argument("--optimization-level", type=int, default=3, choices=range(6))
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from inspark_infer.runtime.config import load as load_config
    from inspark_infer.runtime.deployment import load as load_deployment
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.ops.tensorrt.native113 import _import_trt113
    from trt113_provenance import capture_provenance
    trt = _import_trt113()
    if not trt.__version__.startswith("11.3."):
        raise RuntimeError(f"Expected TensorRT 11.3, got {trt.__version__}")

    config = load_config(args.config); config["max_batch"] = args.batch
    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_deployment(load_deployment(args.deployment))
            provenance = capture_provenance("target", config, args.config, args.deployment, engine)
            tm = engine.rt.target.target.model; blocks = tm.transformer.h; b = args.batch
            logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            keepalive = []; constants_digest = hashlib.sha256()

            def constant(tensor, shape=None, dtype=None):
                value = tensor.detach().cpu().contiguous()
                if dtype == trt.bfloat16 or value.dtype == torch.bfloat16:
                    value = value.bfloat16().view(torch.uint16).numpy().copy(); keepalive.append(value)
                    weights = trt.Weights(trt.bfloat16, value.ctypes.data, value.size)
                else:
                    value = value.float().numpy().copy(); keepalive.append(value); weights = trt.Weights(value)
                constants_digest.update(json.dumps({"shape": list(shape or tuple(tensor.shape)),
                                                    "storage_dtype": str(value.dtype)}, sort_keys=True).encode())
                constants_digest.update(memoryview(value).cast("B"))
                return network.add_constant(shape or tuple(tensor.shape), weights).get_output(0)

            def cast(x, dtype): return network.add_cast(x, dtype).get_output(0)

            def linear(x, module):
                xb = cast(x, trt.bfloat16)
                weight = constant(module.weight, (1, *tuple(module.weight.shape)), trt.bfloat16)
                y = network.add_matrix_multiply(xb, trt.MatrixOperation.NONE,
                                                weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
                if module.bias is not None:
                    bias = constant(module.bias, (1, 1, module.bias.numel()), trt.bfloat16)
                    y = network.add_elementwise(y, bias, trt.ElementWiseOperation.SUM).get_output(0)
                return cast(y, trt.float32)

            def layernorm(x, module):
                scale = constant(module.weight, (1, 1, module.weight.numel()), trt.float32)
                bias = constant(module.bias, (1, 1, module.bias.numel()), trt.float32)
                layer = network.add_normalization_v2(x, scale, bias, 1 << 2); layer.epsilon = module.eps
                return layer.get_output(0)

            hidden = network.add_input("x", trt.float32, (b, 8, 1280))
            mask = network.add_input("mask", trt.bool, (b, 1, 8, 128))
            indices = network.add_input("write_indices", trt.int32, (b,))
            selected = []; cache_names = []
            scale = constant(torch.tensor([0.125]), (1, 1, 1, 1), trt.float32)
            for index, block in enumerate(blocks):
                norm = layernorm(hidden, block.ln_1); qkv = linear(norm, block.attn.c_attn)
                pieces = []
                for offset in (0, 1280, 2560):
                    part = network.add_slice(qkv, (0, 0, offset), (b, 8, 1280), (1, 1, 1)).get_output(0)
                    part = cast(part, trt.bfloat16); shuffle = network.add_shuffle(part)
                    shuffle.reshape_dims = (b, 8, 20, 64); shuffle.second_transpose = trt.Permutation((0, 2, 1, 3))
                    pieces.append(shuffle.get_output(0))
                q, ku, vu = pieces
                ki = network.add_input(f"k_cache_in_{index}", trt.bfloat16, (b, 20, 128, 64))
                vi = network.add_input(f"v_cache_in_{index}", trt.bfloat16, (b, 20, 128, 64))
                ko = network.add_kv_cache_update(ki, ku, indices, trt.KVCacheMode.LINEAR).get_output(0)
                vo = network.add_kv_cache_update(vi, vu, indices, trt.KVCacheMode.LINEAR).get_output(0)
                ko.name = f"k_cache_out_{index}"; vo.name = f"v_cache_out_{index}"
                network.mark_output(ko); network.mark_output(vo); cache_names += [ko.name, vo.name]
                sbf = cast(scale, trt.bfloat16)
                qs = network.add_elementwise(q, sbf, trt.ElementWiseOperation.PROD).get_output(0)
                attn = network.add_attention_v2(qs, ko, vo, trt.AttentionNormalizationOp.SOFTMAX,
                                                trt.CausalMaskKind.NONE)
                attn.mask = mask; attn.decomposable = True
                context = cast(attn.get_output(0), trt.float32)
                shuffle = network.add_shuffle(context); shuffle.first_transpose = trt.Permutation((0, 2, 1, 3))
                shuffle.reshape_dims = (b, 8, 1280); context = shuffle.get_output(0)
                projection = linear(context, block.attn.c_proj)
                hidden = network.add_elementwise(hidden, projection, trt.ElementWiseOperation.SUM).get_output(0)
                ff = linear(layernorm(hidden, block.ln_2), block.mlp.c_fc)
                ff = network.add_activation(ff, trt.ActivationType.GELU_TANH).get_output(0)
                ff = linear(ff, block.mlp.c_proj)
                hidden = network.add_elementwise(hidden, ff, trt.ElementWiseOperation.SUM).get_output(0)
                if index in engine.rt.target.target.target_layer_ids: selected.append(hidden)
            final = layernorm(hidden, tm.transformer.ln_f); final.name = "final"; network.mark_output(final)
            joined = network.add_concatenation(selected); joined.axis = 2
            selected_out = joined.get_output(0); selected_out.name = "selected"; network.mark_output(selected_out)
            # lm_head's extra norm and projection are intentionally FP32.
            logit_input = layernorm(final, tm.lm_head[0]); lm_linear = tm.lm_head[1]
            weight = constant(lm_linear.weight, (1, *tuple(lm_linear.weight.shape)), trt.float32)
            logits = network.add_matrix_multiply(logit_input, trt.MatrixOperation.NONE,
                                                 weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
            if lm_linear.bias is not None:
                bias = constant(lm_linear.bias, (1, 1, lm_linear.bias.numel()), trt.float32)
                logits = network.add_elementwise(logits, bias, trt.ElementWiseOperation.SUM).get_output(0)
            logits.name = "logits"; network.mark_output(logits)
            build = builder.create_builder_config(); build.builder_optimization_level = args.optimization_level
            # lm_head and FP32 interfaces must not silently use single-TF32 GEMM.
            build.clear_flag(trt.BuilderFlag.TF32)
            build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
            started = time.time(); serialized = builder.build_serialized_network(network, build)
            if serialized is None: raise RuntimeError("TensorRT full Target build failed")
            out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
            path = out / f"target_full_b{b}.engine"; path.write_bytes(bytes(serialized))
            artifact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            provenance["constant_data_sha256"] = constants_digest.hexdigest()
            settings = {"optimization_level": args.optimization_level, "workspace_bytes": 8 << 30,
                        "strongly_typed": True, "tf32": bool(build.get_flag(trt.BuilderFlag.TF32)),
                        "batch": b, "query_tokens": 8, "kv_limit": 128,
                        "attention_decomposable": True, "layers": len(blocks)}
            runtime = trt.Runtime(logger); built = runtime.deserialize_cuda_engine(serialized)
            if built is None:
                raise RuntimeError("Built Target engine could not be deserialized")
            tensors = [{"name": built.get_tensor_name(index),
                        "mode": str(built.get_tensor_mode(built.get_tensor_name(index))),
                        "dtype": str(built.get_tensor_dtype(built.get_tensor_name(index))),
                        "shape": list(built.get_tensor_shape(built.get_tensor_name(index)))}
                       for index in range(built.num_io_tensors)]
            report = {"format": 1, "backend": "TensorRT 11.3 native full Target",
                      "precision": "BF16 Linear/KV with FP32 interfaces, norms, residuals and lm_head",
                      "batch": b, "engine": str(path), "bytes": path.stat().st_size, "sha256": artifact_hash,
                      "build_seconds": time.time() - started, "trt": trt.__version__,
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      **provenance["hardware"], "provenance": provenance, "builder_settings": settings,
                      "tf32": settings["tf32"], "strongly_typed": True, "kv_limit": 128,
                      "optimization_level": args.optimization_level, "workspace_bytes": 8 << 30,
                      "tensors": tensors, "cache_outputs": cache_names}
            if args.plan:
                plan_path = Path(args.plan).resolve(); plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan = {key: report[key] for key in ("format", "backend", "precision", "kv_limit", "trt",
                        "torch", "cuda", "gpu_name", "sm", "optimization_level", "workspace_bytes", "tf32", "strongly_typed")}
                plan.update(engines={str(b): os.path.relpath(path, plan_path.parent)},
                            engine_sha256={str(b): artifact_hash}, provenance={str(b): provenance},
                            builder_settings={str(b): settings}, tensors={str(b): tensors},
                            provenance_status="recorded_not_audited")
                plan_path.write_text(json.dumps(plan, indent=2))
            (out / f"target_full_b{b}.json").write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
        finally:
            engine.close()


if __name__ == "__main__": main()
