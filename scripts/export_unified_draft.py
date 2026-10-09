#!/usr/bin/env python3
"""Export Q7 Draft through standard ONNX, with no diagnostic output barriers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128), required=True)
    parser.add_argument("--kv-limit", type=int, default=80)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", default="artifacts/current_release/runtime.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--context',action='store_true',help='Export existing context projection/RMSNorm plus six K/V linears, preserving roles')
    parser.add_argument('--context-gemm-2d',action='store_true',help='Canonical rank2 Gemm bias expression for the context K/V projections')
    parser.add_argument('--context-fp32-bias',action='store_true',help='Explicit candidate: BF16 operands, FP32 accumulation+bias, one final BF16 rounding')
    parser.add_argument('--context-kv-only',action='store_true',help='Only quantized layers1/2 K/V; protected projection and layer0 keep their existing runtime')
    args = parser.parse_args()
    if args.context_gemm_2d and not args.context:parser.error('--context-gemm-2d requires --context')
    if args.context_fp32_bias and not args.context_gemm_2d:parser.error('--context-fp32-bias requires --context-gemm-2d')
    if args.context_kv_only and args.context:parser.error('Choose full context or K/V subset')
    if args.kv_limit < 16 or args.kv_limit % 16:
        parser.error("KV limit must be a positive multiple of 16")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from inspark_infer.build.unified_acoustic_export import export_standard_onnx
    from inspark_infer.build.unified_draft_export import DraftBackbone, DraftContext, DraftContextKV, input_names, prepare_draft_roles
    from inspark_infer.quantization.unified import load_artifact
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.engine import Engine
    from trt113_provenance import capture_onnx_artifact, capture_provenance, file_record

    artifact = load_artifact(args.calibration)
    config = load(args.config)
    config["max_batch"] = args.batch
    output = args.output.resolve()
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                draft = engine.rt.engine.draft
                if len(draft.layers) != 3 or draft.hidden_size != 1280 or draft.vocab_size != 8194:
                    raise ValueError("Export expects the deployed three-layer H1280/V8194 Draft")
                provenance = capture_provenance("draft", config, args.config, model=engine)
                manifest = prepare_draft_roles(engine, artifact)
                if args.context:
                    manifest.update(input_semantics='Target captured hidden [B*8,6400]; native worker owns prefix commit mask',
                                    attention='none: context projection/RMSNorm and six K/V linears only',
                                    fp32_boundaries=['context projection','RMSNorm','K/V outputs'])
                model = (DraftContext(draft,gemm_2d=args.context_gemm_2d,fp32_bf16_bias=args.context_fp32_bias) if args.context else DraftBackbone(draft)).eval()
                if args.context_kv_only:
                    model=DraftContextKV(draft).eval()
                    manifest.update(input_semantics='Already projected/normalized FP32 context [B*8,1280]',
                                    attention='none; only quantized context K/V layers1/2',
                                    fp32_boundaries=['projected context input','K/V output'],
                                    roles=[r for r in manifest['roles'] if r['canonical_path'] in
                                           ('draft.blocks.1.k_proj','draft.blocks.1.v_proj','draft.blocks.2.k_proj','draft.blocks.2.v_proj')])
                device = engine.model.stream.device
                x = torch.zeros(args.batch, 7, 1280, device=device)
                mask = torch.ones(args.batch, 1, 7, args.kv_limit + 7, device=device, dtype=torch.bool)
                caches = tuple(torch.zeros(args.batch, 20, args.kv_limit, 64, device=device)
                               for _ in range(6))
                inputs = (x, mask, *caches)
                names = input_names()
                output_names=['hidden','base']
                if args.context:
                    inputs=(torch.zeros(args.batch*8,6400,device=device),)
                    names=['context_hidden'];output_names=['context','keys','values']
                elif args.context_kv_only:
                    inputs=(torch.zeros(args.batch*8,1280,device=device),)
                    names=['projected_context'];output_names=['keys','values']
                graph = export_standard_onnx(model, inputs, output, input_names=names,
                                              output_names=output_names)
                torch.cuda.synchronize()
                onnx_artifact = capture_onnx_artifact(output)
                specs = {name: value for name, value in artifact["role_specs"].items() if name.startswith("draft.")}
                result = dict(component="draft", kind="context" if args.context else 'context_kv' if args.context_kv_only else "backbone", batch=args.batch, frames=8 if args.context or args.context_kv_only else 7,
                              query_tokens=8 if args.context or args.context_kv_only else 7, kv_limit=args.kv_limit, layers=3,
                              onnx=str(output), bytes=output.stat().st_size,
                              onnx_sha256=onnx_artifact["sha256"], onnx_artifact=onnx_artifact,
                              provenance=provenance, provenance_status=provenance["status"],
                              exporter=file_record(__file__, "onnx_exporter"), plugins=[], plugin_nodes=0,
                              export_settings=dict(opset=20, dynamo=False, external_data=True,
                                                   context_gemm_2d=args.context_gemm_2d,
                                                   protected_context_bias_accumulator='fp32_before_single_bf16_output_round' if args.context_fp32_bias else 'export_default',
                                                   constant_folding=True, batch=args.batch,
                                                   query_tokens=8 if args.context or args.context_kv_only else 7, kv_limit=args.kv_limit, debug_outputs=False,
                                                   attention_graph="standard_onnx_sdpa_decomposition",
                                                   scheme=artifact["scheme"], custom_opsets={},
                                                   precision=artifact["scheme"] + "_static_qdq_fp32_attention_kv_interfaces"),
                              quantization_recipe=dict(scheme=artifact["scheme"],
                                  activation_scaling="offline_static_per_tensor",
                                  calibration=file_record(args.calibration, "calibration_artifact"),
                                  alpha=artifact.get("alpha"), role_manifest=manifest,
                                  role_specs_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()),
                              inputs=[dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                                      for name, value in zip(names, inputs)],
                              output_names=output_names, graph=graph,
                              validation=dict(output_finite=True, numerical_audit="pending_same_recipe_reference",
                                              runtime_integrated=False, performance_validated=False))
                output.with_suffix(".export.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(dict(onnx=str(output), batch=args.batch, kv_limit=args.kv_limit,
                                      scheme=artifact["scheme"], nodes=graph["nodes"], plugins=[])), flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
