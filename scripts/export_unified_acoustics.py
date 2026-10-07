#!/usr/bin/env python3
"""Export calibrated first-chunk CFM/BigVGAN using only standard ONNX ops.

This runs on exactly one explicitly selected GPU. It does not invoke the legacy
FP8 Triton preparation or install Vocoder Quick Plugins. CPU tests exercise the
same graph conversion functions with small independent model instances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--component", choices=("cfm", "vocoder"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 8, 64, 128), required=True)
    parser.add_argument("--kind", choices=("estimator", "full_solver"), default="estimator")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", default="artifacts/current_release/runtime.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deconv-as-conv", action="store_true",
                        help="Rewrite learned ConvTranspose as constant zero insertion plus Conv, preserving dtype/QDQ")
    parser.add_argument("--fir-polyphase", action="store_true",
                        help="Rewrite fixed ratio-2/12-tap FP32 FIR upsampling as grouped Conv plus phase interleave")
    parser.add_argument("--cfm-sdpa-bf16", action="store_true",
                        help="Use BF16 Q/K/V only after CFM RoPE and restore SDPA output to FP32")
    parser.add_argument("--cfm-wavenet-tail", action="store_true",
                        help="Prune masked-prompt WaveNet outputs, retaining the exact tail convolution halo")
    parser.add_argument('--validation-capture', type=Path,
                        help='Real frozen acoustic inputs for an export-only graph equivalence check')
    args = parser.parse_args()
    if args.fir_polyphase and args.component != "vocoder":
        parser.error("--fir-polyphase applies only to the Vocoder fixed FIR filters")
    if args.cfm_sdpa_bf16 and args.component != "cfm":
        parser.error("--cfm-sdpa-bf16 applies only to CFM Attention")
    if args.cfm_wavenet_tail and (args.component != 'cfm' or args.kind != 'full_solver'):
        parser.error('--cfm-wavenet-tail requires a full CFM solver with the fixed P258 prompt mask')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from inspark_infer.build.unified_acoustic_export import (
        INTERVALS, FourStepCFM, TailWaveNetDiT, acoustic_specs_from_artifact,
        export_standard_onnx, prepare_acoustic_model,
    )
    from inspark_infer.quantization.unified import iter_roles, load_artifact
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease, select_gpu
    from inspark_infer.runtime.engine import Engine
    from trt113_provenance import capture_onnx_artifact, capture_provenance, file_record

    artifact = load_artifact(args.calibration)
    output = args.output.resolve()
    config = load(args.config)
    config["max_batch"] = args.batch
    with GPULease(args.gpu):
        select_gpu(args.gpu)
        engine = Engine(config)
        try:
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                model = engine.student.model if args.component == "cfm" else engine.tts.bigvgan
                bindings = list(iter_roles(engine, artifact["scheme"], (args.component,)))
                specs, mapping = acoustic_specs_from_artifact(model, bindings, artifact, args.component)
                if args.deconv_as_conv:
                    modules = dict(model.named_modules())
                    specs = {path: dict(spec) for path, spec in specs.items()}
                    for path, spec in specs.items():
                        if isinstance(modules[path], torch.nn.ConvTranspose1d):
                            spec["conv_transpose_rewrite"] = "zero_insert_conv"
                provenance = capture_provenance(args.component, config, args.config, model=engine)
                manifest = prepare_acoustic_model(model, specs, fir_polyphase=args.fir_polyphase,
                                                  cfm_sdpa_bf16=args.cfm_sdpa_bf16)
                graph_equivalence = None
                if args.cfm_wavenet_tail:
                    if args.validation_capture is None:
                        raise ValueError('Tail WaveNet export requires a real validation capture')
                    frozen = torch.load(args.validation_capture, map_location='cpu', weights_only=True)
                    if (frozen['batch'] != args.batch or frozen['scheme'] != artifact['scheme']
                            or frozen['calibration']['sha256'] != file_record(args.calibration, 'calibration_artifact')['sha256']):
                        raise ValueError('Tail validation capture differs from the export recipe/batch')
                    cpu_mask = frozen['waves'][0]['cfm_inputs'][5]
                    expected_mask = (torch.arange(310)[None, None] < 258).expand(args.batch, 1, -1)
                    if not torch.equal(cpu_mask, expected_mask):
                        raise ValueError('Tail pruning requires the fixed P258 solver mask')
                    actual_inputs = tuple(v.cuda() for v in frozen['waves'][0]['cfm_inputs'])
                    full_result = FourStepCFM(model)(*actual_inputs).detach().cpu().double()
                    tail_check = TailWaveNetDiT(model)
                    tail_result = FourStepCFM(tail_check)(*actual_inputs).detach().cpu().double()
                    diff = tail_result - full_result
                    graph_equivalence = dict(reference='same_recipe_full_solver_real_frozen_inputs',
                        capture=file_record(args.validation_capture, 'validation_capture'),
                        finite=bool(torch.isfinite(tail_result).all()),
                        relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(full_result)),
                        max_abs=float(diff.abs().max()), halo=tail_check.radius,
                        required_masked_prompt_frames=258)
                    if not graph_equivalence['finite']:
                        raise RuntimeError('Tail graph has nonfinite output')
                    print(json.dumps(dict(event='tail_graph_equivalence', **graph_equivalence)), flush=True)
                    del actual_inputs, full_result, tail_result, diff
                for role in manifest["roles"]:
                    role["canonical_path"] = mapping.get(role["path"])
                b = args.batch
                sample = next(model.parameters(), None)
                if sample is None:
                    sample = next(model.buffers())
                device = sample.device
                if args.component == "cfm":
                    x = torch.zeros(b, 80, 310, device=device)
                    prompt = torch.zeros_like(x)
                    lengths = torch.full((b,), 310, device=device, dtype=torch.int64)
                    style = torch.zeros(b, 192, device=device)
                    mu = torch.zeros(b, 310, 512, device=device)
                    if args.kind == "full_solver":
                        mask = (torch.arange(310, device=device)[None, None] < 258).expand(b, 1, -1).contiguous()
                        if args.cfm_wavenet_tail:
                            model = TailWaveNetDiT(model, prompt_frames=258)
                        model = FourStepCFM(model)
                        inputs = (x, prompt, lengths, style, mu, mask)
                        input_names = ["x", "prompt", "lengths", "style", "mu", "mask"]
                        output_names = ["output"]
                    else:
                        times = torch.tensor([[0.0, 0.25]], device=device).expand(b, 2).contiguous()
                        inputs = (x, prompt, lengths, times, style, mu)
                        input_names = ["x", "prompt", "lengths", "times", "style", "mu"]
                        output_names = ["velocity"]
                else:
                    inputs = (torch.zeros(b, 80, 52, device=device),)
                    input_names, output_names = ["mel"], ["pcm"]
                graph = export_standard_onnx(model, inputs, output,
                                              input_names=input_names, output_names=output_names)
                torch.cuda.synchronize()
                model_artifact = capture_onnx_artifact(output)
                result = {
                    "batch": b, "frames": 310 if args.component == "cfm" else 52,
                    "component": args.component, "kind": args.kind if args.component == "cfm" else "vocoder",
                    "onnx": str(output), "bytes": output.stat().st_size,
                    "onnx_sha256": model_artifact["sha256"], "onnx_artifact": model_artifact,
                    "provenance": provenance, "provenance_status": provenance["status"],
                    "exporter": file_record(__file__, "onnx_exporter"),
                    "export_settings": {"opset": 20, "dynamo": False, "external_data": True,
                                        "constant_folding": True, "batch": b,
                                        "scheme": artifact["scheme"],
                                        "fir_polyphase": args.fir_polyphase,
                                        "cfm_sdpa_bf16": args.cfm_sdpa_bf16,
                                        "cfm_wavenet_tail": args.cfm_wavenet_tail,
                                        "precision": artifact["scheme"] + "_static_qdq_fp32_interfaces",
                                        "custom_opsets": {}, "intervals": [list(v) for v in INTERVALS]},
                    "quantization_recipe": {
                        "scheme": artifact["scheme"], "activation_scaling": "offline_static_per_tensor",
                        "calibration": file_record(args.calibration, "calibration_artifact"),
                        "alpha": artifact.get("alpha"), "role_manifest": manifest,
                        "role_specs_sha256": hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(),
                    },
                    "graph": graph, "plugins": [], "plugin_nodes": 0,
                    "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                               for name, value in zip(input_names, inputs)],
                    "validation": {"output_finite": True, "numerical_audit": "deferred_to_final_audit"},
                }
                if args.component == "cfm":
                    result["prompt_frames"] = 258
                if graph_equivalence is not None:
                    result['validation']['graph_equivalence'] = graph_equivalence
                output.with_suffix(".export.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"onnx": str(output), "scheme": artifact["scheme"],
                                  "component": args.component, "batch": b,
                                  "nodes": graph["nodes"], "plugins": []}), flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
