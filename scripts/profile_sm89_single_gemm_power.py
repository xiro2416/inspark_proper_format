#!/usr/bin/env python3
"""Sustain each distinct real B8 Draft/Target GEMM and sample board power."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from profile_sm89_power import NvmlSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--baseline-seconds", type=float, default=4.0)
    parser.add_argument("--m-multiplier", type=int, default=1)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--deployment", default="configs/sm89_bf16_triton_device_control.json")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.m_multiplier < 1:
        raise ValueError("m-multiplier must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn.functional as F
    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")

    config = load_config(args.config)
    config["max_batch"] = 8
    deployment = load_deployment(args.deployment)

    def params(module):
        candidates = [child for child in module.modules()
                      if hasattr(child, "weight") and isinstance(child.weight, torch.Tensor)
                      and child.weight.ndim == 2]
        if not candidates:
            raise RuntimeError(f"No matrix weight in {module}")
        layer = candidates[-1]
        return layer.weight, getattr(layer, "bias", None), type(layer).__name__

    with GPULease(args.gpu):
        engine = Engine(config)
        sampler = None
        try:
            engine.prepare_reference("reference", args.ref_audio)
            manifest = engine.prepare_deployment(deployment)
            torch.cuda.synchronize()
            draft = engine.rt.engine.draft
            draft_layer = draft.layers[0]
            target_model = engine.rt.engine.target.model
            target_block = target_model.transformer.h[0]
            proposal = engine.rt.proposal

            cases = []

            def add(component, role, module, m, multiplicity, note=""):
                weight, bias, module_type = params(module)
                n, k = weight.shape
                cases.append({"component": component, "role": role, "original_m": m,
                              "m": m * args.m_multiplier, "n": n, "k": k,
                              "multiplicity_per_round": multiplicity, "weight": weight,
                              "bias": bias, "module_type": module_type, "note": note})

            # Draft backbone: 8 requests x 7 proposed positions = M56.
            add("draft", "attention.q_proj", draft_layer.q_proj, 56, 3)
            add("draft", "attention.k_proj", draft_layer.k_proj, 56, 3)
            add("draft", "attention.v_proj", draft_layer.v_proj, 56, 3)
            add("draft", "attention.o_proj", draft_layer.o_proj, 56, 3)
            add("draft", "mlp.expand", draft_layer.mlp[0], 56, 3)
            add("draft", "mlp.contract", draft_layer.mlp[2], 56, 3)
            add("draft", "vocab_head", draft.lm_head, 56, 1, "FP32 interface head")

            # Target-selected hidden -> Draft context, then three layers of K/V projection.
            add("draft_context", "selected_hidden_projection", draft.context_projection, 64, 1,
                "5x1280 selected Target features; FP32 interface projection")
            add("draft_context", "attention.k_proj", draft_layer.k_proj, 64, 3)
            add("draft_context", "attention.v_proj", draft_layer.v_proj, 64, 3)

            # Proposal graph GEMMs. MatrixLinear stores real [N,K] BF16 weights.
            add("draft_proposal", "hidden_linear", proposal.hidden_linear, 56, 1)
            add("draft_proposal", "state_linear", proposal.state_linear, 8, 7)
            add("draft_proposal", "output_vocab", proposal.output_linear, 8, 7)

            # Target: 8 requests x 8 verification positions = M64, repeated across 24 blocks.
            add("target", "attention.qkv", target_block.attn.c_attn, 64, 24)
            add("target", "attention.o_proj", target_block.attn.c_proj, 64, 24)
            add("target", "mlp.expand", target_block.mlp.c_fc, 64, 24)
            add("target", "mlp.contract", target_block.mlp.c_proj, 64, 24)
            add("target", "vocab_head", target_model.lm_head, 64, 1, "FP32 interface head")

            sampler = NvmlSampler(args.gpu)
            sampler.start()
            time.sleep(args.baseline_seconds)
            now = time.perf_counter()
            idle_rows = sampler.window(now - args.baseline_seconds, now)
            idle_w = statistics.fmean(row[1] for row in idle_rows)
            results = []
            for case in cases:
                weight = case.pop("weight")
                bias = case.pop("bias")
                x = torch.randn(case["m"], case["k"], device="cuda", dtype=weight.dtype)

                def gemm():
                    return F.linear(x, weight, bias)

                for _ in range(20):
                    output = gemm()
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(100):
                    output = gemm()
                end_event.record(); end_event.synchronize()
                estimate_ms = start_event.elapsed_time(end_event) / 100.0
                inner = max(1, min(4096, int(50.0 / max(estimate_ms, 0.001))))
                settling = time.perf_counter()
                while time.perf_counter() - settling < args.settle_seconds:
                    for _ in range(inner):
                        output = gemm()
                    torch.cuda.synchronize()
                started = time.perf_counter()
                calls = 0
                while time.perf_counter() - started < args.seconds:
                    for _ in range(inner):
                        output = gemm()
                    torch.cuda.synchronize()
                    calls += inner
                completed = time.perf_counter()
                rows = sampler.window(started, completed)
                summary = NvmlSampler.summarise(rows, idle_w, completed - started, calls, 0.9)
                calls_per_second = calls / (completed - started)
                achieved_tflops = 2.0 * case["m"] * case["n"] * case["k"] * calls_per_second / 1e12
                peak_tflops = 82.6 if weight.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32 else 165.2
                summary.update({
                    "calls_per_second": calls_per_second,
                    "latency_us": 1e6 / calls_per_second,
                    "tflops": achieved_tflops,
                    "reference_peak_tflops": peak_tflops,
                    "tensor_core_peak_fraction": achieved_tflops / peak_tflops,
                    "gross_j_per_call": summary["power_w_mean"] / calls_per_second,
                    "dtype": str(weight.dtype).removeprefix("torch."),
                    "bias": bias is not None,
                    "inner_calls_per_sync": inner,
                })
                results.append({**case, **summary})
                del x, output
                torch.cuda.empty_cache()

            sampler.stop(); sampler = None
            report = {
                "gpu": args.gpu, "batch": 8, "deployment": manifest["requested"]["status"],
                "seconds_per_gemm": args.seconds, "settle_seconds_per_gemm": args.settle_seconds,
                "m_multiplier": args.m_multiplier,
                "idle_power_w": idle_w,
                "idle_samples": len(idle_rows),
                "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "reference_audio": args.ref_audio,
                "results": results,
                "attention_note": "Slot Draft/Target score/value attention is a fused Triton kernel, not a standalone GEMM.",
            }
            Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(json.dumps(report, indent=2, ensure_ascii=False))
        finally:
            if sampler is not None:
                sampler.stop()
            engine.close()


if __name__ == "__main__":
    main()
