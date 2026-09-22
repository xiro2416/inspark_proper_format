#!/usr/bin/env python3
"""Build strict static TensorRT candidates for complete acoustic stages.

Unlike V2, the CFM artifact contains the complete fixed two-interval solver,
including recurrent state updates and prompt masking.  Artifacts are built one
batch at a time so batch-specific strategies can be selected independently.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn


class FullCFMSolver(nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.model = solver.model
        self.register_buffer("times", torch.cat(solver.times, dim=0))

    def forward(self, x, prompt, lengths, style, mu, mask):
        x = x.float().masked_fill(mask, 0)
        # Fixed deployment student contract: exactly two integration intervals.
        for index in range(2):
            times = self.times[index : index + 1].expand(x.shape[0], -1)
            velocity = self.model(x, prompt, lengths, times, style, mu)
            x = (x + 0.5 * velocity.float()).masked_fill(mask, 0)
        return x


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--batch", type=int, required=True, choices=(1, 4, 8, 16))
    parser.add_argument("--components", default="cfm,vocoder")
    parser.add_argument("--output", default="artifacts/tensorrt_sm89_bf16_v3")
    parser.add_argument("--tiling", choices=("none", "fast", "moderate", "full"), default="none")
    args = parser.parse_args()
    components = tuple(part.strip() for part in args.components.split(",") if part.strip())
    if not components or not set(components) <= {"cfm", "vocoder"}:
        raise ValueError("components must be cfm and/or vocoder")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU")

    import tensorrt
    import torch_tensorrt
    from acc_infer_clear.config import load
    from acc_infer_clear.streaming.engine import Engine

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    timing_cache = output / "timing.cache"
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        plan = {
            "format_version": 3,
            "backend": "Torch-TensorRT Dynamo",
            "precision": "bf16_fp32_interfaces",
            "sm": torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1],
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "tensorrt_version": tensorrt.__version__,
            "torch_tensorrt_version": torch_tensorrt.__version__,
            "strict_full_compilation": True,
            "tactic_search": {
                "static_shapes": True,
                "optimization_level": 5,
                "workspace_bytes": 8 << 30,
                "num_avg_timing_iters": 5,
                "timing_cache": timing_cache.name,
                "tiling_optimization": args.tiling,
                "l2_limit_for_tiling": torch.cuda.get_device_properties(0).L2_cache_size,
            },
            "engines": [],
        }

    settings = {
        "ir": "dynamo",
        "enabled_precisions": {torch.float32, torch.bfloat16},
        "require_full_compilation": True,
        "min_block_size": 1,
        "optimization_level": 5,
        "workspace_size": 8 << 30,
        "num_avg_timing_iters": 5,
        "timing_cache_path": str(timing_cache),
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "use_fast_partitioner": False,
        "tiling_optimization_level": args.tiling,
        "l2_limit_for_tiling": torch.cuda.get_device_properties(0).L2_cache_size,
    }

    config = load(args.config)
    config["max_batch"] = args.batch
    engine = Engine(config)

    def compile_save(component, module, inputs, frames):
        artifact = output / f"{component}_b{args.batch}.ts"
        if artifact.exists():
            print(json.dumps({"component": component, "batch": args.batch, "status": "exists"}), flush=True)
            return
        started = time.perf_counter()
        compiled = torch_tensorrt.compile(module, inputs=list(inputs), **settings)
        torch.cuda.synchronize()
        with torch.inference_mode():
            expected = module(*inputs)
            actual = compiled(*inputs)
            torch.cuda.synchronize()
        validation = {
            "max_abs": (expected - actual).abs().max().item(),
            "mean_abs": (expected - actual).abs().mean().item(),
        }
        torch_tensorrt.save(compiled, str(artifact), output_format="torchscript", inputs=list(inputs))
        entry = {
            "component": component,
            "batch": args.batch,
            "frames": frames,
            "artifact": artifact.name,
            "sha256": sha256(artifact),
            "compile_seconds": time.perf_counter() - started,
            "validation": validation,
        }
        plan["engines"] = [
            old for old in plan["engines"]
            if not (old["component"] == component and int(old["batch"]) == args.batch)
        ] + [entry]
        plan["batches"] = sorted({int(item["batch"]) for item in plan["engines"]})
        plan["max_batch"] = max(plan["batches"])
        plan["tactic_search"].update({
            "tiling_optimization": args.tiling,
            "l2_limit_for_tiling": torch.cuda.get_device_properties(0).L2_cache_size,
        })
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        del compiled, expected, actual
        torch.cuda.empty_cache()

    try:
        engine.prepare_reference("reference", args.ref_audio)
        engine.prepare_precision("bf16", ["target", "draft", "cfm", "vocoder"], True)
        engine.student.model.requires_grad_(False)
        engine.tts.bigvgan.requires_grad_(False)
        voice = engine.model.bank.get("reference")["values"]
        prompt_frames = voice["voice.cache_mel"].shape[-1]
        frames = prompt_frames + 52
        mu = torch.cat(
            (
                voice["voice.cache_s2mel_prompt"],
                voice["voice.cache_s2mel_prompt"].new_zeros(
                    1, 52, voice["voice.cache_s2mel_prompt"].shape[-1]
                ),
            ),
            1,
        ).repeat(args.batch, 1, 1)
        x = mu.new_zeros(args.batch, 80, frames)
        prompt = x.clone()
        prompt[:, :, :prompt_frames] = voice["voice.cache_mel"]
        lengths = torch.full((args.batch,), frames, device=x.device, dtype=torch.long)
        style = voice["voice.cache_s2mel_style"].repeat(args.batch, 1)
        mask = (torch.arange(frames, device=x.device)[None, None] < prompt_frames).expand(
            args.batch, 1, -1
        ).clone()
        if "cfm" in components:
            compile_save(
                "cfm_solver",
                FullCFMSolver(engine.student).eval(),
                (x, prompt, lengths, style, mu, mask),
                frames,
            )
        if "vocoder" in components:
            compile_save("vocoder", engine.tts.bigvgan, (x[:, :, :52].clone(),), 52)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
