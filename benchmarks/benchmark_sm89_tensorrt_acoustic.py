#!/usr/bin/env python3
"""Isolated CUDA-event benchmark for acoustic TensorRT candidates."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn


class EstimatorSolver(nn.Module):
    def __init__(self, estimator, times):
        super().__init__()
        self.estimator = estimator
        self.register_buffer("times", torch.cat(times, dim=0))

    def forward(self, x, prompt, lengths, style, mu, mask):
        x = x.float().masked_fill(mask, 0)
        for index in range(2):
            times = self.times[index : index + 1].expand(x.shape[0], -1)
            velocity = self.estimator(x, prompt, lengths, times, style, mu)
            x = (x + 0.5 * velocity.float()).masked_fill(mask, 0)
        return x


def bench(name, fn, args, warmups, iterations):
    with torch.inference_mode():
        for _ in range(warmups):
            output = fn(*args)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = fn(*args)
        end.record()
        end.synchronize()
        direct_ms = start.elapsed_time(end) / iterations

        static = tuple(value.clone() for value in args)
        for _ in range(3):
            output = fn(*static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(*static)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        graph_ms = start.elapsed_time(end) / iterations
    return {"name": name, "direct_ms": direct_ms, "cuda_graph_ms": graph_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/common/runtime.yaml")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--batch", type=int, required=True, choices=(1, 4, 8, 16))
    parser.add_argument("--v2-root", default='artifacts/tensorrt_sm89_bf16_v2')
    parser.add_argument("--v3-root", default='artifacts/tensorrt_sm89_bf16_v3')
    parser.add_argument("--v3-full-root", default='artifacts/tensorrt_sm89_bf16_v3_full')
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU")

    import torch_tensorrt
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.ops.triton.alias_free import install as install_alias
    from inspark_infer.ops.triton.stage2_fusions import install as install_cfm

    config = load(args.config)
    config["max_batch"] = args.batch
    engine = Engine(config)
    rows = []
    try:
        engine.prepare_reference("reference", args.ref_audio)
        engine.prepare_precision("bf16", ["target", "draft", "cfm", "vocoder"], True)
        voice = engine.model.bank.get("reference")["values"]
        prompt_frames = voice["voice.cache_mel"].shape[-1]
        frames = prompt_frames + 52
        mu = torch.cat(
            (
                voice["voice.cache_s2mel_prompt"],
                voice["voice.cache_s2mel_prompt"].new_zeros(
                    1, 52, voice["voice.cache_s2mel_prompt"].shape[-1]
                ),
            ), 1,
        ).repeat(args.batch, 1, 1)
        x = mu.new_zeros(args.batch, 80, frames)
        prompt = x.clone()
        prompt[:, :, :prompt_frames] = voice["voice.cache_mel"]
        lengths = torch.full((args.batch,), frames, device=x.device, dtype=torch.long)
        style = voice["voice.cache_s2mel_style"].repeat(args.batch, 1)
        mask = (torch.arange(frames, device=x.device)[None, None] < prompt_frames).expand(
            args.batch, 1, -1
        ).clone()
        cfm_args = (x, prompt, lengths, style, mu, mask)
        vocoder_args = (x[:, :, :52].clone(),)

        eager_solver = EstimatorSolver(engine.student.model, engine.student.times).eval()
        rows.append(bench("cfm_eager_bf16", eager_solver, cfm_args, args.warmups, args.iterations))

        v2_cfm_path = Path(args.v2_root) / f"cfm_b{args.batch}.ts"
        if v2_cfm_path.exists():
            v2_estimator = torch_tensorrt.load(str(v2_cfm_path)).eval().cuda()
            v2_solver = EstimatorSolver(v2_estimator, engine.student.times).eval()
            rows.append(bench("cfm_v2_estimator_twice", v2_solver, cfm_args, args.warmups, args.iterations))

        for name, root in (("cfm_v3_solver_none", args.v3_root), ("cfm_v3_solver_full", args.v3_full_root)):
            path = Path(root) / f"cfm_solver_b{args.batch}.ts"
            if path.exists():
                compiled = torch_tensorrt.load(str(path)).eval().cuda()
                rows.append(bench(name, compiled, cfm_args, args.warmups, args.iterations))

        install_cfm(engine.student.model, ("norm", "rope"))
        current_solver = EstimatorSolver(engine.student.model, engine.student.times).eval()
        rows.append(bench("cfm_current_fusions", current_solver, cfm_args, args.warmups, args.iterations))

        rows.append(bench("vocoder_eager_bf16", engine.tts.bigvgan, vocoder_args, args.warmups, args.iterations))
        v2_vocoder_path = Path(args.v2_root) / f"vocoder_b{args.batch}.ts"
        if v2_vocoder_path.exists():
            v2_vocoder = torch_tensorrt.load(str(v2_vocoder_path)).eval().cuda()
            rows.append(bench("vocoder_v2", v2_vocoder, vocoder_args, args.warmups, args.iterations))
        v3_full_vocoder_path = Path(args.v3_full_root) / f"vocoder_b{args.batch}.ts"
        if v3_full_vocoder_path.exists():
            v3_full_vocoder = torch_tensorrt.load(str(v3_full_vocoder_path)).eval().cuda()
            rows.append(bench("vocoder_v3_full_tiling", v3_full_vocoder, vocoder_args,
                              args.warmups, args.iterations))
        install_alias(engine.tts.bigvgan)
        rows.append(bench("vocoder_current_alias", engine.tts.bigvgan, vocoder_args, args.warmups, args.iterations))

        result = {
            "gpu": torch.cuda.get_device_name(),
            "batch": args.batch,
            "frames": frames,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "rows": rows,
        }
        encoded = json.dumps(result, indent=2, ensure_ascii=False)
        print(encoded)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded + "\n")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
