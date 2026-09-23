#!/usr/bin/env python3
"""Export the fixed two-step CFM solver for native TensorRT 11.3 parsing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=310)
    parser.add_argument("--prompt-frames", type=int, default=258)
    parser.add_argument("--config", default="configs/common/runtime.yaml")
    parser.add_argument("--output", default='artifacts/trt113_cfm/cfm_solver_b8.onnx')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from torch import nn
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.engine import Engine
    from trt113_provenance import capture_onnx_artifact, capture_provenance, file_record

    if args.frames != args.prompt_frames + 52:
        raise ValueError("First-head CFM export requires prompt_frames + 52")

    class FullCFMSolver(nn.Module):
        def __init__(self, solver):
            super().__init__()
            self.model = solver.model
            self.register_buffer("times", torch.cat(solver.times, dim=0))

        def forward(self, x, prompt, lengths, style, mu, mask):
            x = x.float().masked_fill(mask, 0)
            for index in range(2):
                times = self.times[index:index + 1].expand(x.shape[0], -1)
                velocity = self.model(x, prompt, lengths, times, style, mu)
                x = (x + 0.5 * velocity.float()).masked_fill(mask, 0)
            return x

    config = load(args.config)
    config["max_batch"] = args.batch
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_precision("bf16", ["target", "draft", "cfm", "vocoder"], True)
            provenance = capture_provenance("cfm", config, args.config, model=engine)
            provenance["source"]["snapshot"] = "local source files before ONNX export"
            model = FullCFMSolver(engine.student).eval().requires_grad_(False)
            device = next(model.parameters()).device
            b, frames, prompt_frames = args.batch, args.frames, args.prompt_frames
            x = torch.zeros(b, 80, frames, device=device, dtype=torch.float32)
            prompt = torch.zeros_like(x)
            lengths = torch.full((b,), frames, device=device, dtype=torch.long)
            style = torch.zeros(b, 192, device=device, dtype=torch.float32)
            mu = torch.zeros(b, frames, 512, device=device, dtype=torch.float32)
            mask = (torch.arange(frames, device=device)[None, None] < prompt_frames).expand(b, 1, -1).clone()
            inputs = (x, prompt, lengths, style, mu, mask)
            with torch.inference_mode():
                expected = model(*inputs)
                torch.cuda.synchronize()
            torch.onnx.export(
                model,
                inputs,
                str(output),
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=["x", "prompt", "lengths", "style", "mu", "mask"],
                output_names=["output"],
                dynamo=False,
                external_data=True,
            )
            onnx_artifact = capture_onnx_artifact(output)
            report = {
                "batch": b,
                "frames": frames,
                "prompt_frames": prompt_frames,
                "onnx": str(output),
                "bytes": output.stat().st_size,
                "onnx_sha256": onnx_artifact["sha256"],
                "onnx_artifact": onnx_artifact,
                "provenance": provenance,
                "provenance_status": provenance["status"],
                "exporter": file_record(__file__, "onnx_exporter"),
                "export_settings": {"opset": 20, "dynamo": False, "external_data": True,
                                    "constant_folding": True, "precision": "bf16_fp32_interfaces",
                                    "batch": b, "frames": frames, "prompt_frames": prompt_frames,
                                    "intervals": [[0, 0.5], [0.5, 1]], "cfg": 0},
                "inputs": [
                    {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in zip(("x", "prompt", "lengths", "style", "mu", "mask"), inputs)
                ],
                "output": {"shape": list(expected.shape), "dtype": str(expected.dtype)},
                "opset": 20,
            }
            output.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            engine.close()


if __name__ == "__main__":
    main()
