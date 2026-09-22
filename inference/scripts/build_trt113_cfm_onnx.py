#!/usr/bin/env python3
"""Parse a fixed-shape CFM ONNX graph and build a native TRT 11.3 engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="Physical GPU; otherwise require one numeric CUDA_VISIBLE_DEVICES entry")
    parser.add_argument("--onnx", default='../artifacts/trt113_cfm/cfm_solver_b8.onnx')
    parser.add_argument("--engine", default='../artifacts/trt113_cfm/cfm_solver_b8.engine')
    parser.add_argument("--plan", help="Write a loadable plan with an engine path relative to this file")
    parser.add_argument("--batch", type=int, help="Assert the batch inferred from TensorRT IO")
    parser.add_argument("--frames", type=int, help="Assert the frame count inferred from TensorRT IO")
    parser.add_argument("--optimization-level", type=int, default=5, choices=range(6))
    args = parser.parse_args()
    from acc_infer_clear.runtime.device import GPULease, select_gpu
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu = args.gpu if args.gpu is not None else int(visible) if visible.isdecimal() else None
    if gpu is None:
        parser.error("Select exactly one physical GPU with --gpu or CUDA_VISIBLE_DEVICES")
    with GPULease(gpu) as lease:
        select_gpu(gpu)
        args.physical_gpu = gpu
        args.external_memory_mib = lease.initial_memory_mib
        build(args)


def build(args):
    from trt113_provenance import load_onnx_export, source_identity

    onnx_path = Path(args.onnx).resolve()
    export, provenance = load_onnx_export(onnx_path)
    build_source = source_identity()

    # Load the project environment before temporarily exposing the isolated TRT
    # package. Its bundled NumPy must not replace the project's numeric stack.
    import numpy as np
    import torch
    from acc_infer_clear.ops.tensorrt.native113 import _import_trt113

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU before building TensorRT engines")
    trt = _import_trt113()

    if not trt.__version__.startswith("11.3."):
        raise RuntimeError(f"Expected TensorRT 11.3, got {trt.__version__}")
    major, minor = torch.cuda.get_device_capability(0)

    engine_path = Path(args.engine).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser_ = trt.OnnxParser(network, logger)
    if not parser_.parse_from_file(str(onnx_path)):
        errors = [str(parser_.get_error(i)) for i in range(parser_.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = args.optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT CFM build failed")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Built CFM engine could not be deserialized")
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append({
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": list(engine.get_tensor_shape(name)),
        })
    by_name = {entry["name"]: entry for entry in tensors}
    shape = by_name.get("x", {}).get("shape", [])
    if len(shape) != 3 or shape[1] != 80 or min(shape) <= 0:
        raise ValueError(f"Expected static CFM x[B,80,F], got {shape}")
    batch, _, frames = shape
    expected_shapes = {
        "x": [batch, 80, frames], "prompt": [batch, 80, frames],
        "lengths": [batch], "style": [batch, 192],
        "mu": [batch, frames, 512], "mask": [batch, 1, frames],
        "output": [batch, 80, frames],
    }
    if set(by_name) != set(expected_shapes):
        raise ValueError(f"Unexpected CFM IO: {sorted(by_name)}")
    for name, expected in expected_shapes.items():
        dtype = trt.int64 if name == "lengths" else trt.bool if name == "mask" else trt.float32
        mode = trt.TensorIOMode.OUTPUT if name == "output" else trt.TensorIOMode.INPUT
        if (by_name[name]["shape"] != expected or engine.get_tensor_dtype(name) != dtype
                or engine.get_tensor_mode(name) != mode):
            raise ValueError(f"CFM IO contract mismatch for {name}: {by_name[name]}")
    if args.batch is not None and args.batch != batch:
        raise ValueError(f"Requested B{args.batch}, built B{batch}")
    if args.frames is not None and args.frames != frames:
        raise ValueError(f"Requested F{args.frames}, built F{frames}")
    prompt_frames = frames - 52
    if prompt_frames <= 0:
        raise ValueError("First-head CFM requires positive prompt frames plus 52 generated frames")
    if export and any(export.get(key) != value for key, value in
                      (("batch", batch), ("frames", frames), ("prompt_frames", prompt_frames))):
        raise ValueError("CFM export metadata disagrees with the TensorRT IO contract")
    report = {
        "format": 1,
        "backend": "TensorRT 11.3 native full two-step CFM Solver",
        "batch": batch,
        "frames": frames,
        "prompt_frames": prompt_frames,
        "shape_source": "TensorRT engine IO",
        "prompt_frames_source": "export metadata and first-head F-52" if export else "first-head F-52 contract",
        "onnx": str(onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "provenance": provenance,
        "provenance_status": provenance["status"],
        "build_source": build_source,
        "physical_gpu": args.physical_gpu, "external_memory_mib": args.external_memory_mib,
        "engine": str(engine_path),
        "bytes": engine_path.stat().st_size,
        "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "build_seconds": time.time() - started,
        "trt": trt.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "sm": major * 10 + minor,
        "optimization_level": args.optimization_level,
        "workspace_bytes": 8 << 30,
        "strongly_typed": True,
        "tf32": False,
        "precision": "BF16 weights/linears with FP32 interfaces and two-step accumulation",
        "tensors": tensors,
    }
    engine_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    if args.plan:
        plan_path = Path(args.plan).resolve()
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = dict(report, engine=os.path.relpath(engine_path, plan_path.parent))
        plan_path.write_text(json.dumps(plan, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
