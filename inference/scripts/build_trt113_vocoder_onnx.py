#!/usr/bin/env python3
"""Build the static plugin-enabled BigVGAN engine with TensorRT 11.3."""
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
    parser.add_argument("--onnx", default='../artifacts/trt113_vocoder/vocoder_b8.onnx')
    parser.add_argument("--engine", default='../artifacts/trt113_vocoder/vocoder_b8.engine')
    parser.add_argument("--plan", help="Write a loadable plan with an engine path relative to this file")
    parser.add_argument("--batch", type=int, help="Assert the batch inferred from TensorRT IO")
    parser.add_argument("--frames", type=int, help="Assert the frame count inferred from TensorRT IO")
    parser.add_argument("--optimization-level", type=int, default=5, choices=range(6))
    parser.add_argument("--strongly-typed", action="store_true",
                        help="Preserve ONNX dtypes; export uses a BF16 deconvolution plugin")
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

    # Preload NumPy/Torch from the project environment before the isolated TRT
    # importer temporarily prepends its site-packages directory.
    import numpy as np
    import torch
    from acc_infer_clear.ops.tensorrt.native113 import _import_trt113

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU before building TensorRT engines")
    trt = _import_trt113()
    if not trt.__version__.startswith("11.3."):
        raise RuntimeError(f"Expected TensorRT 11.3, got {trt.__version__}")
    major, minor = torch.cuda.get_device_capability(0)
    from acc_infer_clear.ops.tensorrt.vocoder_plugin import register

    register()
    engine_path = Path(args.engine).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger)
    flags = ((1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
             if args.strongly_typed else 0)
    network = builder.create_network(flags); parser_ = trt.OnnxParser(network, logger)
    if not parser_.parse_from_file(str(onnx_path)):
        errors = [str(parser_.get_error(i)) for i in range(parser_.num_errors)]
        raise RuntimeError("TensorRT Vocoder ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config(); config.builder_optimization_level = args.optimization_level
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    started = time.time(); serialized = builder.build_serialized_network(network, config)
    if serialized is None: raise RuntimeError("TensorRT Vocoder build failed")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger); engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Built Vocoder engine could not be deserialized")
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append({"name": name, "mode": str(engine.get_tensor_mode(name)),
                        "dtype": str(engine.get_tensor_dtype(name)),
                        "shape": list(engine.get_tensor_shape(name))})
    by_name = {entry["name"]: entry for entry in tensors}
    shape = by_name.get("mel", {}).get("shape", [])
    if len(shape) != 3 or shape[1] != 80 or min(shape) <= 0:
        raise ValueError(f"Expected static Vocoder mel[B,80,F], got {shape}")
    batch, _, frames = shape
    expected_shapes = {"mel": [batch, 80, frames], "pcm": [batch, 1, frames * 256]}
    if set(by_name) != set(expected_shapes):
        raise ValueError(f"Unexpected Vocoder IO: {sorted(by_name)}")
    for name, expected in expected_shapes.items():
        mode = trt.TensorIOMode.OUTPUT if name == "pcm" else trt.TensorIOMode.INPUT
        if (by_name[name]["shape"] != expected or engine.get_tensor_dtype(name) != trt.float32
                or engine.get_tensor_mode(name) != mode):
            raise ValueError(f"Vocoder IO contract mismatch for {name}: {by_name[name]}")
    if args.batch is not None and args.batch != batch:
        raise ValueError(f"Requested B{args.batch}, built B{batch}")
    if args.frames is not None and args.frames != frames:
        raise ValueError(f"Requested F{args.frames}, built F{frames}")
    if export and (export.get("batch") != batch or export.get("frames") != frames):
        raise ValueError("Vocoder export metadata disagrees with the TensorRT IO contract")
    report = {
        "format": 1, "backend": "TensorRT 11.3 native BigVGAN with alias-free plugin",
        "batch": batch, "frames": frames, "engine": str(engine_path),
        "shape_source": "TensorRT engine IO", "onnx": str(onnx_path),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "provenance": provenance, "provenance_status": provenance["status"],
        "build_source": build_source,
        "physical_gpu": args.physical_gpu, "external_memory_mib": args.external_memory_mib,
        "bytes": engine_path.stat().st_size,
        "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "build_seconds": time.time() - started, "trt": trt.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(0), "sm": major * 10 + minor,
        "optimization_level": args.optimization_level, "workspace_bytes": 8 << 30,
        "strongly_typed": args.strongly_typed,
        "tf32": False,
        "regular_conv": "plugin" if export.get("conv_plugins") else "native" if export else "unknown",
        "plugins": (["inspark::alias_free"] if export.get("plugin_nodes") else [])
                   + (["inspark::deconv1d"] if export.get("deconv_plugins") else [])
                   + (["inspark::conv1d"] if export.get("conv_plugins") else []),
        "export_rewrite_validation": export.get("export_rewrite_validation"),
        "precision": "BF16 learned convolutions; FP32 alias-free plugin and interfaces",
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
