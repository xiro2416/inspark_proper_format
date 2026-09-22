#!/usr/bin/env python3
"""Parse a fixed-shape CFM ONNX graph and build a native TRT 11.3 engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="artifacts/trt113_cfm/cfm_solver_b8.onnx")
    parser.add_argument("--engine", default="artifacts/trt113_cfm/cfm_solver_b8.engine")
    parser.add_argument("--optimization-level", type=int, default=5, choices=range(6))
    args = parser.parse_args()

    import tensorrt as trt

    onnx_path = Path(args.onnx).resolve()
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
    config.builder_optimization_level = args.optimization_level
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT CFM build failed")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append({
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": list(engine.get_tensor_shape(name)),
        })
    report = {
        "format": 1,
        "backend": "TensorRT 11.3 native full two-step CFM Solver",
        "batch": 8,
        "frames": 310,
        "prompt_frames": 258,
        "engine": str(engine_path),
        "bytes": engine_path.stat().st_size,
        "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "build_seconds": time.time() - started,
        "trt": trt.__version__,
        "optimization_level": args.optimization_level,
        "workspace_bytes": 8 << 30,
        "precision": "BF16 weights/linears with FP32 interfaces and two-step accumulation",
        "tensors": tensors,
    }
    engine_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
