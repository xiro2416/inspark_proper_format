#!/usr/bin/env python3
"""Benchmark the TensorRT 11.3 native attention primitive on one visible GPU."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def check(result):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(result)
    return result[1] if len(result) > 1 else None


def dtype_bytes(dtype):
    return {
        trt.DataType.BF16: 2,
        trt.DataType.HALF: 2,
        trt.DataType.FLOAT: 4,
        trt.DataType.INT32: 4,
    }[dtype]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default='../artifacts/trt113_attention')
    parser.add_argument("--out", default="outputs/trt113_attention_benchmark.json")
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    check(cudart.cudaSetDevice(0))
    stream = check(cudart.cudaStreamCreate())
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    results = []
    for batch in (1, 4, 8, 16):
        for decomposable in (False, True):
            path = Path(args.artifact_dir) / f"attention_d{int(decomposable)}_b{batch}.engine"
            engine = runtime.deserialize_cuda_engine(path.read_bytes())
            context = engine.create_execution_context()
            allocations = []
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                shape = tuple(engine.get_tensor_shape(name))
                size = int(np.prod(shape)) * dtype_bytes(engine.get_tensor_dtype(name))
                pointer = check(cudart.cudaMalloc(size))
                check(cudart.cudaMemsetAsync(pointer, 0, size, stream))
                context.set_tensor_address(name, pointer)
                allocations.append(pointer)
                if name == "kv_lengths":
                    host = np.full(shape, 120, dtype=np.int32)
                    check(cudart.cudaMemcpyAsync(pointer, host.ctypes.data, host.nbytes,
                                                 cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
            for _ in range(args.warmups):
                if not context.execute_async_v3(stream):
                    raise RuntimeError("TensorRT execute failed")
            check(cudart.cudaStreamSynchronize(stream))
            samples = []
            # Chunked events retain the distribution without measuring Python enqueue overhead.
            for _ in range(20):
                start = check(cudart.cudaEventCreate())
                end = check(cudart.cudaEventCreate())
                check(cudart.cudaEventRecord(start, stream))
                for _ in range(args.iterations // 20):
                    context.execute_async_v3(stream)
                check(cudart.cudaEventRecord(end, stream))
                check(cudart.cudaEventSynchronize(end))
                elapsed = check(cudart.cudaEventElapsedTime(start, end))
                samples.append(elapsed / (args.iterations // 20))
                check(cudart.cudaEventDestroy(start)); check(cudart.cudaEventDestroy(end))
            inspector = engine.create_engine_inspector()
            layer_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            results.append({
                "batch": batch,
                "decomposable": decomposable,
                "mean_ms": statistics.fmean(samples),
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "engine_bytes": path.stat().st_size,
                "layers": json.loads(layer_info),
            })
            print(batch, decomposable, f"{statistics.median(samples)*1000:.2f} us")
            for pointer in allocations:
                check(cudart.cudaFree(pointer))
    check(cudart.cudaStreamDestroy(stream))
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
