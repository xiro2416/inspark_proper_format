#!/usr/bin/env python3
"""Benchmark TRT 11.3 KVCacheUpdate + causal Target attention."""
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


def nbytes(engine, name):
    widths = {trt.DataType.BF16: 2, trt.DataType.HALF: 2,
              trt.DataType.FLOAT: 4, trt.DataType.INT32: 4, trt.DataType.BOOL: 1}
    return int(np.prod(tuple(engine.get_tensor_shape(name)))) * widths[engine.get_tensor_dtype(name)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", default="artifacts/trt113_target_attention")
    p.add_argument("--out", default="outputs/trt113_target_attention_benchmark.json")
    p.add_argument("--warmups", type=int, default=200)
    p.add_argument("--iterations", type=int, default=2000)
    args = p.parse_args()
    check(cudart.cudaSetDevice(0)); stream = check(cudart.cudaStreamCreate())
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR)); results = []
    for batch in (1, 4, 8, 16):
        path = Path(args.artifact_dir) / f"target_attention_b{batch}.engine"
        engine = runtime.deserialize_cuda_engine(path.read_bytes()); context = engine.create_execution_context()
        pointers = {}; owned = []
        # Inputs first. KV outputs must bind to exactly the same allocation.
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            ptr = check(cudart.cudaMalloc(nbytes(engine, name))); owned.append(ptr); pointers[name] = ptr
            check(cudart.cudaMemsetAsync(ptr, 0, nbytes(engine, name), stream))
            if name in ("write_indices", "kv_lengths"):
                value = 120 if name == "write_indices" else 128
                host = np.full(tuple(engine.get_tensor_shape(name)), value, dtype=np.int32)
                check(cudart.cudaMemcpyAsync(ptr, host.ctypes.data, host.nbytes,
                                             cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
            elif name == "mask":
                positions = np.arange(128)[None, None, None, :]
                queries = np.arange(8)[None, None, :, None]
                host = np.broadcast_to(positions <= 120 + queries, tuple(engine.get_tensor_shape(name))).copy()
                check(cudart.cudaMemcpyAsync(ptr, host.ctypes.data, host.nbytes,
                                             cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        aliases = {}
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                continue
            source = engine.get_aliased_input_tensor(name)
            aliases[name] = source
            if source:
                pointers[name] = pointers[source]
            else:
                ptr = check(cudart.cudaMalloc(nbytes(engine, name))); owned.append(ptr); pointers[name] = ptr
        for name, ptr in pointers.items():
            if not context.set_tensor_address(name, ptr):
                raise RuntimeError(f"failed to bind {name}")
        for _ in range(args.warmups):
            if not context.execute_async_v3(stream): raise RuntimeError("execute failed")
        check(cudart.cudaStreamSynchronize(stream)); samples = []
        for _ in range(20):
            start = check(cudart.cudaEventCreate()); end = check(cudart.cudaEventCreate())
            check(cudart.cudaEventRecord(start, stream))
            for _ in range(args.iterations // 20): context.execute_async_v3(stream)
            check(cudart.cudaEventRecord(end, stream)); check(cudart.cudaEventSynchronize(end))
            elapsed = check(cudart.cudaEventElapsedTime(start, end)); samples.append(elapsed / (args.iterations // 20))
            check(cudart.cudaEventDestroy(start)); check(cudart.cudaEventDestroy(end))
        inspector = engine.create_engine_inspector()
        results.append({"batch": batch, "mean_ms": statistics.fmean(samples),
                        "median_ms": statistics.median(samples), "min_ms": min(samples),
                        "max_ms": max(samples), "aliases": aliases,
                        "layers": json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))})
        print(batch, f"{statistics.median(samples)*1000:.2f} us", aliases)
        for ptr in owned: check(cudart.cudaFree(ptr))
    check(cudart.cudaStreamDestroy(stream))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__": main()
