#!/usr/bin/env python3
"""Two-environment numerical gate for TRT 11.3 Target attention.

Run ``--backend reference`` in the project PyTorch environment, then run
``--backend tensorrt`` in .venv-trt113.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def make_reference(root):
    import torch
    from acc_infer_clear.kernels.kv_attention import append, attention
    torch.manual_seed(20260922)
    for batch in (1, 4, 8, 16):
        q = torch.randn(batch, 20, 8, 64, device="cuda", dtype=torch.bfloat16)
        ku = torch.randn_like(q); vu = torch.randn_like(q)
        kc = torch.randn(batch, 20, 128, 64, device="cuda", dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        lengths = torch.full((batch,), 120, device="cuda", dtype=torch.int32)
        keep = torch.ones(batch, 128, device="cuda", dtype=torch.int32)
        append(ku, vu, kc, vc, slots, lengths)
        out = attention(q, kc, vc, keep, slots, lengths, 128)
        torch.cuda.synchronize()
        np.savez(root / f"b{batch}.npz", q=q.float().cpu().numpy(), ku=ku.float().cpu().numpy(),
                 vu=vu.float().cpu().numpy(), kc=kc.float().cpu().numpy(), vc=vc.float().cpu().numpy(),
                 reference=out.float().cpu().numpy())


def run_tensorrt(root, artifacts):
    import ml_dtypes
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    def check(result):
        if result[0] != cudart.cudaError_t.cudaSuccess: raise RuntimeError(result)
        return result[1] if len(result) > 1 else None

    check(cudart.cudaSetDevice(0)); stream = check(cudart.cudaStreamCreate())
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR)); results = []
    for batch in (1, 4, 8, 16):
        data = np.load(root / f"b{batch}.npz")
        engine = runtime.deserialize_cuda_engine((artifacts / f"target_attention_b{batch}.engine").read_bytes())
        context = engine.create_execution_context(); ptrs = {}; owned = []; hosts = {}
        arrays = {
            "q": data["q"].astype(ml_dtypes.bfloat16),
            "k_update": data["ku"].astype(ml_dtypes.bfloat16),
            "v_update": data["vu"].astype(ml_dtypes.bfloat16),
            # Reference files contain caches after append; restore the old region and
            # let TRT overwrite the final eight positions with the same updates.
            "k_cache_in": data["kc"].astype(ml_dtypes.bfloat16),
            "v_cache_in": data["vc"].astype(ml_dtypes.bfloat16),
            "write_indices": np.full((batch,), 120, np.int32),
        }
        input_names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                       if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT}
        if "qkv" in input_names:
            def token_major(name):
                return data[name].transpose(0, 2, 1, 3).reshape(batch, 8, 1280)
            arrays["qkv"] = np.concatenate((token_major("q"), token_major("ku"),
                                             token_major("vu")), axis=-1).astype(ml_dtypes.bfloat16)
        if "kv_lengths" in input_names:
            arrays["kv_lengths"] = np.full((batch,), 128, np.int32)
        if "mask" in input_names:
            positions = np.arange(128)[None, None, None, :]
            queries = np.arange(8)[None, None, :, None]
            arrays["mask"] = np.broadcast_to(positions <= 120 + queries, (batch, 1, 8, 128)).copy()
        arrays = {name: value for name, value in arrays.items() if name in input_names}
        arrays["k_cache_in"][:, :, 120:] = 0
        arrays["v_cache_in"][:, :, 120:] = 0
        for name, host in arrays.items():
            ptr = check(cudart.cudaMalloc(host.nbytes)); owned.append(ptr); ptrs[name] = ptr
            check(cudart.cudaMemcpyAsync(ptr, host.ctypes.data, host.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        for name in ("k_cache_out", "v_cache_out"):
            source = engine.get_aliased_input_tensor(name); ptrs[name] = ptrs[source]
        out = np.empty((batch, 20, 8, 64), dtype=ml_dtypes.bfloat16); hosts["out"] = out
        ptrs["out"] = check(cudart.cudaMalloc(out.nbytes)); owned.append(ptrs["out"])
        for name, ptr in ptrs.items(): context.set_tensor_address(name, ptr)
        if not context.execute_async_v3(stream): raise RuntimeError("execute failed")
        check(cudart.cudaMemcpyAsync(out.ctypes.data, ptrs["out"], out.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        check(cudart.cudaStreamSynchronize(stream))
        actual = out.astype(np.float32); expected = data["reference"]
        delta = np.abs(actual - expected)
        denom = np.maximum(np.abs(expected), 1e-3)
        results.append({"batch": batch, "max_abs": float(delta.max()),
                        "mean_abs": float(delta.mean()), "max_rel": float((delta / denom).max()),
                        "cosine": float(np.dot(actual.ravel(), expected.ravel()) /
                                        (np.linalg.norm(actual.ravel()) * np.linalg.norm(expected.ravel())))})
        for ptr in owned: check(cudart.cudaFree(ptr))
    check(cudart.cudaStreamDestroy(stream))
    (root / "results.json").write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--backend", choices=("reference", "tensorrt"), required=True)
    p.add_argument("--root", default="outputs/trt113_target_validation")
    p.add_argument("--artifacts", default="artifacts/trt113_target_attention_masked"); args = p.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    if args.backend == "reference": make_reference(root)
    else: run_tensorrt(root, Path(args.artifacts))


if __name__ == "__main__": main()
