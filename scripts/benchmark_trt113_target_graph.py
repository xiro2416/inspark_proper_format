#!/usr/bin/env python3
"""A/B the complete 24-layer Target graph with native TRT 11.3 attention."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def measure(torch, call, warmups=20, iterations=200):
    for _ in range(warmups): call()
    torch.cuda.synchronize(); samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations // 20): call()
        end.record(); end.synchronize(); samples.append(start.elapsed_time(end) / (iterations // 20))
    return {"mean_ms": statistics.fmean(samples), "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--batch", type=int, required=True); p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--deployment", default="configs/sm89_bf16_target_trt113_lab.json")
    p.add_argument("--artifacts", default="artifacts/trt113_target_qkv_attention")
    p.add_argument("--full-engine")
    p.add_argument("--ref-audio", default="outputs/profile_sm89/reference.wav")
    p.add_argument("--out", required=True); args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from acc_infer_clear.config import load as load_config
    from acc_infer_clear.runtime.deployment import load as load_deployment
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.runtime.graphs import capture
    from acc_infer_clear.streaming.engine import Engine
    from acc_infer_clear.kernels.device_commit import mark_keep
    from acc_infer_clear.kernels.kv_attention import append, attention
    from acc_infer_clear.tensorrt_backend.native113 import NativeTargetAttention113,NativeTargetFull113

    config = load_config(args.config); config["max_batch"] = args.batch
    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_reference("reference", args.ref_audio)
            engine.prepare_deployment(load_deployment(args.deployment))
            target = engine.rt.target
            backend = (NativeTargetFull113(args.full_engine,args.batch,target.max_slots,target.storage.device)
                       if args.full_engine else NativeTargetAttention113(
                           args.artifacts,len(target.target.model.transformer.h),target.max_slots,target.storage.device))
            b = args.batch; param = next(target.target.model.parameters())
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                if not args.full_engine: target.attach_native_attention(backend)
                static_x = param.new_zeros(b, 8, target.target.model.transformer.embed_dim)
                static_slots = torch.arange(b, device=param.device, dtype=torch.int32)
                static_lengths = torch.full((b,), 120, device=param.device, dtype=torch.int32)
                target.storage.zero_(); target.keep.fill_(1); backend.cache.zero_()
                current_graph = capture(lambda x,s,l: target.math(x,s,l,128),
                                        (static_x, static_slots, static_lengths))
                native_graph = capture((lambda x,s,l: backend.run(x,target.keep,s,l)) if args.full_engine else
                                       (lambda x,s,l: target.native_math(x,s,l,128)),
                                       (static_x, static_slots, static_lengths))
            memory_after_capture = {"allocated_mib": torch.cuda.memory_allocated() >> 20,
                                    "reserved_mib": torch.cuda.memory_reserved() >> 20}

            identifiers = [f"target-lab-{i}" for i in range(b)]
            for ident in identifiers:
                engine.create_session(ident, "reference", 0); engine.push_text(ident, "他正在整理文件。"); engine.finish_input(ident)
            sessions = [engine.sessions[i] for i in identifiers]
            with torch.cuda.stream(engine.model.stream), torch.inference_mode(): rows = engine.prepare_rows(sessions, 0)
            slots = torch.tensor([r.kv.slot for r in rows], device=param.device, dtype=torch.int32)
            lengths = torch.tensor([r.past_length for r in rows], device=param.device, dtype=torch.int32)
            slot_values = [r.kv.slot for r in rows]
            if slot_values != list(range(b)):
                raise RuntimeError(f"Native compact-cache precondition failed: slots={slot_values}")
            if int(lengths.max().item()) + 8 > 128:
                raise RuntimeError(f"Initial Target history exceeds K128: {lengths.tolist()}")
            mark_keep(target.keep, slots, lengths)
            if args.full_engine:
                backend.cache[:,:,:b].copy_(target.storage[:,:,:b,:,:128])
            torch.manual_seed(20260922)
            x = torch.randn(b, 8, target.target.model.transformer.embed_dim,
                            device=param.device, dtype=param.dtype)
            with torch.inference_mode():
                attention_correctness=None
                if not args.full_engine:
                    block = target.target.model.transformer.h[0]; a = block.attn
                    qkv = a.c_attn(block.ln_1(x)); q,k,v = qkv.split(a.split_size, dim=2)
                    q = q.view(b,8,a.num_heads,a.head_dim).transpose(1,2)
                    k = k.view(b,8,a.num_heads,a.head_dim).transpose(1,2)
                    v = v.view(b,8,a.num_heads,a.head_dim).transpose(1,2)
                    cache_before=(target.storage[0,:,:b,:,:128]-backend.cache[0,:,:b]).abs().float()
                    append(k,v,target.storage[0,0],target.storage[0,1],slots,lengths)
                    current_attention = attention(q,target.storage[0,0],target.storage[0,1],target.keep,slots,lengths,128)
                    native_attention = backend.run(0,qkv,target.keep,slots,lengths)
                    positions=torch.arange(128,device=param.device)[None,None,None,:]
                    queries=torch.arange(8,device=param.device)[None,None,:,None]
                    expected_mask=(positions<=lengths[:,None,None,None]+queries)&target.keep[slots,:128].bool()[:,None,None,:]
                    torch_reference=torch.nn.functional.scaled_dot_product_attention(
                        q,backend.cache[0,0,:b].to(q.dtype),backend.cache[0,1,:b].to(q.dtype),attn_mask=expected_mask)
                    torch.cuda.synchronize(); adelta=(current_attention.float()-native_attention.float()).abs()
                    attention_correctness={"max_abs":float(adelta.max()),"mean_abs":float(adelta.mean()),
                        "cosine":float(torch.nn.functional.cosine_similarity(current_attention.float().flatten(),native_attention.float().flatten(),dim=0)),
                        "cache_before_max_abs":float(cache_before.max()),
                        "cache_after_max_abs":float((target.storage[0,:,:b,:,:128]-backend.cache[0,:,:b]).abs().float().max()),
                        "mask_mismatches":int((expected_mask!=backend.masks[b][0]).sum()),
                        "torch_reference_max_abs":float((torch_reference.float()-native_attention.float()).abs().max())}
                current = current_graph(x, slots, lengths)
                native = native_graph(x, slots, lengths)
                torch.cuda.synchronize()
                c = current[0].float(); n = native[0].float(); delta = (c - n).abs()
                correctness = {"logits_max_abs": float(delta.max()), "logits_mean_abs": float(delta.mean()),
                               "logits_cosine": float(torch.nn.functional.cosine_similarity(c.flatten(), n.flatten(), dim=0))}
                current_time = measure(torch, lambda: current_graph(x, slots, lengths))
                native_time = measure(torch, lambda: native_graph(x, slots, lengths))
            result = {"batch": b, "slots": slot_values, "lengths": lengths.cpu().tolist(),
                      "current": current_time, "native_trt113": native_time,
                      "speedup": current_time["median_ms"] / native_time["median_ms"],
                      "delta_pct": (native_time["median_ms"] / current_time["median_ms"] - 1) * 100,
                      "correctness": correctness, "attention_correctness":attention_correctness,
                      "memory_after_capture": memory_after_capture,
                      "trt_version": backend.trt.__version__, "native_attention_calls": backend.calls}
            destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
        finally:
            engine.close()


if __name__ == "__main__": main()
