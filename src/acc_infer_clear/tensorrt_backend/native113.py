"""Opt-in TensorRT 11.3 Target-attention experiment for SM89.

This module intentionally lives outside the TensorRT 10 / Torch-TensorRT path.
The caller must point ``ACC_TRT113_SITE`` at the isolated 11.3 site-packages.
It uses fixed B/H/Q/K/D engines and a compact first-head KV arena; no precision
or scheduling semantics are changed.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _make_mask(keep, slots, lengths, out, KEEP0: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.arange(0, 1024)
    query = offset // K
    position = offset % K
    slot = tl.load(slots + row)
    length = tl.load(lengths + row)
    valid = (query < 8) & (position < K)
    allowed = (position <= length + query) & (tl.load(keep + slot * KEEP0 + position, mask=valid, other=0) > 0)
    tl.store(out + row * 8 * K + offset, allowed, mask=valid)


@triton.jit
def _make_draft_mask(lengths, out, K: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.arange(0, 1024)
    query = offset // K
    position = offset % K
    length = tl.load(lengths + row)
    valid = (query < 7) & (position < K)
    # The first 128 positions are the persistent context arena.  The final
    # seven are this round's noise tokens and are visible to every query,
    # matching SlotDraft's non-causal attention contract.
    allowed = (position < length) | (position >= 128)
    tl.store(out + row * 7 * K + offset, allowed, mask=valid)


def _import_trt113():
    loaded = sys.modules.get("tensorrt")
    if loaded is not None:
        version = getattr(loaded, "__version__", "")
        if not version.startswith("11.3"):
            raise RuntimeError(f"TensorRT {version} already loaded; native 11.3 isolation lost")
        return loaded
    site = os.environ.get("ACC_TRT113_SITE")
    if not site:
        raise RuntimeError("ACC_TRT113_SITE is required for the isolated TensorRT 11.3 backend")
    site = str(Path(site).resolve())
    sys.path.insert(0, site)
    try:
        import tensorrt as trt
    finally:
        sys.path.remove(site)
    if not trt.__version__.startswith("11.3"):
        raise RuntimeError(f"Expected TensorRT 11.3, got {trt.__version__}")
    return trt


class NativeTargetAttention113:
    """One TRT engine per batch and one execution context per Target layer."""

    def __init__(self, artifact_dir, layers, max_slots, device, dtype=torch.bfloat16):
        if dtype is not torch.bfloat16:
            raise ValueError("Native TRT 11.3 experiment is BF16-only")
        self.trt = _import_trt113()
        self.runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.layers = int(layers); self.max_slots = int(max_slots); self.device = device
        root = Path(artifact_dir)
        self.engines = {}; self.contexts = {}; self.outputs = {}; self.masks = {}
        for batch in (1, 4, 8, 16):
            path = root / f"target_attention_b{batch}.engine"
            if not path.exists() or batch > max_slots // 2:
                continue
            engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize {path}")
            self.engines[batch] = engine
            self.contexts[batch] = [engine.create_execution_context() for _ in range(self.layers)]
            output_dtype = torch.float32 if engine.get_tensor_dtype("out") == self.trt.float32 else dtype
            self.outputs[batch] = [torch.empty(batch, 20, 8, 64, device=device, dtype=output_dtype)
                                   for _ in range(self.layers)]
            self.masks[batch] = [torch.empty(batch, 1, 8, 128, device=device, dtype=torch.bool)
                                 for _ in range(self.layers)]
        self.cache = torch.empty(self.layers, 2, max_slots, 20, 128, 64,
                                 device=device, dtype=dtype)
        self.calls = 0

    def import_slot(self, packed, row, slot, length):
        if length > 128:
            raise ValueError("TRT first-head cache import exceeds K=128")
        self.cache[:, :, slot, :, :length].copy_(packed[:, :, row, :, :length])

    def can_run(self, batch, slots, lengths):
        # Slot identity is checked by the CPU caller before graph replay. TensorRT
        # binds the compact arena directly and therefore cannot gather arbitrary slots.
        return batch in self.engines and int(lengths.max().item()) + 8 <= 128 and torch.equal(
            slots.cpu(), torch.arange(batch, dtype=slots.dtype))

    def run(self, layer, qkv, keep, slots, lengths):
        batch = qkv.shape[0]
        context = self.contexts[batch][layer]
        mask = self.masks[batch][layer]
        _make_mask[(batch,)](keep, slots, lengths, mask, keep.stride(0), 128,
                             num_warps=4, num_stages=1)
        addresses = {
            "qkv": qkv.data_ptr(),
            "k_cache_in": self.cache[layer, 0, :batch].data_ptr(),
            "v_cache_in": self.cache[layer, 1, :batch].data_ptr(),
            "write_indices": lengths.data_ptr(), "mask": mask.data_ptr(),
            "k_cache_out": self.cache[layer, 0, :batch].data_ptr(),
            "v_cache_out": self.cache[layer, 1, :batch].data_ptr(),
            "out": self.outputs[batch][layer].data_ptr(),
        }
        for name, address in addresses.items():
            if not context.set_tensor_address(name, address):
                raise RuntimeError(f"TensorRT refused binding {name}")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT Target attention enqueue failed")
        self.calls += 1
        return self.outputs[batch][layer]

    def export_slots(self, target_storage, batch, length=128):
        target_storage[:, :, :batch, :, :length].copy_(self.cache[:, :, :batch, :, :length])


class NativeTargetFull113:
    """Fixed-batch full Target engine, including all 24 blocks and lm_head."""

    def __init__(self, engine_path, batch, max_slots, device):
        self.trt = _import_trt113(); self.batch = int(batch); self.device = device
        self.runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None: raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.cache = torch.empty(24, 2, max_slots, 20, 128, 64, device=device, dtype=torch.bfloat16)
        self.mask = torch.empty(batch, 1, 8, 128, device=device, dtype=torch.bool)
        self.logits = torch.empty(batch, 8, 8194, device=device, dtype=torch.float32)
        self.selected = torch.empty(batch, 8, 6400, device=device, dtype=torch.float32)
        self.final = torch.empty(batch, 8, 1280, device=device, dtype=torch.float32)
        self.calls = 0

    def import_slot(self, packed, row, slot, length):
        self.cache[:, :, slot, :, :length].copy_(packed[:, :, row, :, :length])

    def run(self, x, keep, slots, lengths):
        if x.shape[0] != self.batch: raise ValueError("Full Target engine batch changed")
        _make_mask[(self.batch,)](keep, slots, lengths, self.mask, keep.stride(0), 128,
                                  num_warps=4, num_stages=1)
        addresses = {"x": x.data_ptr(), "mask": self.mask.data_ptr(),
                     "write_indices": lengths.data_ptr(), "logits": self.logits.data_ptr(),
                     "selected": self.selected.data_ptr(), "final": self.final.data_ptr()}
        for layer in range(24):
            for bank, letter in ((0, "k"), (1, "v")):
                pointer = self.cache[layer, bank, :self.batch].data_ptr()
                addresses[f"{letter}_cache_in_{layer}"] = pointer
                addresses[f"{letter}_cache_out_{layer}"] = pointer
        for name, address in addresses.items():
            if not self.context.set_tensor_address(name, address):
                raise RuntimeError(f"TensorRT refused binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT full Target enqueue failed")
        self.calls += 1
        return self.logits, self.selected, self.final


class NativeTargetFullBank113:
    def __init__(self, plan_path, target):
        import json
        from acc_infer_clear.runtime.graphs import capture
        plan_file = Path(plan_path).resolve(); plan = json.loads(plan_file.read_text())
        self.backends = {}; self.graphs = {}; self.capture = capture; self.target = target
        for raw_batch, raw_path in plan["engines"].items():
            batch = int(raw_batch)
            if batch > target.max_batch: continue
            path = Path(raw_path); path = path if path.is_absolute() else (plan_file.parent / path).resolve()
            self.backends[batch] = NativeTargetFull113(path,batch,target.max_slots,target.storage.device)

    def import_slot(self, packed, row, slot, length):
        if length > 128:
            return
        for backend in self.backends.values(): backend.import_slot(packed,row,slot,length)

    def prepare_graphs(self):
        param=next(self.target.target.model.parameters())
        for batch,backend in self.backends.items():
            x=param.new_zeros(batch,8,self.target.target.model.transformer.embed_dim)
            slots=torch.arange(batch,device=param.device,dtype=torch.int32)
            lengths=torch.full((batch,),120,device=param.device,dtype=torch.int32)
            backend.cache.zero_()
            self.graphs[batch,128]=self.capture(lambda xx,ss,ll,b=backend:b.run(xx,self.target.keep,ss,ll),
                                                 (x,slots,lengths))
        return dict(keys=[list(k) for k in self.graphs],backend="TensorRT 11.3 full Target")

    def eligible(self,batch,slot_values,max_length):
        return (batch,128) in self.graphs and slot_values==list(range(batch)) and max_length+8<=128

    def export(self,batch,target_storage):
        target_storage[:,:,:batch,:,:128].copy_(self.backends[batch].cache[:,:,:batch])


class NativeDraftFull113:
    """Fixed-batch complete three-layer Draft backbone (proposal RNN excluded)."""

    def __init__(self, engine_path, batch, cache, device):
        self.trt = _import_trt113(); self.batch = int(batch); self.device = device
        self.runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None: raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context(); self.cache = cache
        self.mask = torch.empty(batch,1,7,135,device=device,dtype=torch.bool)
        self.hidden = torch.empty(batch,7,1280,device=device,dtype=torch.float32)
        self.base = torch.empty(batch,7,8194,device=device,dtype=torch.float32)
        self.calls = 0

    def run(self, x, slots, lengths):
        if x.shape[0] != self.batch: raise ValueError("Full Draft engine batch changed")
        # Identity-slot eligibility is a host-side scheduler invariant.  Do not
        # launch a comparison kernel here because this method is CUDA-captured.
        _make_draft_mask[(self.batch,)](lengths,self.mask,135,num_warps=4,num_stages=1)
        addresses={"x":x.data_ptr(),"mask":self.mask.data_ptr(),
                   "hidden":self.hidden.data_ptr(),"base":self.base.data_ptr()}
        for layer in range(3):
            addresses[f"k_cache_{layer}"]=self.cache[layer,0,:self.batch].data_ptr()
            addresses[f"v_cache_{layer}"]=self.cache[layer,1,:self.batch].data_ptr()
        for name,address in addresses.items():
            if not self.context.set_tensor_address(name,address):
                raise RuntimeError(f"TensorRT refused binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT full Draft enqueue failed")
        self.calls += 1
        return self.hidden,self.base


class NativeDraftFullBank113:
    """Batch-selective Draft engines sharing one compact K128 cache mirror."""

    def __init__(self, plan_path, draft):
        import json
        plan_file=Path(plan_path).resolve();plan=json.loads(plan_file.read_text())
        model=draft.model;pool=draft.pool;param=next(model.parameters())
        self.cache=torch.empty(len(model.layers),2,pool.max_slots,model.layers[0].num_heads,
                               128,model.layers[0].head_dim,device=param.device,dtype=param.dtype)
        self.backends={};self.graphs={};self.draft=draft
        for raw_batch,raw_path in plan["engines"].items():
            batch=int(raw_batch)
            if batch>pool.max_slots//2:continue
            path=Path(raw_path);path=path if path.is_absolute() else (plan_file.parent/path).resolve()
            self.backends[batch]=NativeDraftFull113(path,batch,self.cache,param.device)
        pool.native_bank=self

    def import_slot(self, cache, slot):
        length=int(cache.length)
        if length>128:return
        for layer,(key,value) in enumerate(zip(cache.keys,cache.values)):
            self.cache[layer,0,slot,:,:length].copy_(key[0,:,:length])
            self.cache[layer,1,slot,:,:length].copy_(value[0,:,:length])

    def prepare_graphs(self):
        from acc_infer_clear.runtime.graphs import capture
        model=self.draft.model;param=next(model.parameters());self.cache.zero_()
        for batch,backend in self.backends.items():
            anchors=torch.zeros(batch,device=param.device,dtype=torch.long)
            positions=torch.arange(7,device=param.device)[None].expand(batch,-1).clone()
            slots=torch.arange(batch,device=param.device,dtype=torch.int32)
            lengths=torch.full((batch,),121,device=param.device,dtype=torch.int32)
            self.graphs[batch,128]=capture(
                lambda aa,pp,ss,ll,b=backend:b.run(model._noise_embeddings(aa,pp),ss,ll),
                (anchors,positions,slots,lengths))
        return dict(keys=[list(k) for k in self.graphs],backend="TensorRT 11.3 full Draft")

    def enabled_for_total(self,total):
        # ``total`` is the padded count of newly committed context tokens, not
        # the active request batch.  Coupling it to an engine batch leaves the
        # compact TRT cache stale whenever requests accept fewer than 8 tokens.
        # The scatter metadata already carries the real slots and lengths, so
        # every captured context bucket must mirror its writes.
        return bool(self.backends)


class NativeCFMSolver113:
    """Static B8/F310 full two-step CFM Solver on the shared TRT 11.3 runtime."""

    def __init__(self, plan_path, eager):
        import json
        plan_file=Path(plan_path).resolve();plan=json.loads(plan_file.read_text())
        if int(plan.get("format",-1))!=1 or int(plan.get("batch",-1))!=8 or int(plan.get("frames",-1))!=310:
            raise ValueError("Expected static TensorRT 11.3 B8/F310 CFM plan")
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        if hashlib.sha256(engine_path.read_bytes()).hexdigest()!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 CFM engine hash mismatch")
        self.trt=_import_trt113();self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine=self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context=self.engine.create_execution_context();self.eager=eager
        self.model=eager.model;self.times=eager.times;self.identity=dict(eager.identity)
        self.identity.update(backend="TensorRT 11.3 native",batch=8,frames=310,plan=str(plan_file))
        self.observer=None;self.calls=0;self.fallbacks=0
        device=next(eager.model.parameters()).device
        self.output=torch.empty(8,80,310,device=device,dtype=torch.float32)

    def __call__(self,x,prompt,lengths,style,mu,mask):
        signature=(x.shape==(8,80,310) and prompt.shape==(8,80,310) and
                   lengths.shape==(8,) and style.shape==(8,192) and mu.shape==(8,310,512) and
                   mask.shape==(8,1,310) and x.dtype is torch.float32 and
                   prompt.dtype is torch.float32 and lengths.dtype is torch.int64 and
                   style.dtype is torch.float32 and mu.dtype is torch.float32 and mask.dtype is torch.bool)
        if not signature:
            self.fallbacks+=1
            return self.eager(x,prompt,lengths,style,mu,mask)
        addresses={"x":x.data_ptr(),"prompt":prompt.data_ptr(),"lengths":lengths.data_ptr(),
                   "style":style.data_ptr(),"mu":mu.data_ptr(),"mask":mask.data_ptr(),
                   "output":self.output.data_ptr()}
        for name,address in addresses.items():
            if not self.context.set_tensor_address(name,address):
                raise RuntimeError(f"TensorRT refused CFM binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
            raise RuntimeError("TensorRT full CFM enqueue failed")
        self.calls+=1
        return self.output

    def stats(self):
        return dict(backend="TensorRT 11.3 native full two-step CFM Solver",batch=8,frames=310,
                    calls=self.calls,fallbacks=self.fallbacks,identity=self.identity)


class NativeVocoder113:
    """Static B8/F52 native BigVGAN engine with in-engine Quick Plugins."""

    def __init__(self, plan_path, eager):
        import json
        from acc_infer_clear.tensorrt_backend.vocoder_plugin import register

        plan_file=Path(plan_path).resolve();plan=json.loads(plan_file.read_text())
        if int(plan.get("format",-1))!=1 or int(plan.get("batch",-1))!=8 or int(plan.get("frames",-1))!=52:
            raise ValueError("Expected static TensorRT 11.3 B8/F52 Vocoder plan")
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        if hashlib.sha256(engine_path.read_bytes()).hexdigest()!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 Vocoder engine hash mismatch")
        self.trt=_import_trt113();register()
        self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine=self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        self.context=self.engine.create_execution_context();self.eager=eager
        self.calls=0;self.fallbacks=0;self.plan=str(plan_file)
        self.output=torch.empty(8,1,13312,device="cuda",dtype=torch.float32)

    def __call__(self,mel):
        if mel.shape!=(8,80,52) or mel.dtype is not torch.float32:
            self.fallbacks+=1;return self.eager(mel)
        for name,address in (("mel",mel.data_ptr()),("pcm",self.output.data_ptr())):
            if not self.context.set_tensor_address(name,address):
                raise RuntimeError(f"TensorRT refused Vocoder binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(mel.device).cuda_stream):
            raise RuntimeError("TensorRT full Vocoder enqueue failed")
        self.calls+=1;return self.output

    def stats(self):
        return dict(backend="TensorRT 11.3 native BigVGAN with alias-free plugin",
                    batch=8,frames=52,calls=self.calls,fallbacks=self.fallbacks,plan=self.plan)
