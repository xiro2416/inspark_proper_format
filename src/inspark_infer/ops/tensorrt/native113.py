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
from copy import deepcopy
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
        expected = _ar_io_signature(self.trt, 'target', self.batch)
        _validate_engine_io(self.engine, self.trt, expected, 'Target')
        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError('Failed to create Target execution context')
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
        from inspark_infer.runtime.graphs import capture
        plan_file = Path(plan_path).resolve(); plan = json.loads(plan_file.read_text())
        self.backends = {}; self.graphs = {}; self.capture = capture; self.target = target
        self.artifacts = {}
        for raw_batch, raw_path in plan["engines"].items():
            batch = int(raw_batch)
            if batch > target.max_batch: continue
            path, identity = _plan_engine_identity(plan_file, plan, raw_batch, raw_path)
            self.artifacts[batch] = identity
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
        return dict(keys=[list(k) for k in self.graphs],backend="TensorRT 11.3 full Target",
                    artifacts=self.artifacts)

    def eligible(self,batch,slot_values,max_length):
        return (batch,128) in self.graphs and slot_values==list(range(batch)) and max_length+8<=128

    def host_engine_batch(self,batch,max_length):
        if max_length+8>128:return None
        return next((size for size in sorted(self.backends) if size>=batch and (size,128) in self.graphs),None)

    def run_with_canonical_cache(self,x,slots,lengths,slot_values,host_lengths):
        """Host-scheduled request-local path, independent of device RNG.

        A previous call may have used another batch engine or the generic path.
        Refresh the bounded prefix from the canonical pool before every call;
        export afterward so cropping, cancellation and later fallback see the
        actual native KV. Copies are part of end-to-end measurements. The
        existing device-round path retains its once-per-loop synchronization.
        """
        batch=len(host_lengths)
        engine_batch=self.host_engine_batch(batch,max(host_lengths))
        if engine_batch is None:raise ValueError('No native Target batch with KV+8 <= 128')
        backend=self.backends[engine_batch]
        # The fixed engine sees dense rows; the canonical request-owned KV slots
        # remain untouched except for the rows that actually ran.  Inactive
        # padding rows are discarded, including their engine-local KV writes.
        for row,(slot,length) in enumerate(zip(slot_values,host_lengths)):
            backend.import_slot(self.target.storage,slot,row,length)
        if engine_batch==batch:
            padded_x,padded_slots,padded_lengths=x,slots,lengths
        else:
            padded_x=x.new_zeros((engine_batch,*x.shape[1:]));padded_x[:batch].copy_(x)
            padded_slots=slots.new_full((engine_batch,),slot_values[0]);padded_slots[:batch].copy_(slots)
            padded_lengths=lengths.new_zeros((engine_batch,));padded_lengths[:batch].copy_(lengths)
        result=self.graphs[engine_batch,128](padded_x,padded_slots,padded_lengths)
        for row,(slot,length) in enumerate(zip(slot_values,host_lengths)):
            self.target.storage[:,:,slot,:,:length+8].copy_(backend.cache[:,:,row,:,:length+8])
        return tuple(tensor[:batch].clone() for tensor in result)

    def export(self,batch,target_storage):
        target_storage[:,:,:batch,:,:128].copy_(self.backends[batch].cache[:,:,:batch])


class NativeDraftFull113:
    """Fixed-batch complete three-layer Draft backbone (proposal RNN excluded)."""

    def __init__(self, engine_path, batch, cache, device):
        self.trt = _import_trt113(); self.batch = int(batch); self.device = device
        self.runtime = self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None: raise RuntimeError(f"Failed to deserialize {engine_path}")
        expected = _ar_io_signature(self.trt, 'draft', self.batch)
        _validate_engine_io(self.engine, self.trt, expected, 'Draft')
        self.context = self.engine.create_execution_context(); self.cache = cache
        if self.context is None: raise RuntimeError('Failed to create Draft execution context')
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
        self.backends={};self.graphs={};self.draft=draft;self.artifacts={}
        for raw_batch,raw_path in plan["engines"].items():
            batch=int(raw_batch)
            if batch>pool.max_slots//2:continue
            path, identity = _plan_engine_identity(plan_file, plan, raw_batch, raw_path)
            self.artifacts[batch] = identity
            self.backends[batch]=NativeDraftFull113(path,batch,self.cache,param.device)
        pool.native_bank=self

    def import_slot(self, cache, slot):
        length=int(cache.length)
        if length>128:return
        for layer,(key,value) in enumerate(zip(cache.keys,cache.values)):
            self.cache[layer,0,slot,:,:length].copy_(key[0,:,:length])
            self.cache[layer,1,slot,:,:length].copy_(value[0,:,:length])

    def prepare_graphs(self):
        from inspark_infer.runtime.graphs import capture
        model=self.draft.model;param=next(model.parameters());self.cache.zero_()
        for batch,backend in self.backends.items():
            anchors=torch.zeros(batch,device=param.device,dtype=torch.long)
            positions=torch.arange(7,device=param.device)[None].expand(batch,-1).clone()
            slots=torch.arange(batch,device=param.device,dtype=torch.int32)
            lengths=torch.full((batch,),121,device=param.device,dtype=torch.int32)
            self.graphs[batch,128]=capture(
                lambda aa,pp,ss,ll,b=backend:b.run(model._noise_embeddings(aa,pp),ss,ll),
                (anchors,positions,slots,lengths))
        return dict(keys=[list(k) for k in self.graphs],backend="TensorRT 11.3 full Draft",
                    artifacts=self.artifacts)

    def host_engine_batch(self,batch,max_length):
        if max_length+7>128:return None
        return next((size for size in sorted(self.backends) if size>=batch and (size,128) in self.graphs),None)

    def run_packed(self,engine_batch,anchors,positions,slots,lengths,canonical_storage):
        """Replay a fixed engine for fewer live rows without altering their KV.

        Native Draft reads the compact mirror but never writes it.  Pack the
        request-owned canonical prefixes into engine rows, then restore the
        mirror so a later exact-batch graph or Context append sees its own slots.
        """
        batch=anchors.shape[0]
        if engine_batch<batch or (engine_batch,128) not in self.graphs:
            raise ValueError('No native Draft engine for packed host batch')
        slot_values=slots.tolist()
        for row,slot in enumerate(slot_values):
            self.cache[:,:,row].copy_(canonical_storage[:,:,slot,:,:128])
        padded_anchors=anchors.new_zeros((engine_batch,));padded_anchors[:batch].copy_(anchors)
        padded_positions=positions.new_zeros((engine_batch,*positions.shape[1:]));padded_positions[:batch].copy_(positions)
        padded_slots=torch.arange(engine_batch,device=slots.device,dtype=slots.dtype)
        padded_lengths=lengths.new_zeros((engine_batch,));padded_lengths[:batch].copy_(lengths)
        try:
            hidden,base=self.graphs[engine_batch,128](padded_anchors,padded_positions,padded_slots,padded_lengths)
            return hidden[:batch].clone(),base[:batch].clone()
        finally:
            self.cache[:,:,:engine_batch].copy_(canonical_storage[:,:,:engine_batch,:,:128])

    def enabled_for_total(self,total):
        # ``total`` is the padded count of newly committed context tokens, not
        # the active request batch.  Coupling it to an engine batch leaves the
        # compact TRT cache stale whenever requests accept fewer than 8 tokens.
        # The scatter metadata already carries the real slots and lengths, so
        # every captured context bucket must mirror its writes.
        return bool(self.backends)


def _plan_engine_identity(plan_file, plan, raw_batch, raw_path):
    path = Path(raw_path)
    path = path if path.is_absolute() else (plan_file.parent / path).resolve()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    expected = plan.get('engine_sha256', {}).get(str(raw_batch))
    if expected is not None and digest != expected:
        raise ValueError(f'TensorRT engine hash mismatch: {path}')
    provenance = plan.get('provenance', {}).get(str(raw_batch))
    return path, dict(path=str(path), sha256=digest, plan_hash_verified=expected is not None,
                      provenance_status='recorded_not_audited' if provenance else 'legacy_unverified',
                      numerical_audit_pass=False)


def _ar_io_signature(trt, component, batch):
    inp, out = trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT
    if component == 'target':
        expected = {'x': ((batch, 8, 1280), trt.float32, inp),
                    'mask': ((batch, 1, 8, 128), trt.bool, inp),
                    'write_indices': ((batch,), trt.int32, inp),
                    'logits': ((batch, 8, 8194), trt.float32, out),
                    'selected': ((batch, 8, 6400), trt.float32, out),
                    'final': ((batch, 8, 1280), trt.float32, out)}
        for layer in range(24):
            for letter in ('k', 'v'):
                for direction, mode in (('in', inp), ('out', out)):
                    expected[f'{letter}_cache_{direction}_{layer}'] = ((batch, 20, 128, 64), trt.bfloat16, mode)
        return expected
    if component == 'draft':
        expected = {'x': ((batch, 7, 1280), trt.float32, inp),
                    'mask': ((batch, 1, 7, 135), trt.bool, inp),
                    'hidden': ((batch, 7, 1280), trt.float32, out),
                    'base': ((batch, 7, 8194), trt.float32, out)}
        for layer in range(3):
            for letter in ('k', 'v'):
                expected[f'{letter}_cache_{layer}'] = ((batch, 20, 128, 64), trt.float32, inp)
        return expected
    raise ValueError(f'Unknown AR component: {component}')


def _validate_engine_io(engine, trt, expected, component):
    """Validate the physical binding contract before allocating/enqueuing buffers."""
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"TensorRT {component} I/O names mismatch: {names}; expected {list(expected)}")
    for name, (shape, dtype, mode) in expected.items():
        actual = (tuple(engine.get_tensor_shape(name)), engine.get_tensor_dtype(name),
                  engine.get_tensor_mode(name))
        if actual != (shape, dtype, mode):
            raise ValueError(f"TensorRT {component} I/O mismatch for {name}: {actual}; "
                             f"expected {(shape, dtype, mode)}")
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"TensorRT {component} I/O {name} must use device memory")
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ValueError(f"TensorRT {component} I/O {name} must use contiguous LINEAR format")


def _validate_acoustic_io(engine, trt, expected):
    return _validate_engine_io(engine, trt, expected, 'acoustic')


def _acoustic_provenance_metadata(plan_file, plan, component, engine_sha256):
    """Freeze build declarations, without pretending to check loader weights.

    No model files are opened here. Full role-by-role checkpoint verification is
    an offline audit responsibility, separate from deserialization and routing.
    """
    declared = plan.get("provenance")
    recorded = (isinstance(declared, dict) and declared.get("status") == "recorded_not_audited"
                and declared.get("schema") == 1 and declared.get("component") == component)
    return dict(status="recorded_not_audited" if recorded else "legacy_unverified",
                declared_status=declared.get("status") if isinstance(declared, dict) else None,
                declared_plan_status=plan.get("provenance_status"),
                plan_sha256=hashlib.sha256(Path(plan_file).read_bytes()).hexdigest(),
                engine_sha256=engine_sha256, engine_hash_verified=plan.get("sha256") == engine_sha256,
                weight_identity_verified=False, loader_weights_checked=False,
                verification_scope="build declaration only; offline audit must compare actual loader files",
                build_provenance=deepcopy(declared) if isinstance(declared, dict) else None)


def _acoustic_route(wrapper, args):
    """Read tensor metadata only: safe during capture and without a CUDA context."""
    reason = None
    for (name, shape, dtype), value in zip(wrapper.input_signature, args):
        if not isinstance(value, torch.Tensor):
            reason = f"{name}.type"
        elif tuple(value.shape) != shape:
            reason = f"{name}.shape"
        elif value.dtype != dtype:
            reason = f"{name}.dtype"
        elif value.device != wrapper.device or value.device.type != "cuda":
            reason = f"{name}.device"
        elif value.layout != torch.strided or not value.is_contiguous():
            reason = f"{name}.layout"
        if reason is not None:
            break
    if len(args) != len(wrapper.input_signature):
        reason = "argument_count"
    first = args[0] if args else None
    shape = tuple(first.shape) if isinstance(first, torch.Tensor) else ()
    return dict(kind="tensorrt" if reason is None else "eager",
                backend="tensorrt113" if reason is None else "eager",
                candidate_backend="tensorrt113", component=wrapper.component,
                batch=shape[0] if shape else None, frames=shape[-1] if len(shape) == 3 else None,
                engine_batch=wrapper.batch, engine_frames=wrapper.frames,
                plan=wrapper.plan, sha256=wrapper.engine_sha256, reason=reason,
                provenance_status=wrapper.provenance["status"],
                plan_sha256=wrapper.provenance["plan_sha256"],
                weight_identity_verified=False, loader_weights_checked=False,
                plugins=list(wrapper.plugins))


class NativeCFMSolver113:
    """Static B1/B4/B8, F310/P258 two-step CFM on the shared TRT 11.3 runtime."""

    def __init__(self, plan_path, eager):
        import json
        plan_file=Path(plan_path).resolve();plan=json.loads(plan_file.read_text())
        if (plan.get("format") != 1 or type(plan.get("batch")) is not int or
                plan["batch"] not in (1, 4, 8) or plan.get("frames") != 310 or
                plan.get("prompt_frames") != 258):
            raise ValueError("Expected static TensorRT 11.3 B1/B4/B8, F310/P258 CFM plan")
        self.batch=plan["batch"];self.frames=310;self.component="cfm";self.plugins=[]
        self.plan=str(plan_file)
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        serialized=engine_path.read_bytes();self.engine_sha256=hashlib.sha256(serialized).hexdigest()
        if self.engine_sha256!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 CFM engine hash mismatch")
        self.provenance=_acoustic_provenance_metadata(plan_file,plan,"cfm",self.engine_sha256)
        self.trt=_import_trt113();self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine=self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        b=self.batch;trt=self.trt
        self.input_signature=(("x",(b,80,310),torch.float32),
                              ("prompt",(b,80,310),torch.float32),
                              ("lengths",(b,),torch.int64),
                              ("style",(b,192),torch.float32),
                              ("mu",(b,310,512),torch.float32),
                              ("mask",(b,1,310),torch.bool))
        dtypes={torch.float32:trt.float32,torch.int64:trt.int64,torch.bool:trt.bool}
        expected={name:(shape,dtypes[dtype],trt.TensorIOMode.INPUT)
                  for name,shape,dtype in self.input_signature}
        expected["output"]=((b,80,310),trt.float32,trt.TensorIOMode.OUTPUT)
        _validate_acoustic_io(self.engine,trt,expected)
        self.context=self.engine.create_execution_context();self.eager=eager
        if self.context is None:raise RuntimeError("Failed to create TensorRT CFM execution context")
        self.model=eager.model;self.times=eager.times;self.identity=dict(eager.identity)
        self.identity.update(backend="TensorRT 11.3 native",batch=b,frames=310,
                             prompt_frames=258,plan=self.plan,engine_sha256=self.engine_sha256,
                             provenance_status=self.provenance["status"],
                             plan_sha256=self.provenance["plan_sha256"],
                             precision=plan.get('precision','BF16 learned matrices; FP32 interfaces and solver accumulation'),
                             interface_precision='FP32')
        self.observer=None;self.calls=0;self.fallbacks=0;self.fallback_reasons={}
        self.device=next(eager.model.parameters()).device
        if self.device.type != "cuda":raise ValueError("TensorRT CFM model must reside on CUDA")
        self.output=torch.empty(b,80,310,device=self.device,dtype=torch.float32)

    def route_for_signature(self,*args):
        return _acoustic_route(self,args)

    describe_route=route_for_signature

    def __call__(self,x,prompt,lengths,style,mu,mask):
        route=self.route_for_signature(x,prompt,lengths,style,mu,mask)
        if route["kind"] != "tensorrt":
            self.fallbacks+=1
            reason=route["reason"];self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
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
        return dict(backend="TensorRT 11.3 native full two-step CFM Solver",batch=self.batch,frames=310,
                    prompt_frames=258,calls=self.calls,fallbacks=self.fallbacks,
                    fallback_reasons=dict(self.fallback_reasons),identity=dict(self.identity),
                    provenance=deepcopy(self.provenance))


class NativeVocoder113:
    """Static B1/B4/B8, F52 BigVGAN TRT engine including in-engine Quick Plugins."""

    def __init__(self, plan_path, eager):
        import json
        from inspark_infer.ops.tensorrt.vocoder_plugin import register

        plan_file=Path(plan_path).resolve();plan=json.loads(plan_file.read_text())
        if (plan.get("format") != 1 or type(plan.get("batch")) is not int or
                plan["batch"] not in (1, 4, 8) or plan.get("frames") != 52):
            raise ValueError("Expected static TensorRT 11.3 B1/B4/B8, F52 Vocoder plan")
        self.batch=plan["batch"];self.frames=52;self.component="vocoder"
        self.plugins=plan.get("plugins",["Quick Plugins (inventory unspecified)"])
        if not isinstance(self.plugins,list) or any(not isinstance(p,str) for p in self.plugins):
            raise ValueError("TensorRT Vocoder plugins must be a list of names")
        self.plan=str(plan_file)
        self.precision=plan.get('precision','BF16 learned convolutions; FP32 alias-free activation and interfaces')
        engine_path=Path(plan["engine"])
        if not engine_path.is_absolute():engine_path=(plan_file.parent/engine_path).resolve()
        serialized=engine_path.read_bytes();self.engine_sha256=hashlib.sha256(serialized).hexdigest()
        if self.engine_sha256!=plan.get("sha256"):
            raise ValueError("TensorRT 11.3 Vocoder engine hash mismatch")
        self.provenance=_acoustic_provenance_metadata(plan_file,plan,"vocoder",self.engine_sha256)
        self.trt=_import_trt113();register()
        self.runtime=self.trt.Runtime(self.trt.Logger(self.trt.Logger.ERROR))
        self.engine=self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:raise RuntimeError(f"Failed to deserialize {engine_path}")
        b=self.batch;trt=self.trt
        self.input_signature=(("mel",(b,80,52),torch.float32),)
        expected={"mel":((b,80,52),trt.float32,trt.TensorIOMode.INPUT),
                  "pcm":((b,1,13312),trt.float32,trt.TensorIOMode.OUTPUT)}
        _validate_acoustic_io(self.engine,trt,expected)
        self.context=self.engine.create_execution_context();self.eager=eager
        if self.context is None:raise RuntimeError("Failed to create TensorRT Vocoder execution context")
        self.calls=0;self.fallbacks=0;self.fallback_reasons={}
        # The serving entrypoint is usually BigVGAN.forward, a bound method.
        from itertools import chain
        model=getattr(eager,"__self__",eager)
        sample=next(chain(model.parameters(),model.buffers()),None)
        if sample is None or sample.device.type != "cuda":
            raise ValueError("TensorRT Vocoder reference model must reside on CUDA")
        self.device=sample.device
        self.output=torch.empty(b,1,13312,device=self.device,dtype=torch.float32)

    def route_for_signature(self,*args):
        return _acoustic_route(self,args)

    describe_route=route_for_signature

    def __call__(self,mel):
        route=self.route_for_signature(mel)
        if route["kind"] != "tensorrt":
            self.fallbacks+=1
            reason=route["reason"];self.fallback_reasons[reason]=self.fallback_reasons.get(reason,0)+1
            return self.eager(mel)
        for name,address in (("mel",mel.data_ptr()),("pcm",self.output.data_ptr())):
            if not self.context.set_tensor_address(name,address):
                raise RuntimeError(f"TensorRT refused Vocoder binding {name}")
        if not self.context.execute_async_v3(torch.cuda.current_stream(mel.device).cuda_stream):
            raise RuntimeError("TensorRT full Vocoder enqueue failed")
        self.calls+=1;return self.output

    def stats(self):
        return dict(backend="TensorRT 11.3 BigVGAN with in-engine Quick Plugins",
                    batch=self.batch,frames=52,calls=self.calls,fallbacks=self.fallbacks,
                    fallback_reasons=dict(self.fallback_reasons),plan=self.plan,
                    sha256=self.engine_sha256,plugins=list(self.plugins),
                    precision=self.precision,interface_precision='FP32',
                    provenance=deepcopy(self.provenance))
