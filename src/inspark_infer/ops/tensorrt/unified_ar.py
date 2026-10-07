"""Shared static TRT compute provider; all cache/mask work uses torch ops.

The same provider is used for FP8 and INT8. It owns one bounded KV store and
imports prefill once; speculative writes remain hidden behind real lengths.
"""
from __future__ import annotations
from inspark_infer.runtime.bundle_paths import read_json

import hashlib
import json
from pathlib import Path

import torch


class StaticEngine:
    def __init__(self, plan_path, batch):
        from inspark_infer.ops.tensorrt.native113 import _import_trt113
        self.trt = trt = _import_trt113()
        path = Path(plan_path).resolve()
        plan = read_json(path)
        if plan['trt'] != trt.__version__:
            raise ValueError('TensorRT runtime and plan versions differ')
        if 'engines' in plan:
            engine_path = Path(plan['engines'][str(batch)])
            expected_hash = plan['engine_sha256'][str(batch)]
        else:
            if plan.get('batch') != batch:
                raise ValueError('Static engine batch mismatch')
            engine_path = Path(plan['engine'])
            expected_hash = plan['sha256']
        if not engine_path.is_absolute():
            engine_path = path.parent / engine_path
        blob = engine_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected_hash:
            raise ValueError('Engine hash mismatch')
        self.plan_path = str(path)
        self.engine_path = str(engine_path.resolve())
        self.engine_sha256 = expected_hash
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        if any(name in plan.get('plugins',[]) for name in ('inspark_custom::fp8_gated_up', 'inspark_custom::fp8_gated_up_interleaved')):
            from inspark_infer.ops.tensorrt.cfm_gated_up_plugin import register
            register()
        if any(name in plan.get('plugins',[]) for name in ('inspark_custom::small_fir_activation','inspark_custom::small_fir_activation_tiled')):
            from inspark_infer.ops.tensorrt.vocoder_small_fir_plugin import register
            register()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Engine context failed')
        self.device = torch.device('cuda', torch.cuda.current_device())
        dtype = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16,
                 trt.float16: torch.float16, trt.int64: torch.int64,
                 trt.int32: torch.int32, trt.bool: torch.bool}
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if min(shape) <= 0:
                raise ValueError('Static engine shape required')
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = (shape, dtype[self.engine.get_tensor_dtype(name)])
            else:
                self.outputs[name] = torch.empty(shape, device='cuda', dtype=dtype[self.engine.get_tensor_dtype(name)])
        self.plan = plan
        self.calls = 0

    def __call__(self, inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError('Engine inputs differ from declared bindings')
        for name, value in inputs.items():
            shape, dtype = self.inputs[name]
            if (tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous()
                    or value.device != self.device):
                raise ValueError(f'Engine input mismatch: {name}, {value.shape}, {value.dtype}')
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f'Cannot bind {name}')
        for name, value in self.outputs.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f'Cannot bind output {name}')
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT enqueue failed')
        self.calls += 1
        return self.outputs


class StaticARProvider:
    def __init__(self, runtime, target_plan, draft_plan, batch):
        self.runtime = runtime
        self.target_engine = StaticEngine(target_plan, batch)
        self.draft_engine = StaticEngine(draft_plan, batch)
        self.batch = batch
        self.capacity = int(self.target_engine.plan['kv_limit'])
        if self.capacity != self.draft_engine.plan['kv_limit']:
            raise ValueError('Target and Draft KV extents must agree')
        self.target_cache = torch.zeros(24, 2, batch, 20, self.capacity, 64, device='cuda', dtype=torch.bfloat16)
        self.target_append = torch.empty(24, 2, batch, 20, 8, 64, device='cuda', dtype=torch.bfloat16)
        # Bind all 48 engine outputs directly into one slab. One batched cache
        # update replaces per-layer gather/where/scatter launches.
        for layer in range(24):
            for plane, letter in enumerate(('k', 'v')):
                self.target_engine.outputs[f'{letter}_append_{layer}'] = self.target_append[layer, plane]
        self.target_keep = torch.zeros(batch, self.capacity, device='cuda', dtype=torch.bool)
        self.draft_cache = torch.zeros(3, 2, batch, 20, self.capacity, 64, device='cuda', dtype=torch.float32)
        self.active = torch.ones(batch, device='cuda', dtype=torch.bool)
        self.positions = torch.arange(self.capacity, device='cuda')
        self.query7 = torch.arange(7, device='cuda')
        self.query8 = torch.arange(8, device='cuda')
        self.causal = (self.query8[:, None] >= self.query8[None, :])[None, None]

    def import_rows(self, rows):
        if not 0 < len(rows) <= self.batch:
            raise ValueError('Admitted batch exceeds the static profile')
        self.host_draft_lengths = [row.cache.length for row in rows] + [0] * (self.batch-len(rows))
        self.target_cache.zero_()
        self.target_keep.zero_()
        self.draft_cache.zero_()
        for row in rows:
            required = max(row.past_length, row.cache.length) + max(0, 31-len(row.codes)) + 8
            if required > self.capacity:
                raise ValueError(f'Insufficient first-chunk KV capacity: need {required}, profile {self.capacity}')
        self.last_import=import_prefill_rows(self.target_cache, self.target_keep, self.draft_cache, rows,
                            pooled=getattr(self, 'pooled_import', False))

    def draft(self, anchors, absolute_positions, lengths):
        x = self.runtime.engine.draft._noise_embeddings(anchors, absolute_positions)
        valid = self.positions[None, :] < lengths[:, None]
        mask = torch.cat((valid[:, None, None, :].expand(-1, 1, 7, -1),
                          torch.ones(self.batch, 1, 7, 7, device='cuda', dtype=torch.bool)), -1)
        inputs = {'x': x.contiguous(), 'mask': mask}
        for layer in range(3):
            inputs[f'k_cache_{layer}'] = self.draft_cache[layer, 0]
            inputs[f'v_cache_{layer}'] = self.draft_cache[layer, 1]
        result = self.draft_engine(inputs)
        return result['hidden'], result['base']

    def target(self, tokens, absolute_positions, lengths):
        model = self.runtime.engine.target.model
        x = model.embeddings(tokens) + model.text_pos_embedding.emb(absolute_positions)
        valid = (self.positions[None, :] < lengths[:, None]) & self.target_keep
        mask = torch.cat((valid[:, None, None, :].expand(-1, 1, 8, -1),
                          self.causal.expand(self.batch, -1, -1, -1)), -1)
        inputs = {'x': x.contiguous(), 'mask': mask}
        for layer in range(24):
            inputs[f'k_cache_in_{layer}'] = self.target_cache[layer, 0]
            inputs[f'v_cache_in_{layer}'] = self.target_cache[layer, 1]
        result = self.target_engine(inputs)
        positions = lengths[:, None].long() + self.query8[None, :]
        valid_write = (positions < self.capacity) & self.active[:, None]
        indices = positions.clamp(0, self.capacity-1)[:, None, :, None].expand(-1, 20, -1, 64)
        if getattr(self,'target_kv_fusion',False):
            from inspark_infer.ops.triton.target_kv_write import write_target_kv
            write_target_kv(self.target_cache,self.target_append,indices,valid_write)
        else:
            update_target_cache(self.target_cache, self.target_append, indices, valid_write)
        keep_indices = positions.clamp(0, self.capacity-1)
        self.target_keep.scatter_(1, keep_indices, torch.where(valid_write, torch.ones_like(valid_write),
                                                              self.target_keep.gather(1, keep_indices)))
        return result['logits'], result['selected'], result['final']

    def context_writer(self, selected, lengths, counts):
        valid = (self.query8[None, :] < counts[:, None]) & self.active[:, None]
        selected = selected.masked_fill(~valid[..., None], 0)
        model = self.runtime.engine.draft
        context = model.project_context(model.prepare_context(selected))
        positions = lengths[:, None].long() + self.query8[None, :]
        valid = valid & (positions < self.capacity)
        indices = positions.clamp(0, self.capacity-1)[:, None, :, None].expand(-1, 20, -1, 64)
        for layer, block in enumerate(model.layers):
            keys, values = block.context_kv(context, positions)
            for plane, value in enumerate((keys, values)):
                cache = self.draft_cache[layer, plane]
                old = cache.gather(2, indices)
                cache.scatter_(2, indices, torch.where(valid[:, None, :, None], value, old))

    def project_native_context(self,hidden):
        return self.context_engine({'context_hidden':hidden.contiguous()})['context']

    def native_context_kv(self,hidden,positions):
        if getattr(self.context_engine,'plan',{}).get('kind')=='context_kv':
            outputs=self.context_engine({'projected_context':hidden.contiguous()})
            layer=self.runtime.engine.draft.layers[0]
            first_k,first_v=layer.context_kv(hidden[None],positions[None])
            return (torch.cat((first_k[0].transpose(0,1)[:,None],outputs['keys']),1),
                    torch.cat((first_v[0].transpose(0,1)[:,None],outputs['values']),1))
        outputs=self.context_engine.outputs
        if hidden.data_ptr()!=outputs['context'].data_ptr() or hidden.dtype!=outputs['context'].dtype:
            raise ValueError('Context K/V must consume the current projected engine output without precision substitution')
        # NVIDIA's caller masks these tensors in-place. Keep TRT-owned output
        # bindings immutable across replay and give the worker owned temporaries.
        return outputs['keys'].clone(),outputs['values'].clone()


def update_target_cache(cache, append, indices, valid):
    """Single existing-operator triplet over all layers/planes (no new kernel)."""
    indices = indices[None, None].expand(cache.shape[0], 2, -1, -1, -1, -1)
    old = cache.gather(4, indices)
    value = torch.where(valid[None, None, :, None, :, None], append, old)
    cache.scatter_(4, indices, value)


def import_prefill_rows(target_cache, target_keep, draft_cache, rows, *, pooled=False):
    """Batch the one-time prefill import instead of launching 48 copies/row."""
    from torch.nn import functional as F
    actual = len(rows)
    if pooled:
        for row in rows:
            if callable(getattr(row.kv,'check',None)):row.kv.check()
            owner=getattr(row.cache,'pool_owner',None)
            if owner is not None:owner.check(row.cache)
    length = max(r.past_length for r in rows)
    keeps = torch.cat([F.pad(r.mask[:, :r.past_length].bool(), (0, length-r.past_length))
                       for r in rows], 0)
    target_keep[:actual, :length].copy_(keeps)
    target_pool=getattr(rows[0].kv,'pool',None)
    draft_pool=getattr(rows[0].cache,'pool_owner',None)
    target_source=getattr(rows[0].kv,'prefill_source',None)
    draft_source=getattr(rows[0].cache,'prefill_source',None)
    routes=dict(target='request_pack',draft='request_pack')
    valid_target=(pooled and target_source is not None and target_source[1]==getattr(target_source[0],'prefill_import_epoch',None)
                  and all(getattr(r.kv,'prefill_source',None) is not None
                  and r.kv.prefill_source[0] is target_source[0] and r.kv.prefill_source[1]==target_source[1]
                  and r.kv.prefill_source[2] is target_source[2]
                  and r.past_length==r.kv.prefill_source[4] for r in rows))
    if valid_target:
        indices=[r.kv.prefill_source[3] for r in rows]
        packed=target_source[2][:,:,:,:,:length]
        packed=(packed[:,:,:actual] if indices==list(range(actual)) else
                packed.index_select(2,torch.tensor(indices,device=target_cache.device)))
        routes['target']='prefill_source'
    elif pooled and target_pool is not None and all(getattr(r.kv,'pool',None) is target_pool for r in rows):
        for row in rows:row.kv.check()
        slots=torch.tensor([r.kv.slot for r in rows],device=target_cache.device)
        packed=target_pool.storage[:,:,:,:,:length].index_select(2,slots)
        routes['target']='pool_gather'
    elif all(hasattr(r.kv, 'packed') for r in rows):
        packed = torch.cat([r.kv.packed[:, :, :, :, :length] for r in rows], 2)
    else:
        packed = torch.stack([torch.stack([torch.stack((
            F.pad(k[0, :, :r.past_length], (0, 0, 0, length-r.past_length)),
            F.pad(v[0, :, :r.past_length], (0, 0, 0, length-r.past_length)))) for k,v in r.kv])
                              for r in rows], 2)
    # Uninitialized padding in request-owned capacity must never enter attention.
    packed = packed.masked_fill(~keeps[None, None, :, None, :, None], 0)
    target_cache[:, :, :actual, :, :length].copy_(packed)
    lengths = [r.cache.length for r in rows]
    maximum = max(lengths)
    if pooled and draft_pool is not None and all(getattr(r.cache,'pool_owner',None) is draft_pool for r in rows):
        for row in rows:draft_pool.check(row.cache)
        slots=torch.tensor([r.cache.pool_slot for r in rows],device=draft_cache.device)
        packed=draft_pool.storage[:,:,:,:,:maximum].index_select(2,slots)
        valid=torch.arange(maximum,device=draft_cache.device)[None] < torch.tensor(lengths,device=draft_cache.device)[:,None]
        draft_cache[:,:,:actual,:,:maximum].copy_(packed.masked_fill(~valid[None,None,:,None,:,None],0))
        routes['draft']='pool_gather'
        return routes
    offsets, total = [], 0
    for value in lengths:
        offsets.append(total); total += value
    indices = torch.tensor(offsets, device=draft_cache.device)[:, None] + torch.arange(maximum, device=draft_cache.device)[None]
    valid = torch.arange(maximum, device=draft_cache.device)[None] < torch.tensor(lengths, device=draft_cache.device)[:, None]
    if not total:
        return routes
    indices = indices.clamp_max(total-1).flatten()
    for layer in range(draft_cache.shape[0]):
        for plane, name in enumerate(('keys', 'values')):
            valid_source=(pooled and draft_source is not None and
                          draft_source[1]==getattr(draft_source[0],'prefill_import_epoch',None) and
                          all(getattr(r.cache,'prefill_source',None) is not None
                              and r.cache.prefill_source[0] is draft_source[0]
                              and r.cache.prefill_source[1]==draft_source[1]
                              and r.cache.prefill_source[4]==offset
                              for r,offset in zip(rows,offsets)))
            if valid_source:
                joined=draft_source[2+plane][layer][0]
                routes['draft']='prefill_source'
            else:
                joined = torch.cat([getattr(r.cache, name)[layer][:, :, :r.cache.length] for r in rows], 2)[0]
            gathered = joined.index_select(1, indices).reshape(joined.shape[0], actual, maximum, joined.shape[-1]).permute(1,0,2,3)
            draft_cache[layer, plane, :actual, :, :maximum].copy_(gathered.masked_fill(~valid[:,None,:,None], 0))
    return routes
