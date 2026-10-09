"""Serial first-head readiness scheduling with request-owned compact AR banks.

The owner thread runs AR, renders a ready acoustic group, then publishes it.
No concurrent model execution, online capture, or cross-project imports occur.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import types
from pathlib import Path

import torch

from .bundle_paths import read_json
from .asset_identity import calibration_path

AR_BUCKETS = {32: (32, 16, 8), 64: (64, 48, 32, 16, 8),
              128: (128, 64, 48, 32, 16, 8)}


def validate_manifest(manifest):
    if (manifest.get('schema') != 1 or manifest.get('admission_batch') not in AR_BUCKETS
            or manifest.get('acoustic_batch') != 16):
        raise ValueError('Ready C requires schema1, admission32/64/128 and acoustic16')
    batch = manifest['admission_batch']
    if (set(manifest.get('ar_buckets', {})) != {str(b) for b in AR_BUCKETS[batch]}
            or not all(isinstance(p, str) and p for p in manifest['ar_buckets'].values())):
        raise ValueError('Ready C AR bucket inventory is incomplete')
    if not isinstance(manifest.get('acoustic_deployment'), str):
        raise ValueError('Ready C requires an acoustic deployment')
    if manifest.get('late_verify_after', 12) != 12 or manifest.get('graph_burst_rounds', 2) != 2:
        raise ValueError('Ready C requires the validated two-round/late12 schedule')
    return manifest


def component_specs(plan, component):
    artifact = read_json(calibration_path(plan, component))
    specs = {k: v for k, v in artifact['role_specs'].items() if k.startswith(component + '.')}
    if not specs:
        raise ValueError('Missing calibration arithmetic roles for ' + component)
    return specs


def validate_component_identity(candidate, reference, candidate_plan, reference_plan, component):
    """Check actual calibration roles and their binding to the engine manifest.

    Composite artifact location/global scheme may differ. Arithmetic specs,
    checkpoint sources and the digest declared by each engine may not.
    """
    def sources(plan):
        rows = plan.get('provenance', {}).get('model_sources', [])
        if not rows:
            raise ValueError('Missing checkpoint provenance for ' + component)
        return sorted((r['role'], r['sha256']) for r in rows)
    if sources(candidate) != sources(reference):
        raise ValueError(component + ' checkpoint sources differ')
    arithmetic = []
    for metadata, deployment in ((candidate, candidate_plan), (reference, reference_plan)):
        specs = component_specs(deployment, component)
        recipe = metadata.get('quantization_recipe', {})
        path = Path(calibration_path(deployment, component))
        if (recipe.get('calibration', {}).get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest()
                or recipe.get('role_specs_sha256') != hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()):
            raise ValueError(component + ' engine calibration/role hash does not match actual data')
        arithmetic.append(specs)
    if arithmetic[0] != arithmetic[1]:
        raise ValueError(component + ' calibrated arithmetic differs')
    for key in ('trt', 'sm', 'gpu_name'):
        if candidate.get(key) != reference.get(key):
            raise ValueError(component + ' engine runtime/hardware differs')


def transfer_native(source, destination, indices):
    from inspark_infer.ops.trtllm.compact_tail import state_pairs, transfer_state
    if source is destination:
        raise ValueError('Cannot compact into itself')
    if (indices.ndim != 1 or indices.dtype != torch.long
            or not 0 < indices.numel() <= destination.batch):
        raise ValueError('Invalid compaction indices')
    if indices.device.type == 'cpu':
        values = indices.tolist()
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= source.batch:
            raise ValueError('Invalid compaction request mapping')
    # Check the complete state inventory before copying anything.
    pairs = list(zip(state_pairs(source), state_pairs(destination)))
    for (src, axis), (dst, target_axis) in pairs:
        if (axis != target_axis or src.ndim != dst.ndim or src.dtype != dst.dtype
                or src.device != dst.device or src.data_ptr() == dst.data_ptr()
                or src.shape[:axis] + src.shape[axis+1:] != dst.shape[:axis] + dst.shape[axis+1:]):
            raise ValueError('State geometry/ownership mismatch')
    transfer_state(source, destination, indices)
    destination.failures.copy_(source.failures)
    destination.capacity_failures.copy_(source.capacity_failures)


def advance_native(runtime, logical_round):
    late = runtime.verify_graph is not None and logical_round >= runtime.late_verify_after
    (runtime.verify_graph if late else runtime.graph).replay()
    start = time.perf_counter()
    observation = torch.cat((runtime.status.reshape(1), runtime.ready.int())).cpu().tolist()
    if observation[0] & 6:
        raise RuntimeError('Native ready pipeline failed status=' + str(observation[0]))
    return dict(ready=observation[1:], status=observation[0],
                launched_rounds=1 if late else runtime.graph_burst, deferred=late,
                status_wait_ms=1000 * (time.perf_counter() - start))


def eager_reference(wrapper):
    """Unwrap readonly references without assuming the B128 wrapper has .eager."""
    seen = set()
    while id(wrapper) not in seen:
        seen.add(id(wrapper))
        if hasattr(wrapper, 'eager'):
            return wrapper.eager
        fallback = getattr(wrapper, 'fallback', None)
        if fallback is None:
            return wrapper
        wrapper = fallback
    raise ValueError('Circular acoustic fallback wrapper')


class ReadyPipeline:
    def __init__(self, engine, manifest, mode='C', acoustic_priority=0):
        from inspark_infer.ops.tensorrt.unified_ar import StaticARProvider, StaticEngine
        from .unified_deployment import FirstChunkController
        from inspark_infer.ops.tensorrt.unified_prefix import NativePrefixBank
        from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113, NativeVocoder113
        from .graphs import HeadGraphs
        validate_manifest(manifest)
        if mode not in ('C', 'A', 'barrier16'):
            raise ValueError('Only serial C/A/barrier16 modes are supported')
        self.engine, self.manifest, self.mode = engine, dict(manifest), mode
        self.batch = manifest['admission_batch']; self.acoustic_batch = 16
        self.closed = False; self.waves = []
        self.controllers = {self.batch: engine.unified_first_chunk}
        root = engine.unified_first_chunk
        if (engine.config['max_batch'] != self.batch or engine.sessions
                or root.runtime.backend != 'native_dspark_worker_trt_compute'):
            raise ValueError('Requires a prepared native first-head engine before admission')
        self.stream = torch.cuda.Stream(priority=acoustic_priority)
        plans = {int(b): read_json(path) for b, path in manifest['ar_buckets'].items()}
        root_plan = getattr(engine, 'ready_scheduler_root_plan', plans[self.batch])
        acoustic_plan = read_json(manifest['acoustic_deployment'])
        if acoustic_plan.get('batch') != 16 or acoustic_plan.get('model_tensor_hashes') != root_plan.get('model_tensor_hashes'):
            raise ValueError('Ready acoustic profile/model differs')
        for b, plan in plans.items():
            if plan.get('batch') != b or plan.get('model_tensor_hashes') != root_plan.get('model_tensor_hashes'):
                raise ValueError('AR bucket shape/model differs')
            for component in ('target', 'draft'):
                validate_component_identity(read_json(plan[component + '_plan']),
                    getattr(root.provider, component + '_engine').plan, plan, root_plan, component)
            if hasattr(root.provider, 'context_engine'):
                validate_component_identity(read_json(plan['context_plan']), root.provider.context_engine.plan,
                                            plan, root_plan, 'draft')
        for component in ('target', 'cfm', 'vocoder'):
            reference = read_json(root_plan['prefill_plan'] if component == 'target' else root_plan[component + '_plan'])
            candidate = read_json(acoustic_plan['prefill_plan'] if component == 'target' else acoustic_plan[component + '_plan'])
            validate_component_identity(candidate, reference, acoustic_plan, root_plan, component)
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            root.provider.late_verify_after = 12
            root.runtime.late_verify_after = 12
            if root.runtime.graph_burst != 2 or root.runtime.verify_graph is None:
                raise ValueError('Root AR graph must already support burst2/late12')
            for b in sorted(plans, reverse=True):
                if b == self.batch:
                    continue
                p = plans[b]
                provider = StaticARProvider(engine.rt, p['target_plan'], p['draft_plan'], b)
                provider.identity_slots = True; provider.head_major_arena = True
                provider.target_kv_fusion = bool(getattr(root.provider, 'target_kv_fusion', False))
                provider.late_verify_after = 12
                provider.context_engine = StaticEngine(p['context_plan'], b)
                self.controllers[b] = FirstChunkController(engine, provider, root.runtime.proposal, True,
                                                          'native_dspark_worker_trt_compute', 2)
        self.stream.wait_stream(engine.model.stream)
        with torch.cuda.stream(self.stream), torch.inference_mode():
            v = self.acoustic = copy.copy(engine)
            v.sessions = {}; v.head_ready_pipeline = None
            v.config = dict(engine.config, max_batch=16, head_graph_batches=[16])
            v.rt = copy.copy(engine.rt); v.rt.target = copy.copy(engine.rt.target)
            v.rt.latent = copy.copy(engine.rt.latent); v.rt.events = []
            v.latent_suffix_plan = acoustic_plan['latent_suffix_plan']; v.latent_reuse_prefill_kv = True
            bank = NativePrefixBank(v, acoustic_plan['prefill_plan'], acoustic_plan['latent_plan'],
                                    calibration_path(acoustic_plan, 'target'), capture_graph=True)
            v.unified_prefix = bank; v.prefix_graphs = bank; v.rt.latent.body = bank.latent
            self.prefix_reuse = bank.reused_latents
            def reused(_bank, prefixes, codes, lengths):
                return self.reused_latents(prefixes, codes, lengths)
            bank.reused_latents = types.MethodType(reused, bank)
            v.student = NativeCFMSolver113(acoustic_plan['cfm_plan'], eager_reference(engine.student))
            v.vocoder = NativeVocoder113(acoustic_plan['vocoder_plan'], eager_reference(engine.vocoder))
            v.head_graphs = HeadGraphs(); v.head_graphs.prepare(v)
            # Condition streams and any captured condition workspace belong to
            # this acoustic view. Do not borrow mutable root graph scratch.
            v.condition_streams = [torch.cuda.Stream() for _ in getattr(engine, 'condition_streams', ())]
            if getattr(engine, 'condition_graph_bank', None) is not None:
                from .condition_graphs import ConditionGraphBank
                v.condition_graph_bank = ConditionGraphBank(v.tts, 16,
                    exact_small_batches=bool(root_plan.get('condition_graph_exact_batches', False)))
            v.output_d2h_stream = None; v.stages = []; v.failures = []; v.profile_spans = []
        engine.model.stream.wait_stream(self.stream)

    def reused_latents(self, prefixes, codes, lengths):
        parent = self.engine.unified_prefix; bank = self.acoustic.unified_prefix
        if not parent.reuse_prefill_valid:
            raise RuntimeError('Original prefix no longer current')
        original = {key: i for i, key in enumerate(parent.reuse_rows)}
        keys = [(x.data_ptr(), tuple(x.shape), x.dtype, x.device) for x in prefixes]
        if not 0 < len(keys) <= self.acoustic_batch or any(k not in original for k in keys):
            raise RuntimeError('Unknown prefix owner or invalid acoustic group')
        index = torch.tensor([original[k] for k in keys], device=prefixes[0].device, dtype=torch.long)
        n = len(keys); src = parent.backends['prefill']; dst = bank.backends['prefill']
        dst.inputs['keep'].zero_(); dst.inputs['keep'][:n].copy_(src.inputs['keep'].index_select(0, index))
        for name, axis in [('packed_kv', 2), ('final', 0)]:
            dst.outputs[name].zero_()
            dst.outputs[name].narrow(axis, 0, n).copy_(src.outputs[name].index_select(axis, index))
        pad = self.acoustic_batch - n
        padded = list(prefixes) + [prefixes[0].new_zeros(prefixes[0].shape) for _ in range(pad)]
        padded_codes = list(codes) + [codes[0].new_full((1, 1), self.acoustic.tts.gpt.stop_mel_token) for _ in range(pad)]
        padded_lengths = list(lengths) + [1] * pad
        bank.reuse_rows = [(x.data_ptr(), tuple(x.shape), x.dtype, x.device) for x in padded]
        bank.reuse_prefill_valid = True
        output = self.prefix_reuse(padded, padded_codes, padded_lengths)
        if output is None:
            raise RuntimeError('Partial latent reuse rejected')
        return output[:n]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            from .cleanup import cleanup_all
            acoustic = getattr(self, 'acoustic', None)
            streams = [('ready acoustic', self.stream)]
            streams.extend(('ready condition ' + str(i), stream)
                           for i, stream in enumerate(getattr(acoustic, 'condition_streams', ())))
            output = getattr(acoustic, 'output_d2h_stream', None)
            if output is not None:
                streams.append(('ready output', output))
            cleanup_all([(name, stream.synchronize) for name, stream in streams])
        finally:
            # Only private banks/storage are discarded; never close the copied
            # Engine/Runtime/Model, which retain shared readonly model handles.
            acoustic = getattr(self, 'acoustic', None)
            if acoustic is not None:
                bank = getattr(acoustic, 'unified_prefix', None)
                if bank is not None:
                    bank.reused_latents = None; bank.engine = None
                acoustic.unified_prefix = None; acoustic.prefix_graphs = None
                acoustic.head_graphs = None; acoustic.student = None; acoustic.vocoder = None
                acoustic.condition_graph_bank = None; acoustic.condition_streams = []
                acoustic.output_d2h_stream = None
                acoustic.rt = None
            self.prefix_reuse = None; self.acoustic = None; self.controllers.clear()

    def render(self, rows, owner, dependencies):
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for event in dependencies:
                self.stream.wait_event(event)
            self.acoustic.acoustic_rows(rows, owner, 0)
            joined = torch.cuda.Event(); joined.record(self.stream)
        self.engine.model.stream.wait_event(joined)
        return rows

    def run(self, rows, owner, on_chunk):
        if self.closed:
            raise RuntimeError('Pipeline closed')
        e = self.engine; c = self.controllers[self.batch]
        if (not c.supports(rows) or any(e.model.bank.get(owner[id(r)]['case']['voice_id'])['values']
                                       ['voice.cache_mel'].shape[-1] != 258 for r in rows)):
            raise ValueError('Outside first-head profile')
        c.begin(rows); mapping = list(range(len(rows))); snapshots = set(); queue = []; events = []
        launched = reads = row_rounds = replays = 0; wait_ms = 0.; prime = 1
        groups = []; moves = []; enqueues = {}
        metadata_reads = metadata_rows = metadata_d2h_bytes = status_d2h_bytes = 0
        def publish(done):
            for row in done:
                session = owner[id(row)]
                if session['error']:
                    raise RuntimeError(session['error'])
                e._finish_or_drain(session); e._release_row(row); session.pop('_row', None)
                item = dict(request_id=session['case']['id'], chunk=session['chunks'][-1], complete=session['complete'])
                events.append(item)
                if on_chunk:
                    on_chunk(item)
        def dispatch(all_ready):
            if self.mode == 'barrier16' and not all_ready:
                return
            while queue and (len(queue) >= self.acoustic_batch or all_ready):
                take = queue[:self.acoustic_batch]; del queue[:len(take)]
                batch = [rows[i] for i in take]
                deps = list({id(row.head_ready_event): row.head_ready_event for row in batch}.values())
                groups.append(dict(indices=take, round=launched, rows=len(take)))
                publish(self.render(batch, owner, deps))
        try:
            while launched < 64:
                observation = advance_native(c.runtime, launched)
                launched += observation['launched_rounds']; reads += 1; replays += 1
                status_d2h_bytes += (c.runtime.batch + 1) * 4
                wait_ms += observation['status_wait_ms']; row_rounds += c.runtime.batch * observation['launched_rounds']
                enqueues[str(c.runtime.batch)] = enqueues.get(str(c.runtime.batch), 0) + observation['launched_rounds']
                new = [i for i, orig in enumerate(mapping) if observation['ready'][i] and orig not in snapshots]
                if new:
                    runtime = c.runtime
                    index = torch.tensor(new, device=runtime.tokens.device, dtype=torch.long)
                    metadata = torch.stack((runtime.token_lengths, runtime.past, runtime.draft_lengths,
                        runtime.rounds, runtime.done.int(), runtime.last.int(), runtime.ready.int()), 1)
                    packed_device = torch.cat((metadata.index_select(0, index).long(),
                                        runtime.accepted.index_select(0, index).long()), 1)
                    metadata_reads += 1; metadata_rows += len(new)
                    metadata_d2h_bytes += packed_device.numel() * packed_device.element_size()
                    packed = packed_device.cpu().tolist()
                    for slot, values in zip(new, packed):
                        count, past, draft_length, n, done, last, ready = values[:7]
                        if not ready:
                            raise RuntimeError('Snapshot not ready')
                        original = mapping[slot]; row = rows[original]
                        tokens = runtime.tokens[slot].clone()
                        row.codes = list(tokens[:count].split(1)); row.device_codes_buffer = tokens
                        row.device_code_count = count; row.device_round_final = True
                        row.last_token_host = last; row.accepted = values[7:7+n]
                        row.past_length = past; row.done = bool(done); owner[id(row)]['rounds'] += n
                        snapshots.add(original); queue.append(original)
                    event = torch.cuda.Event(); event.record(e.model.stream)
                    for slot in new:
                        rows[mapping[slot]].head_ready_event = event
                all_ready = len(snapshots) == len(rows); dispatch(all_ready)
                if all_ready:
                    break
                if self.mode == 'C':
                    live = [i for i in range(len(mapping)) if not observation['ready'][i]]
                    b = next(b for b in sorted(self.controllers) if b >= len(live))
                    if b < c.runtime.batch:
                        dest = self.controllers[b]
                        index = torch.tensor(live, device=c.runtime.tokens.device, dtype=torch.long)
                        transfer_native(c.runtime, dest.runtime, index)
                        moves.append(dict(from_batch=c.runtime.batch, to_batch=b, active_rows=len(live)))
                        mapping = [mapping[i] for i in live]; c = dest
                if observation['deferred']:
                    c.runtime.prime_graph.replay(); prime += 1
            else:
                raise RuntimeError('Ready round bound exceeded')
            dispatch(True)
            if len(events) != len(rows) or len({x['request_id'] for x in events}) != len(rows):
                raise RuntimeError('Duplicate/missing PCM')
        finally:
            self.stream.synchronize()
        e.device_round_attempts += 1; e.device_round_successes += 1
        e.device_round_status_reads += reads; e.device_round_status_wait_ms += wait_ms
        e.device_round_launched_rounds += launched; e.rt.native_target_steps += launched
        e.rt.device_target_steps = getattr(e.rt, 'device_target_steps', 0) + launched
        e.rt.backbone.native_full_steps = getattr(e.rt.backbone, 'native_full_steps', 0) + launched + prime - 1
        root = self.controllers[self.batch]; root.calls += 1
        root.total_launched_rounds += launched; root.total_status_reads += reads
        root.total_status_wait_ms += wait_ms; root.total_graph_replays += replays
        root.total_target_enqueues += launched; root.total_draft_enqueues += launched + prime - 1
        root.total_prime_enqueues += prime
        root.last_run = dict(launched_rounds=launched, status_reads=reads, status_wait_ms=wait_ms,
            graph_replays=replays, target_enqueues_by_batch=enqueues, native_executor=False)
        self.waves.append(dict(mode=self.mode, rounds=launched, row_rounds=row_rounds, ar_batches=enqueues,
            compactions=moves, acoustic_groups=groups, status_reads=reads, status_wait_ms=wait_ms,
            status_d2h_bytes=status_d2h_bytes, metadata_reads=metadata_reads,
            metadata_rows=metadata_rows, metadata_d2h_bytes=metadata_d2h_bytes))
        return events
