"""Explicit common-runtime deployment for calibrated FP8 and INT8 candidates."""
from __future__ import annotations
from inspark_infer.runtime.bundle_paths import read_json

import json
import hashlib
from pathlib import Path

import torch
from torch import nn


PATHS = ('calibration', 'target_plan', 'draft_plan', 'cfm_plan', 'vocoder_plan', 'official_sources')
OPTIONAL_PATHS = ('prefill_plan', 'latent_plan', 'tail_target_plan', 'tail_draft_plan','context_plan',
                  'latent_suffix_plan','tail_context_plan','vocoder_partition_plan','condition_trt_plan',
                  'middle_target_plan','middle_draft_plan','middle_context_plan','cfm_microbatch_plan','vocoder_microbatch_plan','vocoder_serial_plan')
OPTIONAL_SETTINGS = ('rnn_graph_rewrite', 'batch_conditions', 'latent_cached_prefix',
                     'latent_gpu_scalar_lengths', 'condition_per_row_lengths',
                     'graph_burst_rounds', 'head_handoff', 'admission_packing',
                     'draft_identity_slots', 'condition_streams', 'batch_text_dedup',
                     'batch_prefill_eos', 'latent_vector_pack', 'draft_head_major_arena',
                     'batch_head_pcm', 'cpu_text_processes', 'latent_reuse_prefill_kv',
                     'cpu_text_workers','late_verify_after','condition_graphs','condition_graph_exact_batches',
                     'condition_flat_projection','prefill_readonly_views','prefill_context_overlap',
                     'static_gc_freeze','prefill_context_views','condition_projected_vq','tail_compact_after','target_kv_fusion','component_calibrations','component_precisions','model_tensor_hashes','hardware')
BACKENDS = ('framework_dspark_adapter', 'native_dspark_worker_trt_compute')


def validate(plan):
    required = {'schema', 'status', 'precision', 'batch', 'runtime_backend', 'graphs', *PATHS}
    if set(plan)-set(OPTIONAL_PATHS)-set(OPTIONAL_SETTINGS) != required or plan['schema'] != 9:
        raise ValueError('Invalid unified deployment fields')
    allowed_batches=((1,2,4,8,16,32,64,128) if plan['precision'] in ('fp8','int8_smoothquant')
                     else (1,4,8,16,32,64,128) if plan['precision']=='nvfp4_fp8' else (1,8,64,128))
    if (plan['precision'] not in ('fp8','int8_smoothquant','nvfp4','nvfp4_fp8') or type(plan['batch']) is not int
            or plan['batch'] not in allowed_batches):
        raise ValueError('Unsupported unified precision/batch deployment')
    if 'component_precisions' in plan:
        p=plan['component_precisions']
        if not isinstance(p,dict) or set(p)!= {'target','draft','cfm','vocoder'} or any(v not in ('fp8','nvfp4','nvfp4_fp8','int8_smoothquant') for v in p.values()):
            raise ValueError('Explicit component precision mapping must be complete')
    if plan['runtime_backend'] not in BACKENDS:
        raise ValueError('This adapter must not be labelled as the native NVIDIA executor')
    if 'hardware' in plan:
        hardware = plan['hardware']
        if (not isinstance(hardware, dict) or set(hardware) != {'gpu_name', 'sm'}
                or not isinstance(hardware['gpu_name'], str) or not hardware['gpu_name']
                or type(hardware['sm']) is not int or hardware['sm'] <= 0):
            raise ValueError('Expected exact target GPU name and SM identity')
    if 'static_gc_freeze' in plan and type(plan['static_gc_freeze']) is not bool:
        raise ValueError('static_gc_freeze must be an explicit bool')
    if 'target_kv_fusion' in plan and type(plan['target_kv_fusion']) is not bool:
        raise ValueError('target_kv_fusion must be an explicit bool')
    if 'cfm_microbatch_plan' in plan and (plan['batch'] != 128 or not plan.get('graphs')
            or not isinstance(plan['cfm_microbatch_plan'], str) or not plan['cfm_microbatch_plan']):
        raise ValueError('CFM microbatch needs an explicit B64 plan and graphed B128 deployment')
    if 'vocoder_serial_plan' in plan and (plan['batch']!=128 or not plan.get('graphs')
            or 'vocoder_partition_plan' in plan or 'vocoder_microbatch_plan' in plan
            or not isinstance(plan['vocoder_serial_plan'],str) or not plan['vocoder_serial_plan']):
        raise ValueError('Native vocoder serial plan needs a complete graphed B128 native deployment')
    if 'vocoder_microbatch_plan' in plan and (plan['batch']!=128 or not plan.get('graphs')
            or plan['runtime_backend']!='native_dspark_worker_trt_compute'
            or not plan.get('vocoder_partition_plan') or not isinstance(plan['vocoder_microbatch_plan'],str)
            or not plan['vocoder_microbatch_plan']):
        raise ValueError('Vocoder microbatch needs a B64 partition and graphed B128 partition deployment')
    if 'prefill_context_views' in plan and (type(plan['prefill_context_views']) is not bool
            or not plan.get('admission_packing',False) or 'prefill_plan' not in plan):
        raise ValueError('Prefill context views require owned prefix source import')
    if 'condition_projected_vq' in plan and (type(plan['condition_projected_vq']) is not bool
            or not plan.get('batch_conditions',False) or plan.get('condition_graphs',False)):
        raise ValueError('Projected semantic lookup requires grouped conditions without condition Graphs')
    if 'condition_trt_plan' in plan and (not plan.get('batch_conditions',False)
            or plan.get('condition_graphs',False) or plan.get('condition_flat_projection',False)):
        raise ValueError('Condition TRT needs grouped conditions without conflicting Graph/projection routes')
    if type(plan['graphs']) is not bool or not all(isinstance(plan[k], str) and plan[k] for k in PATHS):
        raise ValueError('Expected explicit graph flag and artifact paths')
    if any(k in plan for k in ('prefill_plan','latent_plan')) and not all(isinstance(plan.get(k), str) and plan[k] for k in ('prefill_plan','latent_plan')):
        raise ValueError('Provide both prefill and latent plans')
    if 'latent_suffix_plan' in plan and not (plan.get('latent_reuse_prefill_kv') and
            plan.get('graphs') and 'prefill_plan' in plan and isinstance(plan['latent_suffix_plan'],str)):
        raise ValueError('Cached-latent suffix plan requires graphed prefix reuse')
    if 'tail_context_plan' in plan and not all(k in plan for k in ('context_plan','tail_target_plan','tail_draft_plan')):
        raise ValueError('Tail context requires the matching main context and tail AR plans')
    if 'condition_graph_exact_batches' in plan and (type(plan['condition_graph_exact_batches']) is not bool or
                                                  not plan.get('condition_graphs',False)):
        raise ValueError('Exact condition batches require enabled condition Graphs')
    if 'condition_flat_projection' in plan and (type(plan['condition_flat_projection']) is not bool or
                                               plan.get('condition_graphs',False) or not plan.get('batch_conditions',False)):
        raise ValueError('Flat condition projection requires batched conditions without condition Graphs')
    if 'prefill_readonly_views' in plan and (type(plan['prefill_readonly_views']) is not bool or
                                          not plan.get('admission_packing',False) or 'prefill_plan' not in plan):
        raise ValueError('Read-only prefill views require native prefix ownership and admission packing')
    if 'prefill_context_overlap' in plan and (type(plan['prefill_context_overlap']) is not bool or
                                            not plan.get('admission_packing',False) or 'prefill_plan' not in plan):
        raise ValueError('Prefill context overlap requires owned native prefix inputs')
    if 'cpu_text_workers' in plan and (type(plan['cpu_text_workers']) is not int
            or not 1<=plan['cpu_text_workers']<=64 or not plan.get('cpu_text_processes',False)):
        raise ValueError('CPU text worker count requires process backend and1..64 workers')
    if 'late_verify_after' in plan and not (type(plan['late_verify_after']) is int
            and plan['late_verify_after'] in (8,10,12) and plan['graphs']
            and plan['runtime_backend']=='native_dspark_worker_trt_compute'
            and ('tail_target_plan' not in plan or
                 plan.get('batch') in (16,32,128) and type(plan.get('tail_compact_after')) is int
                 and plan['tail_compact_after']>=(13 if plan.get('batch')==128 else 12))):
        raise ValueError('Verify-only rounds require native Graphs and a standalone late threshold8/10/12')
    if any(k in plan for k in ('tail_target_plan','tail_draft_plan')):
        if not (plan['batch'] in (16,32,64,128) and plan['runtime_backend']=='native_dspark_worker_trt_compute'
                and plan['graphs'] and all(isinstance(plan.get(k),str) and plan[k]
                for k in ('tail_target_plan','tail_draft_plan'))):
            raise ValueError('B8 tail requires a graphed native B64 deployment and both AR plans')
    if 'tail_compact_after' in plan and (type(plan['tail_compact_after']) is not int
            or not 12<=plan['tail_compact_after']<=20 or 'tail_target_plan' not in plan):
        raise ValueError('Tail threshold requires matching plans and round12..20')
    if any(k in plan for k in ('middle_target_plan','middle_draft_plan','middle_context_plan')):
        if not (plan['batch']==128 and all(k in plan for k in
                ('middle_target_plan','middle_draft_plan','middle_context_plan','tail_target_plan','tail_draft_plan','tail_context_plan'))
                and plan.get('late_verify_after')==12 and plan.get('tail_compact_after')==13):
            raise ValueError('Middle compaction requires native B128 round10/13 stack with all matching plans')
    if 'context_plan' in plan and (plan['runtime_backend']!='native_dspark_worker_trt_compute'
                                   or not isinstance(plan['context_plan'],str) or not plan['context_plan']):
        raise ValueError('Context engine requires the native worker and an explicit plan')
    if plan.get('rnn_graph_rewrite', 'none') not in ('none', 'factorized', 'compiled'):
        raise ValueError('Unknown RNN graph rewrite')
    if 'batch_conditions' in plan and type(plan['batch_conditions']) is not bool:
        raise ValueError('batch_conditions must be an explicit bool')
    if 'latent_cached_prefix' in plan and type(plan['latent_cached_prefix']) is not bool:
        raise ValueError('latent_cached_prefix must be an explicit bool')
    for name in ('latent_gpu_scalar_lengths','condition_per_row_lengths',
                 'head_handoff','admission_packing','draft_identity_slots','batch_text_dedup',
                 'batch_prefill_eos','latent_vector_pack','draft_head_major_arena','batch_head_pcm',
                 'cpu_text_processes','latent_reuse_prefill_kv','condition_graphs'):
        if name in plan and type(plan[name]) is not bool:
            raise ValueError(name+' must be an explicit bool')
    if type(plan.get('condition_streams', 1)) is not int or plan.get('condition_streams', 1) not in (1, 2, 4):
        raise ValueError('condition_streams must be 1, 2, or 4')
    if 'graph_burst_rounds' in plan and (type(plan['graph_burst_rounds']) is not int
                                         or plan['graph_burst_rounds'] not in (1, 2, 4)):
        raise ValueError('graph_burst_rounds must be 1, 2, or 4')
    if plan.get('draft_head_major_arena',False) and not (
            plan['runtime_backend']=='native_dspark_worker_trt_compute' and plan.get('draft_identity_slots',False)):
        raise ValueError('Head-major arena requires fixed identity slots in the native bridge')
    return dict(plan)


def load(path):
    path = Path(path).resolve()
    plan = validate(read_json(path))
    for key in (*PATHS, *(k for k in OPTIONAL_PATHS if k in plan)):
        value = Path(plan[key])
        plan[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    for component,value in plan.get('component_calibrations',{}).items():
        p=Path(value);plan['component_calibrations'][component]=str(p if p.is_absolute() else (path.parent/p).resolve())
    return plan


def validate_artifact_recipes(plan):
    from inspark_infer.runtime.asset_identity import calibration_digest,validate_component_roles,component_precision
    validate_component_roles(plan)
    digest = hashlib.sha256(Path(plan['calibration']).read_bytes()).hexdigest()
    for component in ('target', 'draft', 'cfm', 'vocoder'):
        artifact = read_json(Path(plan[component+'_plan']))
        if component in ('target', 'draft') and 'quantization_recipe' not in artifact:
            recipe = artifact.get('provenance', {}).get(str(plan['batch']), {}).get('quantization_recipe', {})
            observed = recipe.get('calibration_sha256')
        else:
            recipe = artifact.get('quantization_recipe', {})
            observed = recipe.get('calibration', {}).get('sha256')
        if observed != calibration_digest(plan,component) or recipe.get('scheme') != component_precision(plan,component):
            raise ValueError(f'{component} engine does not match the declared calibration/precision')
    return digest


def install_reference_recipe(engine, artifact):
    """Same-recipe floating reference for prefill/latent/tail and context math.

    These operators simulate explicit Q/DQ; they are not counted as native
    FP8/INT8 kernels. The measured engine stages are separately attested.
    """
    from inspark_infer.quantization.unified import iter_roles
    from inspark_infer.build.unified_acoustic_export import ExportWeightOp, fold_weight_norm_
    if artifact['scheme'] in ('nvfp4','nvfp4_fp8'):
        from inspark_infer.quantization.nvfp4 import install
        return [{k:v for k,v in r.items() if k!='module'} for r in install(engine,artifact)]
    roles = list(iter_roles(engine, artifact['scheme']))
    fold_weight_norm_(engine.student.model)
    manifest = []
    for role in roles:
        spec = artifact['role_specs'][role.path]
        module = role.module
        if type(module).__name__ == 'Conv1D':
            weight = role.weight()
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=module.bias is not None, device=weight.device)
            linear.weight.data.copy_(weight)
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.detach())
            module = linear
        reference = ExportWeightOp(module, spec)
        reference.prepare_reference_weights()
        role.parent.add_module(role.child_name, reference)
        manifest.append(dict(path=role.path, precision=spec['precision'], backend='torch_qdq_reference'))
    return manifest


class FirstChunkController:
    def __init__(self, engine, provider, proposal, capture_graph,
                 runtime_backend='framework_dspark_adapter', graph_burst_rounds=2):
        from inspark_infer.ops.trtllm.runtime import FrameworkRoundRuntime
        constructor = FrameworkRoundRuntime
        if runtime_backend == 'native_dspark_worker_trt_compute':
            from inspark_infer.ops.trtllm.native_runtime import NativeWorkerRoundRuntime
            constructor = NativeWorkerRoundRuntime
        self.engine, self.provider = engine, provider
        b = provider.batch
        initial = torch.zeros(b, 1, device='cuda', dtype=torch.long)
        lengths = torch.ones(b, device='cuda', dtype=torch.int32)
        prefix = torch.full_like(lengths, min(40, provider.capacity-8))
        self.runtime = constructor(
            proposal=proposal, groups=engine.rt.engine.dense_groups,
            draft_provider=provider.draft, target_provider=provider.target,
            context_writer=provider.context_writer, initial_tokens=initial,
            token_lengths=lengths, past_lengths=prefix, draft_lengths=prefix,
            mel_lengths=prefix-1, seeds=list(range(b)),
            eos=engine.rt.engine.target.gpt.stop_mel_token, kv_capacity=provider.capacity,
            max_tokens=engine.config['max_speech_tokens'])
        provider.active = self.runtime.active
        if capture_graph:
            self.runtime.capture(burst_rounds=graph_burst_rounds)
        self.calls = 0
        self.status_reads = 0
        self.status_wait_ms = 0.0
        self.launched_rounds = 0
        self.failed = False
        self.fallback_reason = None
        self.failure_count = 0
        self.last_run = None
        self.total_launched_rounds = 0
        self.total_status_reads = 0
        self.total_status_wait_ms = 0.0
        self.total_draft_enqueues = 0
        self.total_target_enqueues = 0
        self.total_prime_enqueues = 0
        self.total_graph_replays=0

    def supports(self, rows):
        return (0 < len(rows) <= self.provider.batch and
                all(max(r.past_length,r.cache.length)+max(0,31-len(r.codes))+8 <= self.provider.capacity
                    for r in rows))

    def begin(self, rows):
        self.provider.import_rows(rows)
        width = max(len(row.codes) for row in rows)
        padding = self.provider.batch-len(rows)
        initial = torch.full((self.provider.batch, width), self.runtime.eos, device='cuda', dtype=torch.long)
        if getattr(self.engine, 'admission_packing', False):
            from torch.nn import functional as F
            initial[:len(rows)].copy_(torch.stack([
                F.pad(torch.cat(row.codes), (0,width-len(row.codes)), value=self.runtime.eos)
                for row in rows]))
        else:
            for i, row in enumerate(rows):
                initial[i, :len(row.codes)].copy_(torch.cat(row.codes))
        vector = lambda values: torch.tensor(values, device='cuda', dtype=torch.int32)
        values = ([len(r.codes) for r in rows]+[1]*padding,
                  [r.past_length for r in rows]+[0]*padding,
                  [r.cache.length for r in rows]+[0]*padding,
                  [r.mel_length for r in rows]+[0]*padding)
        vectors = (vector(values) if getattr(self.engine, 'admission_packing', False)
                   else [vector(value) for value in values])
        self.runtime.reset(initial_tokens=initial, token_lengths=vectors[0],
                           past_lengths=vectors[1], draft_lengths=vectors[2],
                           mel_lengths=vectors[3],
                           seeds=[r.request['seed'] for r in rows]+[0]*padding)
    def run(self, rows):
        self.begin(rows)
        stats = self.runtime.run()
        self.last_run = stats
        self.total_launched_rounds += stats['launched_rounds']
        self.total_status_reads += stats['status_reads']
        self.total_status_wait_ms += stats['status_wait_ms']
        self.total_draft_enqueues += stats.get('draft_enqueues', stats['launched_rounds'])
        self.total_target_enqueues += stats.get('target_enqueues', stats['launched_rounds'])
        self.total_prime_enqueues += stats.get('prime_enqueues', 0)
        self.total_graph_replays+=stats.get('graph_replays',stats['launched_rounds']//self.runtime.graph_burst
                                           if self.runtime.graph is not None else 0)
        result = self.runtime.result()
        accepted = result['accepted'].cpu().tolist()
        self.status_reads = stats['status_reads']
        self.status_wait_ms = stats['status_wait_ms']
        self.launched_rounds = stats['launched_rounds']
        for i, (row, meta) in enumerate(zip(rows, result['metadata'])):
            count, past, draft_length, rounds, done, last, ready = meta
            if not ready:
                raise RuntimeError('GPU controller returned a row without a first chunk')
            row.codes = (list(result['tokens'][i,:count].split(1))
                         if getattr(self.engine, 'head_handoff', False) else
                         [result['tokens'][i, j:j+1] for j in range(count)])
            row.device_codes_buffer = result['tokens'][i]
            row.device_code_count = count
            row.device_round_final = True
            row.last_token_host = last
            row.accepted = accepted[i][:rounds]
            row.past_length = past
            row.done = bool(done)
            # Head completion immediately releases prefill storage. Subsequent
            # text uses the existing correctly rebuilt prefix, not stale KV.
        self.calls += 1
        self.engine.rt.native_target_steps += stats['launched_rounds']
        self.engine.rt.device_target_steps = getattr(self.engine.rt, 'device_target_steps', 0) + stats['launched_rounds']
        self.engine.rt.backbone.native_full_steps = getattr(self.engine.rt.backbone, 'native_full_steps', 0) + stats.get('draft_enqueues', stats['launched_rounds'])
        return stats


def prepare(engine, plan):
    from inspark_infer.quantization.unified import load_artifact
    from inspark_infer.ops.trtllm.adapter import OfficialRNNProposal
    from inspark_infer.ops.trtllm.official import source_rnn_factory, native_rnn_factory
    from inspark_infer.ops.tensorrt.unified_ar import StaticARProvider
    from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113, NativeVocoder113
    plan = validate(plan)
    validate_artifact_recipes(plan)
    from inspark_infer.runtime.asset_identity import validate_model_identities,calibration_digest,calibration_path,component_precision
    validate_model_identities(engine,plan)
    if engine.sessions or engine.deployment_state != 'raw' or engine.config['max_batch'] != plan['batch']:
        raise ValueError('Prepare an exact-batch fresh engine before admission')
    engine.deployment_state = 'preparing'
    try:
        with torch.cuda.stream(engine.model.stream), torch.inference_mode():
            recipe = load_artifact(plan['calibration'], plan['precision'])
            reference_roles = install_reference_recipe(engine, recipe)
            provider = StaticARProvider(engine.rt, plan['target_plan'], plan['draft_plan'], plan['batch'])
            if 'context_plan' in plan:
                from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
                artifact=read_json(Path(plan['context_plan']))
                context_recipe=artifact.get('quantization_recipe',{})
                if (artifact.get('kind') not in ('context','context_kv') or context_recipe.get('scheme')!=component_precision(plan,'draft')
                        or context_recipe.get('calibration',{}).get('sha256')!=calibration_digest(plan,'draft')):
                    raise ValueError('Context engine kind/calibration/precision differs from deployment')
                if any(hasattr(layer,'rope_inv_freq') for layer in engine.rt.engine.draft.layers):
                    raise ValueError('Context engine does not implement position-dependent RoPE')
                provider.context_engine=StaticEngine(plan['context_plan'],plan['batch'])
            provider.identity_slots = plan.get('draft_identity_slots', False)
            provider.target_kv_fusion=plan.get('target_kv_fusion',False)
            provider.head_major_arena=plan.get('draft_head_major_arena',False)
            provider.late_verify_after=plan.get('late_verify_after',0)
            provider.pooled_import = plan.get('admission_packing', False)
            engine.head_handoff = plan.get('head_handoff', False)
            engine.admission_packing = plan.get('admission_packing', False)
            engine.batch_text_dedup=plan.get('batch_text_dedup',False)
            engine.batch_prefill_eos=plan.get('batch_prefill_eos',False)
            engine.latent_vector_pack=plan.get('latent_vector_pack',False)
            engine.latent_reuse_prefill_kv=plan.get('latent_reuse_prefill_kv',False)
            engine.latent_suffix_plan=plan.get('latent_suffix_plan')
            engine.batch_head_pcm=plan.get('batch_head_pcm',False)
            if plan.get('cpu_text_processes',False):
                from inspark_infer.runtime.text_process_pool import TextProcessPool
                frontend=engine.rt.frontend
                new_pool=TextProcessPool(frontend,plan.get('cpu_text_workers',engine.config['cpu_threads']))
                frontend.pool.shutdown(wait=True)
                frontend.pool=new_pool
            engine.rt.target.prefill_import_sources = engine.admission_packing
            engine.rt.target.prefill_readonly_views=plan.get('prefill_readonly_views',False)
            engine.prefill_context_overlap=plan.get('prefill_context_overlap',False)
            if engine.prefill_context_overlap:engine.prefill_context_stream=torch.cuda.Stream()
            engine.rt.context.prefill_import_sources = engine.admission_packing
            engine.rt.context.prefill_context_views=plan.get('prefill_context_views',False)
            native = plan['runtime_backend'] == 'native_dspark_worker_trt_compute'
            factory = native_rnn_factory() if native else source_rnn_factory(plan['official_sources'])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            proposal = OfficialRNNProposal(engine.rt.engine.draft, factory,
                                           provenance='installed_official_module' if native else
                                           'pinned_official_RNN_source_existing_torch_ops')
            if plan.get('rnn_graph_rewrite', 'none') != 'none':
                proposal = proposal.graph_rewrite(compile_graph=plan['rnn_graph_rewrite']=='compiled')
            engine.unified_first_chunk = FirstChunkController(
                engine, provider, proposal, plan['graphs'], plan['runtime_backend'],
                plan.get('graph_burst_rounds', 2))
            if 'tail_target_plan' in plan:
                from inspark_infer.ops.trtllm.compact_tail import CompactTail
                tail_plan=dict(plan,batch=8,target_plan=plan['tail_target_plan'],draft_plan=plan['tail_draft_plan'])
                # These plans must have the same calibration/precision as B64.
                for component in ('target','draft'):
                    artifact=read_json(Path(tail_plan[component+'_plan']))
                    recipe=artifact.get('quantization_recipe') or artifact.get('provenance',{}).get('8',{}).get('quantization_recipe',{})
                    digest=recipe.get('calibration',{}).get('sha256',recipe.get('calibration_sha256'))
                    if digest!=calibration_digest(plan,component) or recipe.get('scheme')!=component_precision(plan,component):
                        raise ValueError('Tail AR calibration differs from B64')
                tail_provider=StaticARProvider(engine.rt,tail_plan['target_plan'],tail_plan['draft_plan'],8)
                tail_provider.identity_slots=plan.get('draft_identity_slots',False)
                tail_provider.target_kv_fusion=plan.get('target_kv_fusion',False)
                tail_provider.head_major_arena=plan.get('draft_head_major_arena',False)
                if 'tail_context_plan' in plan:
                    tail_context=read_json(Path(plan['tail_context_plan']))
                    main_context=read_json(Path(plan['context_plan']))
                    from inspark_infer.runtime.asset_identity import same_quantization_recipe
                    if (tail_context.get('kind')!='context_kv' or tail_context.get('batch')!=8 or
                            main_context.get('kind')!='context_kv' or tail_context.get('tf32') is not False or
                            not same_quantization_recipe(tail_context.get('quantization_recipe',{}),main_context.get('quantization_recipe',{}))):
                        raise ValueError('Tail context differs in precision/calibration/role policy')
                    tail_provider.context_engine=StaticEngine(plan['tail_context_plan'],8)
                tail_controller=FirstChunkController(engine,tail_provider,proposal,True,plan['runtime_backend'],2)
                engine.unified_first_chunk.runtime.compact_tail=CompactTail(tail_controller.runtime,
                    after=plan.get('tail_compact_after',12))
                if 'middle_target_plan' in plan:
                    engine.unified_first_chunk.runtime.single_compact_tail=engine.unified_first_chunk.runtime.compact_tail
                    middle=StaticARProvider(engine.rt,plan['middle_target_plan'],plan['middle_draft_plan'],64)
                    middle.identity_slots=plan.get('draft_identity_slots',False)
                    middle.target_kv_fusion=plan.get('target_kv_fusion',False)
                    middle.head_major_arena=plan.get('draft_head_major_arena',False)
                    middle.late_verify_after=2
                    middle.context_engine=StaticEngine(plan['middle_context_plan'],64)
                    for backend in (middle.target_engine,middle.draft_engine,middle.context_engine):
                        artifact=backend.plan
                        recipe=artifact.get('quantization_recipe') or artifact.get('provenance',{}).get('64',{}).get('quantization_recipe',{})
                        digest=recipe.get('calibration',{}).get('sha256',recipe.get('calibration_sha256'))
                        component=artifact.get('component','target' if backend is middle.target_engine else 'draft')
                        if digest!=calibration_digest(plan,component) or recipe.get('scheme')!=component_precision(plan,component):
                            raise ValueError('Middle precision/calibration differs from main')
                    middle_controller=FirstChunkController(engine,middle,proposal,True,plan['runtime_backend'],2)
                    middle_controller.runtime.compact_tail=CompactTail(tail_controller.runtime,after=3)
                    middle_tail=CompactTail(middle_controller.runtime,after=10)
                    middle_tail.fallback=engine.unified_first_chunk.runtime.single_compact_tail
                    engine.unified_first_chunk.runtime.compact_tail=middle_tail
            engine.latent_cached_prefix_enabled = plan.get('latent_cached_prefix', True)
            engine.latent_gpu_scalar_lengths = plan.get('latent_gpu_scalar_lengths', False)
            engine.condition_per_row_lengths = plan.get('condition_per_row_lengths', False)
            engine.strict_request_isolation = True
            engine.rng_policy = 'request_owned_framework_dspark'
            engine.student = NativeCFMSolver113(plan['cfm_plan'], engine.student)
            if 'cfm_microbatch_plan' in plan:
                from inspark_infer.ops.tensorrt.cfm_serial_microbatch import CFMSerialMicrobatch
                engine.student = CFMSerialMicrobatch(plan['cfm_microbatch_plan'], engine.student)
            engine.vocoder = NativeVocoder113(plan['vocoder_plan'], engine.vocoder)
            if 'vocoder_partition_plan' in plan:
                from inspark_infer.ops.tensorrt.official_vocoder_segments import OfficialVocoderSegments
                partition=read_json(Path(plan['vocoder_partition_plan']))
                recipe=partition.get('quantization_recipe',{})
                if (partition.get('batch')!=plan['batch'] or recipe.get('scheme')!=component_precision(plan,'vocoder') or
                        recipe.get('calibration',{}).get('sha256')!=calibration_digest(plan,'vocoder')):
                    raise ValueError('Vendor vocoder partition differs in batch/precision/calibration')
                engine.vocoder=OfficialVocoderSegments(plan['vocoder_partition_plan'],fallback=engine.vocoder)
            if 'vocoder_microbatch_plan' in plan:
                from inspark_infer.ops.tensorrt.vocoder_serial_microbatch import VocoderSerialMicrobatch
                engine.vocoder=VocoderSerialMicrobatch(plan['vocoder_microbatch_plan'],engine.vocoder)
            engine.head_batch_barrier = True
            engine.config['batch_conditions'] = plan.get('batch_conditions', False)
            if 'vocoder_serial_plan' in plan:
                from inspark_infer.ops.tensorrt.native_vocoder_microbatch import NativeVocoderSerialMicrobatch
                engine.vocoder=NativeVocoderSerialMicrobatch(plan['vocoder_serial_plan'],engine.vocoder)
            if plan.get('condition_projected_vq',False):
                from inspark_infer.runtime.projected_vq import ProjectedVQTable
                engine.tts.projected_vq_table=ProjectedVQTable(engine.tts.semantic_codec.quantizer)
            engine.condition_stream_count = plan.get('condition_streams', 1)
            engine.condition_flat_projection=plan.get('condition_flat_projection',False)
            engine.condition_streams = [torch.cuda.Stream() for _ in range(engine.condition_stream_count)] if engine.condition_stream_count > 1 else []
            if plan.get('condition_graphs',False):
                from inspark_infer.runtime.condition_graphs import ConditionGraphBank
                engine.condition_graph_bank=ConditionGraphBank(engine.tts,plan['batch'],
                    exact_small_batches=plan.get('condition_graph_exact_batches',False))
            if 'condition_trt_plan' in plan:
                from inspark_infer.ops.tensorrt.condition_group import TRTConditionBank,TRTConditionCoverage
                manifest=read_json(Path(plan['condition_trt_plan']))
                factory=TRTConditionCoverage if manifest.get('kind')=='protected_fp32_condition_bank' else TRTConditionBank
                engine.condition_graph_bank=factory(plan['condition_trt_plan'])
            engine.config['head_graph_batches'] = [plan['batch']]
            if 'prefill_plan' in plan:
                from inspark_infer.ops.tensorrt.unified_prefix import NativePrefixBank
                NativePrefixBank(engine, plan['prefill_plan'], plan['latent_plan'],
                                 calibration_path(plan,'target'), capture_graph=plan['graphs']).install()
            if plan['graphs']:
                engine.prepare_head_graphs()
        engine.deployment_state = 'ready'
        return dict(requested=plan, resolved_precision=plan['precision'],
                    runtime_backend=plan['runtime_backend'], native_trtllm_executor=False,
                    ar_backend='TensorRT explicit Q/DQ', cfm=engine.student.stats(),
                    vocoder=engine.vocoder.stats(), reference_roles=reference_roles,
                    prefix=getattr(engine,'unified_prefix',None).stats() if hasattr(engine,'unified_prefix') else None,
                    prefill_latent_backend='tensorrt' if 'prefill_plan' in plan else 'torch_same_recipe_reference_pending_optimization',
                    context_backend='native_dspark_worker' if native else 'torch_same_recipe_reference_pending_optimization',
                    rnn_graph_rewrite=proposal.stats() if hasattr(proposal,'stats') else {'optimization':'none'},
                    tf32=False,
                    kv_capacity=provider.capacity, custom_math_kernels=bool(plan.get('target_kv_fusion') or plan.get('cfm_microbatch_plan') or getattr(engine.vocoder,'new_gpu_math',False) or plan.get('vocoder_microbatch_plan')))
    except BaseException:
        engine.deployment_state = 'failed'
        raise
