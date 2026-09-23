"""One explicit deployment contract; no hidden backend fallbacks or online tuning."""
import json
from pathlib import Path

FIELDS={'schema','status','precision','components','convolutions','rnn_precision',
        'target_graphs','draft_graphs','proposal_graphs','prefix_graphs','head_graphs',
        'slot_draft','overlap_acoustics','fused_acceptance','tail_graphs'}
OPTIONAL={'draft_qkv_fusion','cublas_bf16_up','batched_proposal_rng',
          'batched_proposal_rng_min_batch','full_m_plan','target_norm_quant',
          'target_residual_fusion','context_graphs','context_scatter',
          'device_accept_plan','device_residual','device_round_b8','device_parent_graph',
          'attention_consumer_layout','context_direct_slot','unified_ar',
          'planner_v2_manifest','planner_v2_apply','cfm_triton_fusions',
          'head_batch_barrier','acoustic_kernels','acoustic_plan','tensorrt_plan',
          'tensorrt113_target_full_plan','tensorrt113_draft_full_plan','tensorrt113_cfm_plan',
          'tensorrt113_vocoder_plan','compute_backend','compile_components',
          'compile_max_signatures','compile_error_fallback','compile_pattern_matcher','strict_request_isolation'}

def validate(plan):
    extra={'acoustic_kernels','acoustic_plan'} if plan.get('schema') in (2,3,4,5,6,7,8) else set()
    if plan.get('schema') in (3,4,5,6,7,8):extra=extra|{'acoustic_stage2_plan'}
    if plan.get('schema') in (4,5,6,7,8):extra=extra|{'acoustic_pipeline_plan'}
    if plan.get('schema') in (5,6,7,8):extra=extra|{'acoustic_refine_plan'}
    if plan.get('schema') in (6,7,8):extra=extra|{'ar_refine_plan'}
    if plan.get('schema') in (7,8):extra=extra|{'ar_pipeline_plan'}
    if plan.get('schema')==8:extra=extra|{'target_seven_plan'}
    if plan.get('schema')==8 and 'prefix_plan' in plan:
        extra=extra|{'prefix_plan'}
        if not isinstance(plan['prefix_plan'],str) or not plan['prefix_plan'] or not plan.get('prefix_graphs'):raise ValueError('Prefix optimization requires an offline plan and prefix_graphs')
    if plan.get('schema')==8 and 'prefix_padding_plan' in plan:
        extra=extra|{'prefix_padding_plan'}
        if not isinstance(plan['prefix_padding_plan'],str) or not plan['prefix_padding_plan'] or not plan.get('prefix_graphs'):raise ValueError('Padding requires an offline plan and prefix graphs')
    optional=set(plan)&OPTIONAL
    if set(plan)!=FIELDS|extra|optional:raise ValueError('Unknown/missing deployment keys: '+str(set(plan)^(FIELDS|extra|optional)))
    if plan['schema'] not in (1,2,3,4,5,6,7,8):raise ValueError('Unknown deployment schema')
    if plan['schema'] in (3,4,5,6,7,8):
        if plan['acoustic_kernels']!='both' or not isinstance(plan['acoustic_stage2_plan'],str) or not plan['acoustic_stage2_plan']:raise ValueError('Stage2 requires both stage1 and an offline plan')
    if plan['schema'] in (4,5,6,7,8) and (not isinstance(plan['acoustic_pipeline_plan'],str) or not plan['acoustic_pipeline_plan']):raise ValueError('Pipeline requires an offline source/device-bound plan')
    if plan['schema'] in (5,6,7,8) and (not isinstance(plan['acoustic_refine_plan'],str) or not plan['acoustic_refine_plan']):raise ValueError('Refinement requires an offline source/device-bound plan')
    if plan['schema'] in (6,7,8) and (not isinstance(plan['ar_refine_plan'],str) or not plan['ar_refine_plan']):raise ValueError('AR refinement requires an offline source/device-bound plan')
    if plan['schema'] in (7,8) and (not isinstance(plan['ar_pipeline_plan'],str) or not plan['ar_pipeline_plan']):raise ValueError('AR pipeline requires an offline source/device-bound plan')
    if plan['schema']==8 and (not isinstance(plan['target_seven_plan'],str) or not plan['target_seven_plan']):raise ValueError('Target seven requires a validated offline plan')
    if extra or 'acoustic_kernels' in optional or 'acoustic_plan' in optional:
        if 'acoustic_kernels' not in plan or 'acoustic_plan' not in plan:
            raise ValueError('Acoustic kernels require explicit acoustic_kernels and acoustic_plan keys')
        if plan['acoustic_kernels'] not in ('off','alias','ntc','both'):raise ValueError('Invalid acoustic kernel mode')
        if plan['acoustic_plan'] is not None and not isinstance(plan['acoustic_plan'],str):raise ValueError('Expected acoustic plan path or null')
        if plan['acoustic_kernels'] in ('ntc','both') and not plan['acoustic_plan']:raise ValueError('NTC requires offline shape plan')
    if 'head_batch_barrier' in optional and not isinstance(plan['head_batch_barrier'],bool):raise ValueError('Expected bool for head_batch_barrier')
    if plan['precision'] not in ('fp32','bf16','fp8','auto'):raise ValueError('Invalid precision')
    if plan['rnn_precision'] not in ('fp32','bf16','fp8','auto'):raise ValueError('Invalid RNN precision')
    if not isinstance(plan['components'],list) or not set(plan['components'])<= {'target','draft','cfm','vocoder'}:raise ValueError('Invalid component list')
    if len(set(plan['components']))!=len(plan['components']):raise ValueError('Duplicate components')
    for key in FIELDS-{'schema','status','precision','components','rnn_precision'}:
        if not isinstance(plan[key],bool):raise ValueError('Expected bool for '+key)
    for key in optional-{'batched_proposal_rng_min_batch','full_m_plan',
                         'planner_v2_manifest','cfm_triton_fusions','acoustic_kernels',
                         'acoustic_plan','tensorrt_plan','tensorrt113_target_full_plan',
                         'tensorrt113_draft_full_plan','tensorrt113_cfm_plan','tensorrt113_vocoder_plan',
                         'compute_backend','compile_components','compile_max_signatures'}:
        if not isinstance(plan[key],bool):raise ValueError('Expected bool for '+key)
    if 'cfm_triton_fusions' in optional:
        parts=plan['cfm_triton_fusions']
        if not isinstance(parts,list) or not parts or not set(parts)<= {'norm','gate','rope'} or len(parts)!=len(set(parts)):
            raise ValueError('cfm_triton_fusions must be a non-empty unique subset of norm/gate/rope')
    if 'full_m_plan' in optional and (not isinstance(plan['full_m_plan'],str) or not plan['full_m_plan']):raise ValueError('Expected validated full-M plan path')
    if 'batched_proposal_rng_min_batch' in optional and plan['batched_proposal_rng_min_batch'] not in (1,2,3,4,5,6,7,8,16,32):raise ValueError('Invalid batched Proposal RNG threshold')
    if 'planner_v2_manifest' in optional and (not isinstance(plan['planner_v2_manifest'],str) or not plan['planner_v2_manifest']):raise ValueError('Expected Planner V2 manifest path')
    if 'tensorrt_plan' in optional and (not isinstance(plan['tensorrt_plan'],str) or not plan['tensorrt_plan']):raise ValueError('Expected TensorRT plan path')
    if 'tensorrt113_target_full_plan' in optional and (not isinstance(plan['tensorrt113_target_full_plan'],str) or not plan['tensorrt113_target_full_plan']):raise ValueError('Expected TensorRT 11.3 Target plan path')
    if 'tensorrt113_draft_full_plan' in optional and (not isinstance(plan['tensorrt113_draft_full_plan'],str) or not plan['tensorrt113_draft_full_plan']):raise ValueError('Expected TensorRT 11.3 Draft plan path')
    if 'tensorrt113_cfm_plan' in optional and (not isinstance(plan['tensorrt113_cfm_plan'],str) or not plan['tensorrt113_cfm_plan']):raise ValueError('Expected TensorRT 11.3 CFM plan path')
    if 'tensorrt113_vocoder_plan' in optional and (not isinstance(plan['tensorrt113_vocoder_plan'],str) or not plan['tensorrt113_vocoder_plan']):raise ValueError('Expected TensorRT 11.3 Vocoder plan path')
    if plan.get('planner_v2_apply') and not plan.get('planner_v2_manifest'):raise ValueError('Planner V2 apply requires a manifest')
    if plan['tail_graphs']:raise ValueError('Variable-tail acoustic graphs are not implemented')
    if plan.get('strict_request_isolation') and (plan.get('batched_proposal_rng') or plan.get('device_round_b8') or plan.get('device_residual')):
        raise ValueError('Strict request isolation forbids shared batch/device RNG and non-legacy residual sampling')
    from inspark_infer.ops.backend import validate_reference_plan
    validate_reference_plan(plan)
    return dict(plan)

def load(path):
    path=Path(path).resolve();plan=validate(json.loads(path.read_text()))
    if plan.get('acoustic_plan') and not Path(plan['acoustic_plan']).is_absolute():plan['acoustic_plan']=str((path.parent/plan['acoustic_plan']).resolve())
    if plan.get('acoustic_stage2_plan') and not Path(plan['acoustic_stage2_plan']).is_absolute():plan['acoustic_stage2_plan']=str((path.parent/plan['acoustic_stage2_plan']).resolve())
    if plan.get('acoustic_pipeline_plan') and not Path(plan['acoustic_pipeline_plan']).is_absolute():plan['acoustic_pipeline_plan']=str((path.parent/plan['acoustic_pipeline_plan']).resolve())
    if plan.get('acoustic_refine_plan') and not Path(plan['acoustic_refine_plan']).is_absolute():plan['acoustic_refine_plan']=str((path.parent/plan['acoustic_refine_plan']).resolve())
    if plan.get('ar_refine_plan') and not Path(plan['ar_refine_plan']).is_absolute():plan['ar_refine_plan']=str((path.parent/plan['ar_refine_plan']).resolve())
    if plan.get('ar_pipeline_plan') and not Path(plan['ar_pipeline_plan']).is_absolute():plan['ar_pipeline_plan']=str((path.parent/plan['ar_pipeline_plan']).resolve())
    if plan.get('target_seven_plan') and not Path(plan['target_seven_plan']).is_absolute():plan['target_seven_plan']=str((path.parent/plan['target_seven_plan']).resolve())
    if plan.get('prefix_plan') and not Path(plan['prefix_plan']).is_absolute():plan['prefix_plan']=str((path.parent/plan['prefix_plan']).resolve())
    if plan.get('prefix_padding_plan') and not Path(plan['prefix_padding_plan']).is_absolute():plan['prefix_padding_plan']=str((path.parent/plan['prefix_padding_plan']).resolve())
    if plan.get('full_m_plan') and not Path(plan['full_m_plan']).is_absolute():plan['full_m_plan']=str((path.parent/plan['full_m_plan']).resolve())
    if plan.get('planner_v2_manifest') and not Path(plan['planner_v2_manifest']).is_absolute():plan['planner_v2_manifest']=str((path.parent/plan['planner_v2_manifest']).resolve())
    if plan.get('tensorrt_plan') and not Path(plan['tensorrt_plan']).is_absolute():plan['tensorrt_plan']=str((path.parent/plan['tensorrt_plan']).resolve())
    if plan.get('tensorrt113_target_full_plan') and not Path(plan['tensorrt113_target_full_plan']).is_absolute():plan['tensorrt113_target_full_plan']=str((path.parent/plan['tensorrt113_target_full_plan']).resolve())
    if plan.get('tensorrt113_draft_full_plan') and not Path(plan['tensorrt113_draft_full_plan']).is_absolute():plan['tensorrt113_draft_full_plan']=str((path.parent/plan['tensorrt113_draft_full_plan']).resolve())
    if plan.get('tensorrt113_cfm_plan') and not Path(plan['tensorrt113_cfm_plan']).is_absolute():plan['tensorrt113_cfm_plan']=str((path.parent/plan['tensorrt113_cfm_plan']).resolve())
    if plan.get('tensorrt113_vocoder_plan') and not Path(plan['tensorrt113_vocoder_plan']).is_absolute():plan['tensorrt113_vocoder_plan']=str((path.parent/plan['tensorrt113_vocoder_plan']).resolve())
    return plan

def prepare(engine,plan):
    from inspark_infer.runtime import device
    from inspark_infer.ops.planning.planner import DeviceCaps
    plan=validate(plan)
    if engine.sessions or getattr(engine,'deployment_state','raw')!='raw':raise RuntimeError('Deployment can only be prepared once, before admission')
    caps=DeviceCaps.current()
    if plan.get('unified_ar') and engine.config['max_batch']>8:raise ValueError('Unified AR is validated only for B1..B8')
    if caps.sm<80:raise RuntimeError('This implementation requires NVIDIA SM80 or newer')
    resolve=lambda value:('fp8' if caps.native_fp8 else 'bf16') if value=='auto' else value
    result=dict(requested=plan,resolved_precision=resolve(plan['precision']),
                resolved_rnn_precision=resolve(plan['rnn_precision']),sm=caps.sm,
                hardware_validation='Profile-specific numerical and performance evidence required; runtime identity is not an audit pass',
                hardware=dict(name=caps.name,sm=caps.sm,sms=caps.sms),online_learning=False,tail_graphs=False)
    engine.deployment_state='preparing'
    try:
        if plan.get('compute_backend') in ('eager','compile'):
            from inspark_infer.ops.backend import prepare_reference
            return prepare_reference(engine,plan,result)
        if plan.get('planner_v2_manifest'):
            from inspark_infer.ops.planning.v2.deploy import prepare as prepare_planner_v2
            result['planner_v2']=prepare_planner_v2(engine,plan['planner_v2_manifest'],apply=plan.get('planner_v2_apply',False))
        if result['resolved_precision']!='fp32':
            result['precision']=engine.prepare_precision(result['resolved_precision'],plan['components'],plan['convolutions'])
        if result['resolved_rnn_precision']!='fp32':result['rnn']=engine.prepare_rnn_precision(result['resolved_rnn_precision'])
        # TensorRT is attached before any project-specific Triton/acoustic fusion.
        # This keeps the TensorRT arm an eager-origin compiler comparison rather
        # than stacking TensorRT on top of the candidate being compared.
        if plan.get('tensorrt_plan'):
            from inspark_infer.ops.tensorrt.deploy import prepare as prepare_tensorrt
            result['tensorrt']=prepare_tensorrt(engine,plan['tensorrt_plan'])
        if plan.get('tensorrt113_cfm_plan'):
            from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                engine.student=NativeCFMSolver113(plan['tensorrt113_cfm_plan'],engine.student)
                result['tensorrt113_cfm']=engine.student.stats()
        if plan.get('cfm_triton_fusions'):
            from inspark_infer.ops.triton.stage2_fusions import install as install_cfm_fusions
            result['cfm_triton_fusions']=install_cfm_fusions(engine.student.model,tuple(plan['cfm_triton_fusions']))
        if plan.get('batched_proposal_rng'):
            engine.rt.proposal.batched_rng=True
            engine.rt.proposal.batched_rng_min_batch=plan.get('batched_proposal_rng_min_batch',1)
            result['batched_proposal_rng']=dict(distribution='iid Exponential(1)',request_seed_bitwise=False,direct_graph_input=True,min_batch=engine.rt.proposal.batched_rng_min_batch,online_tuning=False)
        if plan.get('acoustic_kernels','off')!='off':
            result['acoustic_kernels']=engine.prepare_acoustic_kernels(plan.get('acoustic_plan'),plan['acoustic_kernels'])
        if plan.get('tensorrt113_vocoder_plan'):
            from inspark_infer.ops.tensorrt.native113 import NativeVocoder113
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                engine.vocoder=NativeVocoder113(plan['tensorrt113_vocoder_plan'],engine.vocoder)
                result['tensorrt113_vocoder']=engine.vocoder.stats()
        if plan.get('acoustic_stage2_plan'):result['acoustic_stage2']=engine.prepare_acoustic_stage2(plan['acoustic_stage2_plan'])
        if plan.get('acoustic_pipeline_plan'):result['acoustic_pipeline']=engine.prepare_acoustic_pipeline(plan['acoustic_pipeline_plan'])
        if plan.get('acoustic_refine_plan'):result['acoustic_refine']=engine.prepare_acoustic_refine(plan['acoustic_refine_plan'])
        if plan.get('ar_refine_plan'):result['ar_refine']=engine.prepare_ar_refine(plan['ar_refine_plan'])
        if plan.get('ar_pipeline_plan'):result['ar_pipeline']=engine.prepare_ar_pipeline(plan['ar_pipeline_plan'])
        if plan.get('target_norm_quant') or plan.get('target_residual_fusion'):
            from inspark_infer.ops.target_norm_quant.deploy_experiment import attach as attach_norm_quant
            attach_norm_quant(engine,dict(batches=list(range(1,9))+[48,64],division=0,sync_strategy=1,roles=['qkv','up'] if plan.get('target_norm_quant') else [],residual=plan.get('target_residual_fusion',False)))
        if plan.get('target_seven_plan'):result['target_seven']=engine.prepare_target_seven(plan['target_seven_plan'])
        if plan.get('full_m_plan'):
            from inspark_infer.ops.triton.target_full_m.deploy import prepare as prepare_full_m
            result['full_m']=prepare_full_m(engine,plan['full_m_plan'])
        if plan.get('cublas_bf16_up'):
            if result['resolved_precision']!='fp8':raise ValueError('cuBLAS BF16 Up candidate requires FP8 deployment')
            from inspark_infer.cublas_candidates.deploy import prepare as prepare_cublas_candidates
            result['cublas_bf16_up']=prepare_cublas_candidates(engine)
        engine.head_batch_barrier=plan.get('head_batch_barrier',False)
        engine.attention_consumer_layout=bool(plan.get('attention_consumer_layout',False))
        if plan.get('unified_ar'):
            from inspark_infer.ops.triton.unified_ar.deploy import attach_target
            result['unified_ar']=attach_target(engine)
        result['target']=engine.prepare_slot_target(graphs=plan['target_graphs'])
        if plan.get('tensorrt113_target_full_plan'):
            from inspark_infer.ops.tensorrt.native113 import NativeTargetFullBank113
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                bank=NativeTargetFullBank113(plan['tensorrt113_target_full_plan'],engine.rt.target)
                engine.rt.target.attach_native_full_bank(bank)
                result['tensorrt113_target_full']=bank.prepare_graphs()
        if plan['slot_draft']:engine.prepare_slot_draft()
        if plan.get('tensorrt113_draft_full_plan'):
            from inspark_infer.ops.tensorrt.native113 import NativeDraftFullBank113
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                bank=NativeDraftFullBank113(plan['tensorrt113_draft_full_plan'],engine.rt.backbone)
                engine.rt.backbone.native_full_bank=bank
                result['tensorrt113_draft_full']=bank.prepare_graphs()
        if plan.get('draft_qkv_fusion'):
            if not plan['slot_draft'] or result['resolved_precision']!='fp8':raise ValueError('Draft QKV fusion requires slot Draft and FP8')
            from inspark_infer.ops.triton.draft_fusion.deploy import prepare as prepare_draft_fusion
            result['draft_qkv_fusion']=prepare_draft_fusion(engine)
        if plan.get('full_m_plan'):
            from inspark_infer.ops.triton.target_full_m.deploy import prepare_combined
            result['full_m_combined_qkv']=prepare_combined(engine,plan['full_m_plan'])
        if plan.get('unified_ar'):
            from inspark_infer.ops.triton.unified_ar.deploy import attach_draft
            result['unified_ar_draft']=attach_draft(engine)
        if plan.get('context_graphs'):
            # The validated candidate targets the serving B1..B8 regime. Larger
            # active batches retain the exact eager context path.
            from inspark_infer.runtime.graph_policy import batches as graph_batches
            selected=graph_batches(engine.config['max_batch']);totals=tuple([b*8 for b in selected])
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['context_graphs']=engine.rt.context.prepare_graphs(
                    engine.config['max_batch'],scatter=plan.get('context_scatter',False),totals=totals,
                    direct=plan.get('context_direct_slot',False))
        if plan['proposal_graphs']:result['proposal']=engine.prepare_proposal_graphs()
        if plan['draft_graphs']:result['draft']=engine.prepare_draft_graphs()
        if plan['prefix_graphs']:result['prefix']=engine.prepare_prefix_graphs()
        if plan['fused_acceptance']:engine.prepare_acceptance_fusion()
        device_b8_available=bool(plan.get('device_round_b8'))
        if plan.get('device_accept_plan') and (not plan.get('device_round_b8') or device_b8_available):
            if not plan['fused_acceptance']:raise ValueError('Device accept plan requires fused acceptance')
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['device_accept_plan']=engine.rt.accept.prepare_device_plan(
                    int(engine.rt.engine.target.gpt.stop_mel_token),engine.config['max_speech_tokens'])
        if plan.get('device_residual') and (not plan.get('device_round_b8') or device_b8_available):
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['device_residual']=engine.rt.residual.prepare_device_normal()
        if plan.get('device_round_b8'):
            required=('device_accept_plan','device_residual','context_scatter')
            if not all(plan.get(k) for k in required):raise ValueError('Device B8 round requires '+','.join(required))
            if device_b8_available:
                # DeviceB8Head calls the prepared kernels explicitly. Generic
                # B1/tail/fallback paths retain production acceptance/residual.
                engine.rt.accept.device_plan=False;engine.rt.residual.device_normal=False
                from inspark_infer.ops.triton.device_commit import prepare as prepare_device_commit
                from inspark_infer.runtime.graph_policy import batches as graph_batches
                selected=graph_batches(engine.config['max_batch'])
                with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                    compiled=prepare_device_commit(engine.rt.device,int(engine.rt.engine.target.gpt.stop_mel_token),engine.config['max_speech_tokens'],selected)
                parent=bool(plan.get('device_parent_graph',False))
                if parent:
                    from inspark_infer.runtime.indextts2.device_round import DeviceB8GraphBank
                    engine.device_round_bank=DeviceB8GraphBank(engine.rt,engine.config['max_speech_tokens'])
                engine.device_round_batches=set(selected);engine.device_round_b8=True;result['device_round_b8']=dict(head_only=True,batches=list(selected),kv_limit=128,host_boundary='all_ready scalar per round',compiled=compiled,parent_graph=parent,online_capture=False)
            else:result['device_round_b8']=dict(enabled=False,reason='max_batch_lt_8')
        if plan['head_graphs']:result['acoustics']=engine.prepare_head_graphs()
        if plan['overlap_acoustics']:
            import torch
            engine.acoustic_stream=torch.cuda.Stream(priority=-1);engine.overlap_acoustics=True
        if plan.get('prefix_plan'):
            from inspark_infer.prefix_opt.deploy import prepare as prepare_prefix_opt
            result['prefix_opt']=prepare_prefix_opt(engine,plan['prefix_plan'])
        if plan.get('prefix_padding_plan'):
            from inspark_infer.runtime.prefix_padding.deploy import prepare as prepare_padding
            result['prefix_padding']=prepare_padding(engine,plan['prefix_padding_plan'])
    except BaseException:
        engine.deployment_state='failed'
        raise
    if hasattr(engine,'validate_request_isolation'):
        result['request_isolation']=engine.validate_request_isolation(strict=plan.get('strict_request_isolation',False))
    result['compute_backend']='optimized'
    engine.deployment_state='ready';engine.deployment_manifest=result
    return result
