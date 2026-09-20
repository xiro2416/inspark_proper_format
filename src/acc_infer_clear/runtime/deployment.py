"""One explicit deployment contract; no hidden backend fallbacks or online tuning."""
import json
from pathlib import Path

FIELDS={'schema','status','precision','components','convolutions','rnn_precision',
        'target_graphs','draft_graphs','proposal_graphs','prefix_graphs','head_graphs',
        'slot_draft','overlap_acoustics','fused_acceptance','tail_graphs'}
OPTIONAL={'draft_qkv_fusion','batched_proposal_rng','batched_proposal_rng_min_batch','full_m_plan','target_norm_quant','target_residual_fusion','context_graphs','context_scatter','device_accept_plan','device_residual','device_round_b8'}

def validate(plan):
    extra={'acoustic_kernels','acoustic_plan','head_batch_barrier'} if plan.get('schema') in (2,3,4,5,6,7,8) else set()
    if plan.get('schema') in (3,4,5,6,7,8):extra=extra|{'acoustic_stage2_plan'}
    if plan.get('schema') in (4,5,6,7,8):extra=extra|{'acoustic_pipeline_plan'}
    if plan.get('schema') in (5,6,7,8):extra=extra|{'acoustic_refine_plan'}
    if plan.get('schema') in (6,7,8):extra=extra|{'ar_refine_plan'}
    if plan.get('schema') in (7,8):extra=extra|{'ar_pipeline_plan'}
    if plan.get('schema')==8:extra=extra|{'target_seven_plan'}
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
    if extra:
        if plan['acoustic_kernels'] not in ('off','alias','ntc','both'):raise ValueError('Invalid acoustic kernel mode')
        if not isinstance(plan['head_batch_barrier'],bool):raise ValueError('Expected bool for head_batch_barrier')
        if plan['acoustic_plan'] is not None and not isinstance(plan['acoustic_plan'],str):raise ValueError('Expected acoustic plan path or null')
        if plan['acoustic_kernels'] in ('ntc','both') and not plan['acoustic_plan']:raise ValueError('NTC requires offline shape plan')
    if plan['precision'] not in ('fp32','bf16','fp8','auto'):raise ValueError('Invalid precision')
    if plan['rnn_precision'] not in ('fp32','bf16','fp8','auto'):raise ValueError('Invalid RNN precision')
    if not isinstance(plan['components'],list) or not set(plan['components'])<= {'target','draft','cfm','vocoder'}:raise ValueError('Invalid component list')
    if len(set(plan['components']))!=len(plan['components']):raise ValueError('Duplicate components')
    for key in FIELDS-{'schema','status','precision','components','rnn_precision'}:
        if not isinstance(plan[key],bool):raise ValueError('Expected bool for '+key)
    for key in optional-{'batched_proposal_rng_min_batch','full_m_plan'}:
        if not isinstance(plan[key],bool):raise ValueError('Expected bool for '+key)
    if 'full_m_plan' in optional and (not isinstance(plan['full_m_plan'],str) or not plan['full_m_plan']):raise ValueError('Expected validated full-M plan path')
    if 'batched_proposal_rng_min_batch' in optional and plan['batched_proposal_rng_min_batch'] not in (1,2,3,4,5,6,7,8,16,32):raise ValueError('Invalid batched Proposal RNG threshold')
    if plan['tail_graphs']:raise ValueError('Variable-tail acoustic graphs are not implemented')
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
    if plan.get('prefix_padding_plan') and not Path(plan['prefix_padding_plan']).is_absolute():plan['prefix_padding_plan']=str((path.parent/plan['prefix_padding_plan']).resolve())
    if plan.get('full_m_plan') and not Path(plan['full_m_plan']).is_absolute():plan['full_m_plan']=str((path.parent/plan['full_m_plan']).resolve())
    return plan

def prepare(engine,plan):
    from . import device
    from acc_infer_clear.kernels.planner import DeviceCaps
    plan=validate(plan)
    if engine.sessions or getattr(engine,'deployment_state','raw')!='raw':raise RuntimeError('Deployment can only be prepared once, before admission')
    caps=DeviceCaps.current()
    if caps.sm<80:raise RuntimeError('This implementation requires NVIDIA SM80 or newer')
    resolve=lambda value:('fp8' if caps.native_fp8 else 'bf16') if value=='auto' else value
    result=dict(requested=plan,resolved_precision=resolve(plan['precision']),
                resolved_rnn_precision=resolve(plan['rnn_precision']),sm=caps.sm,
                hardware_validation='RTX6000D SM120 tested; other device models require validation',online_learning=False,tail_graphs=False)
    engine.deployment_state='preparing'
    try:
        if result['resolved_precision']!='fp32':
            result['precision']=engine.prepare_precision(result['resolved_precision'],plan['components'],plan['convolutions'])
        if result['resolved_rnn_precision']!='fp32':result['rnn']=engine.prepare_rnn_precision(result['resolved_rnn_precision'])
        if plan.get('batched_proposal_rng'):
            engine.rt.proposal.batched_rng=True
            engine.rt.proposal.batched_rng_min_batch=plan.get('batched_proposal_rng_min_batch',1)
            result['batched_proposal_rng']=dict(distribution='iid Exponential(1)',request_seed_bitwise=False,direct_graph_input=True,min_batch=engine.rt.proposal.batched_rng_min_batch,online_tuning=False)
        if plan.get('acoustic_kernels','off')!='off':
            result['acoustic_kernels']=engine.prepare_acoustic_kernels(plan.get('acoustic_plan'),plan['acoustic_kernels'])
        if plan.get('acoustic_stage2_plan'):result['acoustic_stage2']=engine.prepare_acoustic_stage2(plan['acoustic_stage2_plan'])
        if plan.get('acoustic_pipeline_plan'):result['acoustic_pipeline']=engine.prepare_acoustic_pipeline(plan['acoustic_pipeline_plan'])
        if plan.get('acoustic_refine_plan'):result['acoustic_refine']=engine.prepare_acoustic_refine(plan['acoustic_refine_plan'])
        if plan.get('ar_refine_plan'):result['ar_refine']=engine.prepare_ar_refine(plan['ar_refine_plan'])
        if plan.get('ar_pipeline_plan'):result['ar_pipeline']=engine.prepare_ar_pipeline(plan['ar_pipeline_plan'])
        if plan.get('target_norm_quant') or plan.get('target_residual_fusion'):
            from acc_infer_clear.target_norm_quant.deploy import attach as attach_norm_quant
            attach_norm_quant(engine,dict(batches=list(range(1,9))+[48,64],division=0,sync_strategy=1,roles=['qkv','up'] if plan.get('target_norm_quant') else [],residual=plan.get('target_residual_fusion',False)))
        if plan.get('target_seven_plan'):result['target_seven']=engine.prepare_target_seven(plan['target_seven_plan'])
        if plan.get('full_m_plan'):
            from acc_infer_clear.target_full_m.deploy import prepare as prepare_full_m
            result['full_m']=prepare_full_m(engine,plan['full_m_plan'])
        engine.head_batch_barrier=plan.get('head_batch_barrier',False)
        result['target']=engine.prepare_slot_target(graphs=plan['target_graphs'])
        if plan['slot_draft']:engine.prepare_slot_draft()
        if plan.get('draft_qkv_fusion'):
            if not plan['slot_draft'] or result['resolved_precision']!='fp8':raise ValueError('Draft QKV fusion requires slot Draft and FP8')
            from acc_infer_clear.draft_fusion.deploy import prepare as prepare_draft_fusion
            result['draft_qkv_fusion']=prepare_draft_fusion(engine)
        if plan.get('full_m_plan'):
            from acc_infer_clear.target_full_m.deploy import prepare_combined
            result['full_m_combined_qkv']=prepare_combined(engine,plan['full_m_plan'])
        if plan.get('context_graphs'):
            from acc_infer_clear.runtime.graph_policy import batches as graph_batches
            selected=graph_batches(engine.config['max_batch']);totals=tuple([b*8 for b in selected])
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['context_graphs']=engine.rt.context.prepare_graphs(
                    engine.config['max_batch'],scatter=plan.get('context_scatter',False),totals=totals)
        if plan['proposal_graphs']:result['proposal']=engine.prepare_proposal_graphs()
        if plan['draft_graphs']:result['draft']=engine.prepare_draft_graphs()
        if plan['prefix_graphs']:result['prefix']=engine.prepare_prefix_graphs()
        if plan['fused_acceptance']:engine.prepare_acceptance_fusion()
        device_available=bool(plan.get('device_round_b8'))
        if plan.get('device_accept_plan') and (not plan.get('device_round_b8') or device_available):
            if not plan['fused_acceptance']:raise ValueError('Device accept plan requires fused acceptance')
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['device_accept_plan']=engine.rt.accept.prepare_device_plan(
                    int(engine.rt.engine.target.gpt.stop_mel_token),engine.config['max_speech_tokens'])
        if plan.get('device_residual') and (not plan.get('device_round_b8') or device_available):
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                result['device_residual']=engine.rt.residual.prepare_device_normal()
        if plan.get('device_round_b8'):
            required=('device_accept_plan','device_residual','context_scatter')
            if not all(plan.get(k) for k in required):raise ValueError('Device round requires '+','.join(required))
            engine.rt.accept.device_plan=False;engine.rt.residual.device_normal=False
            from acc_infer_clear.kernels.device_commit import prepare as prepare_device_commit
            from acc_infer_clear.runtime.graph_policy import batches as graph_batches
            selected=graph_batches(engine.config['max_batch'])
            with engine.torch.cuda.stream(engine.model.stream),engine.torch.inference_mode():
                compiled=prepare_device_commit(engine.rt.device,int(engine.rt.engine.target.gpt.stop_mel_token),engine.config['max_speech_tokens'],selected)
            engine.device_round_batches=set(selected);engine.device_round_b8=True
            result['device_round_b8']=dict(head_only=True,batches=list(selected),kv_limit=128,
                host_boundary='all_ready/fallback scalar per round',compiled=compiled,
                parent_graph=False,online_capture=False)
        if plan['head_graphs']:result['acoustics']=engine.prepare_head_graphs()
        if plan['overlap_acoustics']:
            import torch
            engine.acoustic_stream=torch.cuda.Stream(priority=-1);engine.overlap_acoustics=True
        if plan.get('prefix_padding_plan'):
            from acc_infer_clear.prefix_padding.deploy import prepare as prepare_padding
            result['prefix_padding']=prepare_padding(engine,plan['prefix_padding_plan'])
    except BaseException:
        engine.deployment_state='failed'
        raise
    engine.deployment_state='ready';engine.deployment_manifest=result
    return result
