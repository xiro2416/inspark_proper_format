"""Review actual retained execution and publish target-specific dispositions."""
from pathlib import Path
from deployment.multibatch.matrix import ROOT,BASE,read,save,run,verified_engine,manifest_for,digest


def review(b):
    h=BASE/f'b{b}';current=read(h/'current-best-selected.json')
    if current!=read(ROOT/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json'):
        raise RuntimeError('Default differs from retained target')
    components=[]
    for name in ('target','draft','prefill','latent','cfm','vocoder'):
        directory=Path(current[name+'_plan']).parent;assert verified_engine(directory,b)
        plan=read(directory/'model.plan.json');layers=read(directory/'model.inspector.json')['Layers']
        components.append(dict(component=name,plan=str(directory/'model.plan.json'),sha256=plan['sha256'],
            integer_tactics=sum('i8i32' in v.get('TacticName','') for v in layers),
            fused_mha=sum('gemm_mha' in v.get('TacticName','') or 'gemm_mha' in v['Name'] for v in layers)))
        if name=='vocoder':
            plugins=[v for v in layers if v.get('LayerType')=='PluginV3']
            fir=[v for v in plugins if v['Name'].startswith('b32_tiled_fir_')]
            fused=[v for v in plugins if v['Name'].startswith('fir_quant_')]
            conv=[v for v in plugins if v['Name'].startswith('b32_implicit_conv_')]
            assert len(fir)+len(fused)==109 and len(conv)==76
            assert all(v['Inputs'][i]['Datatype']=='Int8' for v in conv for i in (0,1))
            assert all(v['Outputs'][0]['Datatype']=='Int8' for v in fused)
            components[-1].update(protected_fp32_fir_paths=len(fir),protected_fp32_fir_terminal_int8_paths=len(fused),signed_int8_convolutions=len(conv))
            profile=read(h/('profile-retained-vocoder.json' if (h/'profile-retained-vocoder.json').exists() else 'profile-final-vocoder.json'))
            if profile['engine_sha256']!=plan['sha256']:
                run(b,'profile-retained-vocoder','deployment/b32/profile_vocoder.py','--batch',b,'--history-dir',h,
                    '--capture',h/'optimized-capture-acoustics.pt','--component','vocoder','--plan',current['vocoder_plan'],'--out',h/'profile-retained-vocoder.json')
                profile=read(h/'profile-retained-vocoder.json')
            costly=sorted(profile['layers'],key=lambda v:v['mean_ms'],reverse=True)[:8]
    assert next(v for v in components if v['component']=='cfm')['fused_mha']==13
    decisions={name:read(h/(name+'-decision.json')) for name in ('schedule-review','runtime-composition','fir-quant')}
    result=read(h/'optimization-complete.json')
    for key in ('lifecycle','ar_audit','acoustic_audit'):
        assert (h/result[key]).exists()
    assert read(h/result['lifecycle'])['passed']
    inventory=dict(batch=b,components=components,target_decisions=decisions,runtime={k:current.get(k) for k in
        ('runtime_backend','graph_burst_rounds','batch_conditions','latent_cached_prefix','admission_packing','latent_vector_pack','condition_flat_projection')},
        manifest=dict(path=str(manifest_for(b)),sha256=digest(manifest_for(b))),
        remaining_costs=[dict(name=v['name'],diagnostic_mean_ms=v['mean_ms'],tactic=v.get('tactic')) for v in costly],
        stopping_reason='Original coverage and strong framework build verified; target tile/rematerialization and independently promising runtime combinations resolved by local and matched E2E evidence. Protected floating policy/four sequential CFM intervals preserved. Full solver compiler failures have unchanged prerequisites. No further exposed mechanism with supported material net benefit remains from this investigation; not a global-optimal guarantee.',
        limits='First-chunk fixed profile, same-recipe Torch tails; generated smoke inputs and short first segments; numeric audits reporting-only, no ASR/MOS; diagnostic timings are not additive application E2E.')
    save(h/'retained-execution-review.json',inventory)
    return inventory
