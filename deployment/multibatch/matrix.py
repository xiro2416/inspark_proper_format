"""Serial resumable B32 compute migration, independent of source records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'deployment/multibatch/history'
BUILD=ROOT/'.venv/bin/python'
RUNTIME=ROOT/'.venv-native/bin/python'
BATCHES=(1,2,4,8,16,64,128)


def manifest_for(batch):
    return BASE/'validation-manifest-128.json' if batch>=64 else ROOT/'deployment/history/validation-manifest.json'


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(batch,label,script,*args,build=False):
    h=BASE/f'b{batch}';h.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,INDEX_HISTORY_DIR=str(h))
    env['ACC_TRT_SITE']=str(ROOT/('.venv' if build else '.venv-native')/'lib/python3.12/site-packages')
    log=h/(label+'.log');started=time.time()
    save(BASE/'active.json',dict(batch=batch,stage=label,log=str(log),started=started))
    print(json.dumps(dict(batch=batch,stage=label,log=str(log))),flush=True)
    with log.open('w') as f:
        subprocess.run([str(BUILD if build else RUNTIME),str(script),*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def benchmark(b,path,label,final=False):
    out=BASE/f'b{b}'/(label+'.json')
    if out.resolve()==Path(path).resolve():raise ValueError('Benchmark output must not overwrite deployment input')
    run(b,label,'benchmarks/benchmark_unified_first_chunk.py','run','--gpu',1,'--batch',b,
        '--manifest',manifest_for(b),'--deployment',path,
        '--config','local_assets/runtime/runtime_fp32_b1.yaml','--out',out,
        '--warmups',5 if final else 3,'--waves',30 if final else 10,
        '--power-seconds',30 if final else 0,'--label',label,'--quant-recipe','int8_smoothquant_alpha1.0')
    result=read(out);c=result['measured_counters']
    if not result['execution_pass'] or any(c[k] for k in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks')):
        raise RuntimeError('Unexpected missing first-chunk compute: '+label)
    if any(c['head_graph_hits'][k]!=(30 if final else 10) for k in ('cfm','vocoder')):
        raise RuntimeError('Missing acoustic Graph coverage: '+label)
    return result


def verified_engine(directory,b):
    path=directory/'model.plan.json'
    if not path.exists():return False
    p=read(path)
    if (p['batch']!=b or p['sm']!=89 or p['sha256']!=digest(directory/'model.engine')
            or p['optimization_level']!=5 or p['tiling_optimization_level']!='full' or p['max_num_tactics']!=2147483646):
        raise RuntimeError('Existing artifact identity or strong-build policy mismatch: '+str(path))
    binding=p['provenance']['onnx_binding']
    onnx_directory=Path(binding['onnx']['path']).parent
    for record in [binding['onnx'],*binding.get('external_data',[])]:
        input_path=Path(record['path'])
        if not input_path.is_absolute():input_path=onnx_directory/input_path
        if digest(input_path)!=record['sha256']:
            raise RuntimeError('Existing engine/ONNX binding mismatch: '+str(path))
    return True


def build_components(b):
    assets=ROOT/f'artifacts/sm89/int8_smoothquant/b{b}'
    for component in ('target','draft','prefill','latent','cfm'):
        directory=assets/('cfm-estimator' if component=='cfm' else component)
        if verified_engine(directory,b):continue
        if not directory.joinpath('model.export.json').exists():
            extra=['--cfm-kind','estimator'] if component=='cfm' else []
            run(b,'export-'+component,'deployment/export.py','--component',component,'--batch',b,*extra,build=True)
        run(b,'build-'+component,'scripts/build_unified_onnx.py','--gpu',1,'--onnx',directory/'model.onnx',
            '--engine',directory/'model.engine','--optimization-level',5,'--tiling','full','--aux-streams',0,build=True)
    directory=assets/'vocoder-implicit-tuned'
    if not verified_engine(directory,b):
        if not (assets/'vocoder-gemm/model.export.json').exists():
            run(b,'export-vocoder-gemm','deployment/export.py','--component','vocoder','--batch',b,'--vocoder-gemm',build=True)
        for stage in ('fir','implicit','tuned'):
            run(b,'rewrite-'+stage,f'deployment/b32/export_{stage}_candidate.py','--batch',b,'--history-dir',BASE/f'b{b}',build=True)
        run(b,'build-migrated-vocoder','scripts/build_unified_onnx.py','--gpu',1,'--onnx',directory/'model.onnx',
            '--engine',directory/'model.engine','--optimization-level',5,'--tiling','full','--aux-streams',0,
            '--small-fir-plugin','--small-fir-layout','all_tiled','--implicit-int8-conv','--migrated-implicit-int8-conv',build=True)
    # Verify complete inherited custom/INT8 coverage from the actual retained graph.
    p=read(directory/'model.plan.json');export=read(directory/'model.export.json')
    fir=export['custom_small_fir']['rewrites'];conv=export['custom_implicit_int8_conv']['rewrites']
    if len(fir)!=109 or len(conv)!=76 or len(export['custom_implicit_int8_conv']['tuned_regions'])!=6:
        raise RuntimeError('Incomplete inherited Vocoder coverage')
    actual=read(directory/'model.inspector.json')['Layers']
    integer_plugins=[v for v in actual if v['Name'].startswith('b32_implicit_conv_')]
    fir_plugins=[v for v in actual if v['Name'].startswith('b32_tiled_fir_')]
    if len(integer_plugins)!=76 or len(fir_plugins)!=109 or any(v['Inputs'][i]['Datatype']!='Int8' for v in integer_plugins for i in (0,1)):
        raise RuntimeError('Actual engine lost inherited custom/signed INT8 coverage')
    evidence=[]
    for c in ('target','draft','prefill','latent','cfm-estimator'):
        d=assets/c;layers=read(d/'model.inspector.json')['Layers']
        integer=[v for v in layers if 'i8i32' in v.get('TacticName','')]
        if not integer:raise RuntimeError('Missing actual INT8: '+c)
        mha=[v['Name'] for v in layers if 'gemm_mha' in v.get('TacticName','') or 'gemm_mha' in v['Name']]
        if c=='cfm-estimator' and len(mha)!=13:raise RuntimeError('Missing inherited fused CFM attention; investigate target compiler coverage')
        evidence.append(dict(component=c,int8_tactics=len(integer),fused_mha=mha,plan=str(d/'model.plan.json')))
    save(BASE/f'b{b}'/'compute-coverage.json',dict(batch=b,components=evidence,vocoder=dict(fir=109,int8_convolutions=76,actual_inspector_fir_plugins=len(fir_plugins),actual_inspector_int8_plugins=len(integer_plugins),both_operands_signed_int8=True,migrated_schedule_regions=6,engine_sha256=p['sha256'])))


def audit(b,path,label):
    h=BASE/f'b{b}'
    for c in ('ar','acoustics'):
        script=f'scripts/audit_unified_{c}.py';capture=h/f'{label}-capture-{c}.pt';out=h/f'{label}-audit-{c}.json'
        run(b,f'{label}-capture-{c}',script,'capture','--gpu',1,'--config','local_assets/runtime/runtime_fp32_b1.yaml',
            '--manifest',manifest_for(b),'--deployment',path,'--output',capture,*(['--waves',1] if c=='acoustics' else []))
        run(b,f'{label}-audit-{c}',script,'audit','--gpu',1,'--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',capture,'--output',out)


def lifecycle(b,path,label):
    h=BASE/f'b{b}';out=h/f'{label}-lifecycle.json'
    run(b,label+'-lifecycle','deployment/validate_lifecycle.py','--batch',b,'--deployment',path,
        '--output',out,'--wav-directory',ROOT/f'outputs/multibatch/b{b}/{label}-lifecycle')
    if not read(out)['passed']:raise RuntimeError('Lifecycle failed')


def compare(b,first,second,label):
    from benchmarks.unified_first_chunk import compare_reports
    result=compare_reports(first,second);save(BASE/f'b{b}'/(label+'.json'),result)
    return result['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms']


def migrate(b):
    h=BASE/f'b{b}';h.mkdir(exist_ok=True)
    if (h/'migration-complete.json').exists():
        selected=read(h/'migration-selected.json')
        for c in ('target','draft','prefill','latent','cfm','vocoder'):
            verified_engine(Path(selected[c+'_plan']).parent,b)
        print(json.dumps(dict(batch=b,stage='previously_validated_migration')),flush=True);return
    build_components(b)
    # Compute first: no inherited application condition/prefix reuse yet.
    source=read(ROOT/'deployment/b32/history/current-best-selected.json')
    source.update(batch=b,status='local_sm89_migration_candidate_not_yet_validated',batch_conditions=False,latent_cached_prefix=False)
    for c,d in [('target','target'),('draft','draft'),('prefill','prefill'),('latent','latent'),('cfm','cfm-estimator'),('vocoder','vocoder-implicit-tuned')]:
        source[c+'_plan']=str(ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/{d}/model.plan.json')
    minimal=h/'minimal-selected.json';save(minimal,source)
    plain=benchmark(b,minimal,'minimal-compute')
    audit(b,minimal,'minimal')
    save(h/'compute-baseline-passed.json',dict(passed=True,batch=b,config=str(minimal),coverage='compute-coverage.json'))
    selected=minimal
    if b>1:
        grouped=h/'grouped-selected.json';save(grouped,dict(source,batch_conditions=True,latent_cached_prefix=True))
        measured=benchmark(b,grouped,'migration-grouped')
        ci=compare(b,plain,measured,'compare-migration-scheduling')
        if measured['summary']['wave_admission_to_last_pcm_ms']['p50']<plain['summary']['wave_admission_to_last_pcm_ms']['p50'] and ci[0]>0:selected=grouped
    current=read(selected);save(h/'migration-selected.json',current)
    lifecycle(b,h/'migration-selected.json','migration')
    if selected!=minimal:audit(b,h/'migration-selected.json','migration')
    final=benchmark(b,h/'migration-selected.json','migration-final',final=True)
    # Original prior deployment is a target-shaped comparison, not B32 source timing.
    if (h/'original-selected.json').exists():
        original=benchmark(b,h/'original-selected.json','delivery-original',final=True)
        compare(b,original,final,'compare-original-migration')
    save(h/'current-best-selected.json',current)
    save(h/'migration-complete.json',dict(status='migration_validated',batch=b,selected=str(h/'migration-selected.json'),benchmark='migration-final.json',next='separately requested optimization',p50_ms=final['summary']['wave_admission_to_last_pcm_ms']['p50']))
    print(json.dumps(read(h/'migration-complete.json')),flush=True)


def optimize(b):
    h=BASE/f'b{b}'
    if not (h/'migration-complete.json').exists():raise RuntimeError('Migration must pass before optimization')
    if (h/'optimization-complete.json').exists():
        selected=read(h/'current-best-selected.json')
        for c in ('target','draft','prefill','latent','cfm','vocoder'):verified_engine(Path(selected[c+'_plan']).parent,b)
        return
    build_components(b)  # CPU identity/coverage checks of already-built migration artifacts
    source=h/'current-best-selected.json';current=read(source)
    capture=h/('migration-capture-acoustics.pt' if (h/'migration-capture-acoustics.pt').exists() else 'minimal-capture-acoustics.pt')
    # Check exposed component cost and actual fusion before choosing remedies.
    for c in ('vocoder','cfm'):
        run(b,'profile-'+c,'deployment/b32/profile_vocoder.py','--batch',b,'--history-dir',h,'--capture',capture,
            '--component',c,'--plan',current[c+'_plan'],'--out',h/f'profile-{c}.json')
    run(b,'large-conv-tile-probe','deployment/b32/probe_large_conv_tiles.py','--batch',b,'--out',h/'large-conv-tile-probe.json')
    probe=read(h/'large-conv-tile-probe.json')
    tiles={d:min((r for r in probe['rows'] if r['dilation']==d),key=lambda r:r['p50_ms']) for d in (1,3,5)}
    run(b,'residual-conv-tile-probe','deployment/multibatch/probe_residual_tiles.py','--batch',b,'--profile',h/'profile-vocoder.json',
        '--export',Path(current['vocoder_plan']).with_name('model.export.json'),'--out',h/'residual-conv-tile-probe.json')
    residual=read(h/'residual-conv-tile-probe.json')
    opportunities=[]
    net_ms=0.
    for d,row in tiles.items():
        old=next(r for r in probe['rows'] if r['dilation']==d and r['tile']==[64,64,64])
        # Two original convolutions per dilation; local estimate, not E2E credit.
        net_ms+=2*max(0.,old['p50_ms']-row['p50_ms'])
    residual_changes=False
    for group in residual['groups']:
        old=next(r for r in group['rows'] if r['tile']==[32,32,64]);best=min(group['rows'],key=lambda r:r['p50_ms'])
        if best['p50_ms']<old['p50_ms']*.98 and best['tile']!=[32,32,64]:
            net_ms+=len(group['indices'])*(old['p50_ms']-best['p50_ms']);residual_changes=True
    source_p50=read(h/'migration-final.json')['summary']['wave_admission_to_last_pcm_ms']['p50']
    opportunities.append(dict(mechanism='six large-convolution schedules',estimated_net_ms=net_ms,estimated_e2e_pct=100*net_ms/source_p50,tiles=tiles))
    if net_ms/source_p50>=0.001 and (residual_changes or any(r['tile']!=[64,64,64] for r in tiles.values())):
        run(b,'export-target-schedule','deployment/multibatch/export_schedule.py','--batch',b,'--probe',h/'large-conv-tile-probe.json','--residual-probe',h/'residual-conv-tile-probe.json',build=True)
        directory=ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/vocoder-target-schedule'
        if not verified_engine(directory,b):
            run(b,'build-target-schedule','scripts/build_unified_onnx.py','--gpu',1,'--onnx',directory/'model.onnx',
                '--engine',directory/'model.engine','--optimization-level',5,'--tiling','full','--aux-streams',0,
                '--small-fir-plugin','--small-fir-layout','all_tiled','--implicit-int8-conv','--target-implicit-int8-schedule',build=True)
        candidate=h/'schedule-candidate-selected.json';save(candidate,dict(current,vocoder_plan=str(directory/'model.plan.json')))
        control=benchmark(b,source,'schedule-control');measured=benchmark(b,candidate,'schedule-candidate')
        ci=compare(b,control,measured,'compare-target-schedule')
        if ci[0]>0 and measured['summary']['wave_admission_to_last_pcm_ms']['p50']<control['summary']['wave_admission_to_last_pcm_ms']['p50']:
            current=read(candidate);save(source,current)
    # Source scheduling rejections depend on batch; recheck actual invocation choices.
    baseline=benchmark(b,source,'runtime-control')
    options=[('burst1',dict(graph_burst_rounds=1)),('burst4',dict(graph_burst_rounds=4)),
        ('packing',dict(admission_packing=True,latent_vector_pack=True))]
    if current.get('batch_conditions'):options.append(('flat-projection',dict(condition_flat_projection=True)))
    elif b>1:
        options.extend([('grouped-prefix',dict(batch_conditions=True,latent_cached_prefix=True)),
            ('flat-grouped-prefix',dict(batch_conditions=True,latent_cached_prefix=True,condition_flat_projection=True))])
    options.append(('native-worker',dict(runtime_backend='native_dspark_worker_trt_compute')))
    rows=[];chosen=None;chosen_p50=baseline['summary']['wave_admission_to_last_pcm_ms']['p50']
    for name,changes in options:
        candidate=h/f'runtime-{name}-selected.json';save(candidate,dict(current,**changes))
        control=baseline
        long_recheck=name in ('grouped-prefix','flat-grouped-prefix')
        if long_recheck:control=benchmark(b,source,'runtime-'+name+'-control',final=True)
        measured=benchmark(b,candidate,'runtime-'+name,final=long_recheck);ci=compare(b,control,measured,'compare-runtime-'+name)
        p50=measured['summary']['wave_admission_to_last_pcm_ms']['p50']
        rows.append(dict(name=name,changes=changes,p50_ms=p50,paired_mean_gain_ci=ci,candidate=str(candidate)))
        save(h/'runtime-options.json',rows)
        if ci[0]>0 and p50<chosen_p50:chosen=candidate;chosen_p50=p50
    if chosen:
        # Confirm promising runtime choice with more samples before selecting it.
        control=benchmark(b,source,'runtime-recheck-control',final=True)
        measured=benchmark(b,chosen,'runtime-recheck-candidate',final=True)
        ci=compare(b,control,measured,'compare-runtime-recheck')
        if ci[0]>0 and measured['summary']['wave_admission_to_last_pcm_ms']['p50']<control['summary']['wave_admission_to_last_pcm_ms']['p50']:
            current=read(chosen);save(source,current)
    lifecycle(b,source,'optimized')
    audit(b,source,'optimized')
    # Frozen input comparison allows compiler/route differences to be reported.
    replay=h/'optimized-replay-acoustics.pt'
    run(b,'optimized-replay-acoustics','scripts/audit_unified_acoustics.py','replay','--gpu',1,
        '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',capture,
        '--cfm-plan',current['cfm_plan'],'--vocoder-plan',current['vocoder_plan'],'--output',replay)
    run(b,'optimized-replay-audit','scripts/audit_unified_acoustics.py','audit','--gpu',1,
        '--config','local_assets/runtime/runtime_fp32_b1.yaml','--capture',replay,'--output',h/'optimized-replay-audit.json')
    baseline_path=h/'original-selected.json' if (h/'original-selected.json').exists() else h/'migration-selected.json'
    original=benchmark(b,baseline_path,'delivery-original-final',final=True)
    final=benchmark(b,source,'delivery-best-final',final=True)
    ci=compare(b,original,final,'compare-delivery')
    # Review retained execution and residuals; diagnostics never substitute for E2E.
    for c in ('vocoder','cfm'):
        run(b,'profile-final-'+c,'deployment/b32/profile_vocoder.py','--batch',b,'--history-dir',h,
            '--capture',h/'optimized-capture-acoustics.pt','--component',c,'--plan',current[c+'_plan'],'--out',h/f'profile-final-{c}.json')
    cfm_layers=read(Path(current['cfm_plan']).with_name('model.inspector.json'))['Layers']
    fusion=[v['Name'] for v in cfm_layers if 'gemm_mha' in v.get('TacticName','') or 'gemm_mha' in v['Name']]
    save(h/'optimization-review.json',dict(opportunities=opportunities,cfm_fused_mha=fusion,
        runtime_probes='runtime-options.json',residual='Original protected floating policy and sequential four-step CFM retained; profile-final component records identify remaining costs. No global-optimal claim.'))
    current['status']='validated_local_sm89_int8';save(source,current)
    destination=ROOT/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json'
    previous=read(destination) if destination.exists() else None
    save(destination,current)
    try:
        wav=ROOT/f'outputs/multibatch/b{b}/cli-smoke.wav'
        run(b,'cli-smoke','deployment/infer.py','--batch',b,'--ref-audio','/workspace/A_TEST_REF/male_news.wav',
            '--text','你好，这是批次引擎的部署验证。','--output',wav)
        import numpy as np
        import soundfile as sf
        pcm,sr=sf.read(wav)
        if sr!=22050 or not len(pcm) or not np.isfinite(pcm).all() or not np.any(pcm):raise RuntimeError('Invalid CLI WAV')
        save(h/'cli-smoke.json',dict(passed=True,batch=b,output=str(wav),sample_rate=sr,seconds=len(pcm)/sr,sha256=digest(wav),scope='Single-request functional smoke; not full-batch throughput'))
    except Exception:
        if previous is not None:save(destination,previous)
        else:destination.unlink(missing_ok=True)
        raise
    def metrics(x):
        s=x['summary']['wave_admission_to_last_pcm_ms'];w=x['sustained_power']
        return dict(p50_ms=s['p50'],p95_ms=s['p95'],mean_power_w=w['power_w']['mean'],peak_power_w=w['power_w']['max'],first_chunk_requests_s=w['requests_per_second'])
    save(h/'optimization-complete.json',dict(status='validated_complete',batch=b,original=metrics(original),best=metrics(final),gain_ci_ms=ci,
        selected=str(source),migration='migration-complete.json',lifecycle='optimized-lifecycle.json',ar_audit='optimized-audit-ar.json',acoustic_audit='optimized-replay-audit.json',review='optimization-review.json'))
    print(json.dumps(read(h/'optimization-complete.json')),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=list(BATCHES));parser.add_argument('--phase',choices=['migrate','optimize','all'],required=True);a=parser.parse_args()
    os.chdir(ROOT)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Only GPU1; source deployment/multibatch/env.sh')
    for phase in (['migrate','optimize'] if a.phase=='all' else [a.phase]):
        for b in a.batches:
            try:(migrate if phase=='migrate' else optimize)(b)
            except Exception as e:
                save(BASE/f'b{b}'/'incomplete.json',dict(status='incomplete',phase=phase,batch=b,error=str(e),active=read(BASE/'active.json')));raise
        save(BASE/(phase+'-matrix-complete.json'),dict(status=phase+'_validated',batches=a.batches))
        save(BASE/'active.json',dict(stage=phase+'_complete',batches=a.batches))


if __name__=='__main__':main()
