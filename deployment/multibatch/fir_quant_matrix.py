"""Follow actual remaining materialization evidence, after initial target optimization."""
import argparse,json
from pathlib import Path
from deployment.multibatch.matrix import ROOT,BASE,BATCHES,read,save,run,benchmark,compare,lifecycle,digest,verified_engine


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=[128,64,16,8,4,2,1]);a=p.parse_args()
    results=[]
    for b in a.batches:
        h=BASE/f'b{b}'
        if not (h/'optimization-complete.json').exists():raise RuntimeError('Initial optimization/validation required before residual fusion')
        if (h/'fir-quant-decision.json').exists():results.append(read(h/'fir-quant-decision.json'));continue
        from deployment.multibatch.review_schedule import review as review_schedule
        review_schedule(b)
        from deployment.multibatch.compose_runtime import review
        review(b)
        baseline=h/'before-fir-quant-selected.json'
        current=read(baseline if baseline.exists() else h/'current-best-selected.json');save(baseline,current)
        directory=ROOT/f'artifacts/sm89/int8_smoothquant/b{b}/vocoder-fir-quant'
        run(b,'fir-quant-export','deployment/multibatch/export_fir_quant.py','--source-plan',current['vocoder_plan'],'--out-dir',directory,build=True)
        run(b,'fir-quant-local-probe','deployment/multibatch/probe_fir_quant.py','--onnx',directory/'model.onnx','--out',h/'fir-quant-local-probe.json')
        probe=read(h/'fir-quant-local-probe.json');export=read(directory/'model.export.json')
        import onnx
        model=onnx.load(directory/'model.onnx',load_external_data=False)
        shapes={v.name:tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in model.graph.value_info}
        gain_by_shape={tuple(r['shape']):r['current_ms']-r['fused_ms'] for r in probe['rows']}
        net=sum(gain_by_shape[shapes[r['input']]] for r in export['custom_fir_quant']['paths'])
        best=read(h/'optimization-complete.json')['best'];estimated_pct=100*net/best['p50_ms']
        decision=dict(batch=b,status='not_retained',eligible_paths=probe['eligible_paths'],estimated_local_net_ms=net,estimated_e2e_pct=estimated_pct,
            estimate_scope='Sum of complete local paired-Graph savings for exclusive independent FIR outputs; not an E2E gain or proof of native-div identity',local_probe='fir-quant-local-probe.json')
        if estimated_pct<.1:
            decision['reason']='Local exposed materialization potential below0.1%; no whole-engine build justified'
            save(h/'fir-quant-decision.json',decision);results.append(decision);continue
        source_plan=read(current['vocoder_plan']);plugins=source_plan['plugins']
        flags=['--small-fir-plugin','--small-fir-layout','all_tiled','--implicit-int8-conv','--fir-int8-quant-plugin']
        if 'inspark_custom::implicit_int8_conv_1d_schedule' in plugins:flags.append('--target-implicit-int8-schedule')
        elif 'inspark_custom::implicit_int8_conv_1d_migrated' in plugins:flags.append('--migrated-implicit-int8-conv')
        else:raise RuntimeError('Unknown source schedule inventory')
        if not verified_engine(directory,b):
            run(b,'fir-quant-build','scripts/build_unified_onnx.py','--gpu',1,'--onnx',directory/'model.onnx','--engine',directory/'model.engine',
                '--optimization-level',5,'--tiling','full','--aux-streams',0,*flags,build=True)
        candidate=h/'fir-quant-candidate-selected.json';save(candidate,dict(current,vocoder_plan=str(directory/'model.plan.json')))
        original=benchmark(b,baseline,'fir-quant-control');measured=benchmark(b,candidate,'fir-quant-candidate')
        ci=compare(b,original,measured,'compare-fir-quant-screen');decision['screen_gain_ci_ms']=ci
        # If uncertainty overlaps a material predicted benefit, resolve with30waves.
        if ci[0]<=0<=ci[1] and estimated_pct>=.1:
            original=benchmark(b,baseline,'fir-quant-recheck-control',final=True)
            measured=benchmark(b,candidate,'fir-quant-recheck-candidate',final=True)
            ci=compare(b,original,measured,'compare-fir-quant-recheck');decision['recheck_gain_ci_ms']=ci
        key='wave_admission_to_last_pcm_ms'
        if ci[0]<=0 or measured['summary'][key]['p50']>=original['summary'][key]['p50']:
            decision['reason']='No retained matched E2E gain after relevant sampling; source current best preserved'
            save(h/'fir-quant-decision.json',decision);results.append(decision);continue
        lifecycle(b,candidate,'fir-quant')
        capture=h/'optimized-capture-acoustics.pt';replay=h/'fir-quant-replay.pt'
        run(b,'fir-quant-replay','scripts/audit_unified_acoustics.py','replay','--gpu',1,'--config','local_assets/runtime/runtime_fp32_b1.yaml',
            '--capture',capture,'--cfm-plan',current['cfm_plan'],'--vocoder-plan',directory/'model.plan.json','--output',replay)
        run(b,'fir-quant-audit','scripts/audit_unified_acoustics.py','audit','--gpu',1,'--config','local_assets/runtime/runtime_fp32_b1.yaml',
            '--capture',replay,'--output',h/'fir-quant-audit.json')
        control=benchmark(b,baseline,'fir-quant-final-control',final=True)
        final=benchmark(b,candidate,'fir-quant-final-candidate',final=True)
        ci=compare(b,control,final,'compare-fir-quant-final');decision['final_gain_ci_ms']=ci
        if ci[0]<=0 or final['summary'][key]['p50']>=control['summary'][key]['p50']:
            decision['reason']='Final rebuilt artifact gain did not persist; prior validated route preserved'
            save(h/'fir-quant-decision.json',decision);results.append(decision);continue
        value=read(candidate);value['status']='validated_local_sm89_int8'
        destination=ROOT/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json';previous=read(destination);save(destination,value)
        try:
            wav=ROOT/f'outputs/multibatch/b{b}/fir-quant-cli.wav'
            run(b,'fir-quant-cli','deployment/infer.py','--batch',b,'--ref-audio','/workspace/A_TEST_REF/male_news.wav',
                '--text','你好，这是量化融合引擎的部署验证。','--output',wav)
            import soundfile as sf
            import numpy as np
            pcm,sr=sf.read(wav);assert sr==22050 and len(pcm)>0 and np.isfinite(pcm).all() and np.any(pcm)
            save(h/'fir-quant-cli.json',dict(passed=True,batch=b,wav=str(wav),sha256=digest(wav),seconds=len(pcm)/sr))
        except Exception:
            save(destination,previous);raise
        save(h/'current-best-selected.json',value)
        baseline_path=h/'original-selected.json' if (h/'original-selected.json').exists() else h/'migration-selected.json'
        before=benchmark(b,baseline_path,'fir-quant-delivery-original',final=True)
        best_result=benchmark(b,h/'current-best-selected.json','fir-quant-delivery-best',final=True)
        gain=compare(b,before,best_result,'compare-fir-quant-delivery')
        report=read(h/'optimization-complete.json');save(h/'before-fir-quant-optimization-complete.json',report)
        def metrics(r):
            s=r['summary'][key];w=r['sustained_power']
            return dict(p50_ms=s['p50'],p95_ms=s['p95'],mean_power_w=w['power_w']['mean'],peak_power_w=w['power_w']['max'],first_chunk_requests_s=w['requests_per_second'])
        report.update(original=metrics(before),best=metrics(best_result),gain_ci_ms=gain,lifecycle='fir-quant-lifecycle.json',acoustic_audit='fir-quant-audit.json',cli='fir-quant-cli.json',residual_fusion='fir-quant-decision.json')
        decision.update(status='retained_validated',reason='Complete local math probe, matched E2E, full lifecycle, frozen same-recipe audit and CLI passed',selected=str(destination))
        save(h/'fir-quant-decision.json',decision);save(h/'optimization-complete.json',report);results.append(decision)
    from deployment.multibatch.final_review import review as final_review
    for b in a.batches:final_review(b)
    save(BASE/'fir-quant-matrix-decisions.json',results)
    print(json.dumps(dict(status='residual_fusion_review_complete',results=results)),flush=True)


if __name__=='__main__':main()
