"""Resume the evidence-gated B64 TF32 build after serial graph validation/probe."""
import argparse,json,subprocess,time
from pathlib import Path
from run_zipvoice_validation import ROOT,environment,sha
REPORTS=ROOT/'reports/sm89/zipvoice/a1007'


def live(pid):
    state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return state.returncode==0 and bool(state.stdout.strip()) and not state.stdout.strip().startswith('Z')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph-pid',type=int,required=True)
    p.add_argument('--probe-pid',type=int,required=True)
    p.add_argument('--suffix',choices=('normtf32','normtf32k16'),default='normtf32')
    p.add_argument('--probe-evidence',type=Path,default=REPORTS/'b64/history/028-normal-tf32-microprobe.json')
    p.add_argument('--probe-candidate')
    args=p.parse_args()
    args.probe_evidence=args.probe_evidence.resolve();args.probe_evidence.relative_to(ROOT)
    evidence=REPORTS/('normal-k16-execution.json' if args.suffix=='normtf32k16' else 'normal-tf32-execution.json')
    record={'status':'waiting_verified_graph_and_probe_processes','graph_pid':args.graph_pid,'probe_pid':args.probe_pid,'suffix':args.suffix,'commands':[]}
    def save():evidence.write_text(json.dumps(record,indent=2)+'\n')
    def run(command,label):
        record['status']=label;record['commands'].append(command);save()
        print(label,flush=True)
        with (ROOT/f'outputs/build-logs/{label}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        while live(args.graph_pid) or live(args.probe_pid):time.sleep(10)
        for batch in (1,2,4,8,16,32,64):
            d=json.loads((REPORTS/f'b{batch}/history/027-model-graph-validation.json').read_text())
            assert d['status']=='model_graph_full_mapping_quality_unchanged_e2e_power_complete_review_pending'
        probe=json.loads(args.probe_evidence.read_text())
        assert probe['status'] in ('synthetic_complete_normal_write_read_microprobe_not_deployment_acceptance','synthetic_normal_tf32_geometry_microprobe_not_deployment_acceptance')
        for key,name in [('control','b64'),(args.probe_candidate or 'candidate','b64_'+args.suffix)]:
            assert probe['source_sha256'][key]==sha(ROOT/f'.work/zipvoice/{name}/code/online_branch_stats_runtime_kernel.py')
        by_t={r['frames']:r for r in probe['records'] if not args.probe_candidate or r['candidate']==args.probe_candidate}
        savings_ms=8*sum(n*(by_t[t]['median_ms']['control']-by_t[t]['median_ms']['candidate']) for t,n in [(760,6),(380,6),(190,4)])
        baseline=json.loads((REPORTS/'migration-baseline.json').read_text())
        graph=json.loads((REPORTS/'b64/history/027-model-graph-validation.json').read_text())
        current_ms=graph['shapes']['primary760']['statistics']['candidate']['median_s']*1000
        potential=100*savings_ms/current_ms
        record.update(synthetic_stage_weighted_savings_ms=savings_ms,synthetic_potential_percent=potential,potential_limits='Synthetic JIT serial stage estimate, not AOT/E2E attribution. Eight FM steps, stage layer counts2/2/4/4/4. Used only to decide deeper investment.')
        if potential<1:
            record['status']='microprobe_deep_build_deferred_review_other_paths';save();return
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/review_zipvoice_model_graph.py')],'normal-tf32-refresh-graph-review')
        current=json.loads((REPORTS/'model-graph-review.json').read_text())['batches']['64']
        assert current['decision']=='retain_graph','Resolve current-best shape tradeoff before independent TF32 comparison'
        assert current['route']=='inherited'
        app=current['application'];policy=current['pcm_policy']
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/build_zipvoice_variants.py'),'--batches','64','--suffix',args.suffix],'normal-tf32-build-minimal')
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/validate_zipvoice_variant.py'),'--batch','64','--route',args.suffix,'--control','inherited','--application',app,'--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers'])],'normal-tf32-full-mapping-quality-launch')
        job=json.loads((REPORTS/f'b64/history/019-{args.suffix}-quality-job.json').read_text())
        record.update(status='waiting_specific_cpu_quality_job',quality_pid=job['pid']);save()
        while live(job['pid']):time.sleep(10)
        quality=json.loads((REPORTS/f'b64/history/019-{args.suffix}-quality.json').read_text())
        inputs=ROOT/f'outputs/zipvoice-validation/b64/{args.suffix}-validation/quality-inputs.json'
        assert quality['status']=='complete' and quality['input_inventory_sha256']==sha(inputs)
        run([str(ROOT/'.venv-zipvoice/bin/python'),str(ROOT/'scripts/measure_zipvoice_variant.py'),'--batch','64','--route',args.suffix,'--control','inherited','--application',app,'--control-application',app,'--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers']),'--current-best-review',str(REPORTS/'model-graph-review.json')],'normal-tf32-current-best-formal')
        record['status']='normal_tf32_full_evidence_complete_review_pending';save()
    except Exception as exc:
        record.update(status='stopped_requires_evidence_review',error=repr(exc));save();raise


if __name__=='__main__':main()
