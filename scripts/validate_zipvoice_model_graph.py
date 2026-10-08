"""Validate isolated model graphs against accepted migration WAVs and matched E2E."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import sys

from run_zipvoice_validation import ROOT, invoke, sha, contract_cases
sys.path.insert(0,str(ROOT/'src'))
from zipvoice_selected_route import idle_gpu_preflight
from measure_zipvoice_baselines import percentile

REPORTS=ROOT/'reports/sm89/zipvoice/a1007'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16,32,64])
    args=p.parse_args()
    baseline_path=REPORTS/'migration-baseline.json';baseline=json.loads(baseline_path.read_text())
    assert baseline['status']=='all_seven_migrations_accepted_optimization_pending'
    lease=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=idle_gpu_preflight()
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    cases={'short':min(manifest['quality_cases'],key=lambda c:c['total_frames']),'primary760':manifest['primary_760'],'long':max(manifest['quality_cases'],key=lambda c:c['total_frames'])}
    for batch in args.batches:
        selected=baseline['batches'][str(batch)];route=selected['route'];control=selected['application'];candidate=control+'_graph';policy=selected['pcm_policy']
        inputs=ROOT/f'outputs/zipvoice-validation/b{batch}'/('quality-inputs.json' if route in ('native','inherited') else f'{route}-validation/quality-inputs.json')
        accepted_audio={(row['case'],row['row']):row['wav_sha256'] for row in json.loads(inputs.read_text()) if row['route']==route}
        history=REPORTS/f'b{batch}/history';evidence=history/'027-model-graph-validation.json'
        report={'status':'running','batch':batch,'route':route,'control':control,'candidate':candidate,'pcm_policy':policy,'preflight':preflight,'baseline_sha256':sha(baseline_path),'functional':[],'shapes':{}}
        def save():evidence.write_text(json.dumps(report,indent=2)+'\n')
        def call(application,case,dest,**kwargs):
            return invoke(batch,route,case,dest,application=application,extra=['--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers'])],**kwargs)
        save()
        for index,case in enumerate(manifest['quality_cases']):
            dest=ROOT/f'outputs/zipvoice-validation/b{batch}/model-graph/real-{index:03d}'
            data,rows=call(candidate,case,dest)
            assert data['graph_wave_bitwise_guard']
            for row in rows:assert sha(dest/f'{row:04d}.wav')==accepted_audio[index,row],(batch,index,row,'Model capture changed accepted quality audio')
            report['functional'].append({'kind':'accepted_quality_case','case':index,'frames':case['total_frames'],'full_state_wave_direct_graph_exact':True,'accepted_selected_pcm_exact':True,'report':str(dest/'report.json')})
            save();print(f'B{batch} full-model graph real{index} directstate/wave andacceptedPCM exact',flush=True)
        # Extra interface extremes and complete distinct text rows.
        for case in contract_cases(batch,manifest):
            dest=ROOT/f'outputs/zipvoice-validation/b{batch}/model-graph'/case['name']
            data,_=call(candidate,case,dest);assert data['graph_wave_bitwise_guard']
            if case['kind']=='natural_same_length_mixed':assert not data['text_reuse']
            report['functional'].append({'kind':case['kind'],'name':case['name'],'frames':case['workload']['total_frames'],'full_state_wave_direct_graph_exact':True,'report':str(dest/'report.json')});save()
        for label,case in cases.items():
            values={'control':[],'candidate':[]};blocks=[]
            for index,kind in enumerate(('control','candidate','candidate','control','candidate','control','control','candidate')):
                app=control if kind=='control' else candidate
                dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/model-graph/{label}/{index}-{kind}'
                data,_=call(app,case,dest,warmup=3,repetitions=5,functional=False)
                samples=[r['all_pcm_s'] for r in data['results']];assert len(samples)==5
                values[kind]+=samples;blocks.append({'kind':kind,'median_s':statistics.median(samples),'report':str(dest/'report.json')})
            stats={kind:{'count':len(v),'median_s':statistics.median(v),'p95_s':percentile(v,.95)} for kind,v in values.items()}
            report['shapes'][label]={'frames':case['total_frames'],'statistics':stats,'gain_percent':100*(stats['control']['median_s']-stats['candidate']['median_s'])/stats['control']['median_s'],'blocks':blocks};save()
            print(json.dumps({'batch':batch,'shape':label,'gain_percent':report['shapes'][label]['gain_percent'],'statistics':stats}),flush=True)
        dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/model-graph/sustained760'
        data,_=invoke(batch,route,cases['primary760'],dest,application=candidate,warmup=5,functional=False,extra=['--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers']),'--minimum-seconds','32'])
        duration=(data['results'][-1]['request_done_ns']-data['results'][0]['request_start_ns'])/1e9;assert duration>=30
        report.update(status='model_graph_full_mapping_quality_unchanged_e2e_power_complete_review_pending',quality_evidence='All11 accepted-corpus selectedPCM bytes unchanged plusdirect complete state/wave guards; reuse accepted CER/UTMOS/SIM-o, no fabricated rescore',sustained_power={'duration_s':duration,'telemetry':data['power_telemetry'],'report':str(dest/'report.json')},candidate_runner_sha256=sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{candidate}.py'))
        save();print(json.dumps({'batch':batch,'status':report['status']}),flush=True)


if __name__=='__main__':main()
