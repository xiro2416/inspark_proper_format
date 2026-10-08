"""Confirm inherited compute, PCM and delivery as one target migration scheme."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import sys

from run_zipvoice_validation import ROOT, invoke, sha, contract_cases, inventory
sys.path.insert(0,str(ROOT/'src'))
from zipvoice_selected_route import selected_route, idle_gpu_preflight
from measure_zipvoice_baselines import percentile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schemes-json',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16,32,64])
    args=p.parse_args()
    args.schemes_json=args.schemes_json.resolve()
    args.schemes_json.relative_to(ROOT)
    choice=json.loads(args.schemes_json.read_text())
    assert choice['status']=='migration_scheduling_combination_reviewed'
    routes=ROOT/'reports/sm89/zipvoice/a1007/reviewed-compute-routes.json'
    lease=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=idle_gpu_preflight()
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text())
    cases={'short':min(manifest['quality_cases'],key=lambda c:c['total_frames']),
           'primary760':manifest['primary_760'],
           'long':max(manifest['quality_cases'],key=lambda c:c['total_frames'])}
    fresh={'status':'combined_functional_inputs_running','batches':{}}
    for batch in args.batches:
        history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history'
        route,compute=selected_route(batch,routes,cases['primary760'])
        selected=choice['batches'][str(batch)]
        assert selected['route']==route
        for name,digest in selected['evidence'].items():
            path=(ROOT/name).resolve();path.relative_to(ROOT)
            assert sha(path)==digest,name
        source={'application':'a1007_delivery' if batch==32 else 'a1007','chunk':6 if batch==32 else 16,'workers':4}
        candidate={k:selected[k] for k in ('application','chunk','workers')}
        assert candidate['application'] in ('a1007','a1007_delivery')
        assert candidate['chunk']>0 and candidate['workers']>0
        schemes={'control':source,'candidate':candidate}
        def call(kind,case,destination,**kw):
            config=schemes[kind]
            return invoke(batch,route,case,destination,application=config['application'],
                extra=['--pcm-chunk',str(config['chunk']),'--pcm-workers',str(config['workers'])],**kw)
        evidence=history/'024-combined-migration.json'
        report={'status':'running','batch':batch,'compute_evidence':compute,'source':source,'candidate':candidate,
                'preflight':preflight,'selection_sha256':sha(args.schemes_json),'shapes':{},'functional':[]}
        if evidence.exists():
            previous=json.loads(evidence.read_text())
            if (previous.get('selection_sha256')==sha(args.schemes_json)
                and previous.get('compute_evidence')==compute
                and previous.get('source')==source and previous.get('candidate')==candidate):
                report=previous
                print(json.dumps({'batch':batch,'event':'resume_bound_completed_measurements'}),flush=True)
        def save():evidence.write_text(json.dumps(report,indent=2)+'\n')
        save()
        for label,case in cases.items():
            if label in report['shapes']:
                prior=report['shapes'][label]
                paths=[Path(block['report']) for block in prior.get('blocks',prior.get('source_reports',[]))]
                assert paths, 'Completed timing lacks source-bound reports'
                for path in paths:
                    timing=json.loads(path.read_text())
                    manifest_path=path.parent/'engines.json'
                    assert timing['engine_manifest_sha256']==sha(manifest_path)
                    assert json.loads(manifest_path.read_text())['engines']['fm']['sha256']==compute['engine_sha256']
                    kind='candidate' if path.parent.name.endswith('candidate') else 'control'
                    application=schemes[kind]['application']
                    if 'source_reports' in prior:application=source['application']
                    assert timing['runner_sha256']==sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{application}.py')
                continue
            if candidate==source:
                # The delivery recheck measured this exact unchanged source
                # policy. Reuse those bound samples rather than compare it to itself.
                previous=history/'023-selected-delivery.json'
                data=json.loads(previous.read_text())
                assert data['status']=='delivery_reuse_measured_review_pending'
                assert data['compute_evidence']['engine_sha256']==compute['engine_sha256']
                values=[];bound=[]
                for block in data['shapes'][label]['blocks']:
                    if block['application']!=source['application']:continue
                    path=Path(block['report']);path.resolve().relative_to(ROOT)
                    timing=json.loads(path.read_text())
                    assert timing['pcm_chunk']==source['chunk'] and timing['pcm_workers']==source['workers']
                    assert timing['shape']==[batch,case['total_frames'],100]
                    assert timing['input_sha256']==sha(case['condition'])
                    assert timing['runner_sha256']==sha(ROOT/f"src/inspark_infer/runtime/zipvoice/routes/{source['application']}.py")
                    manifest_path=path.parent/'engines.json'
                    assert timing['engine_manifest_sha256']==sha(manifest_path)
                    engine_manifest=json.loads(manifest_path.read_text())
                    assert engine_manifest['engines']['fm']['sha256']==compute['engine_sha256']
                    assert all(item['pcm_items']==batch for item in timing['results'])
                    values.extend(item['all_pcm_s'] for item in timing['results'])
                    bound.append({'report':str(path),'report_sha256':sha(path)})
                assert len(values)==20
                report['shapes'][label]={'frames':case['total_frames'],
                    'comparison':'unchanged_source_policy_existing_matched_samples_reused',
                    'statistics':{'candidate':{'count':len(values),'median_s':statistics.median(values),'p95_s':percentile(values,.95)}},
                    'gain_percent':None,'selected_wav_exact':True,'source_evidence_sha256':sha(previous),'source_reports':bound}
                save();continue
            samples={'control':[],'candidate':[]};blocks=[];pcm=None
            for index,kind in enumerate(('control','candidate','candidate','control','candidate','control','control','candidate')):
                dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/combined-migration/{label}/{index}-{kind}'
                data,rows=call(kind,case,dest,warmup=3,repetitions=5,functional=False)
                current={row:sha(dest/f'{row:04d}.wav') for row in rows}
                if pcm is None:pcm=current
                assert pcm==current,(batch,label,'Combined scheduling changed selected WAV')
                values=[x['all_pcm_s'] for x in data['results']];assert len(values)==5
                samples[kind]+=values;blocks.append({'scheme':kind,'median_s':statistics.median(values),'report':str(dest/'report.json')})
            stats={kind:{'count':len(values),'median_s':statistics.median(values),'p95_s':percentile(values,.95)} for kind,values in samples.items()}
            report['shapes'][label]={'frames':case['total_frames'],'statistics':stats,'gain_percent':100*(stats['control']['median_s']-stats['candidate']['median_s'])/stats['control']['median_s'],'selected_wav_exact':True,'blocks':blocks}
            save();print(json.dumps({'batch':batch,'shape':label,'statistics':stats,'gain_percent':report['shapes'][label]['gain_percent']}),flush=True)
        # All-row transport/PCM oracle and selected complete-state agreement,
        # including the profile extremes and distinct natural text rows.
        checks=[cases['primary760']]+[c for c in contract_cases(batch,manifest) if c['name'] in ('synthetic-t600-l52','synthetic-t920-l141') or c['kind']=='natural_same_length_mixed']
        from safetensors.torch import load_file
        import torch
        fresh_cases=[]
        report['functional']=[]
        for index,case in enumerate(checks):
            outputs={};proofs={}
            for kind in (('candidate',) if candidate==source else ('control','candidate')):
                dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/combined-migration/functional/{index}-{kind}'
                proofs[kind],rows=call(kind,case,dest)
                outputs[kind]=dest
                if schemes[kind]['application']=='a1007_delivery':
                    assert proofs[kind]['compact_input_all_batch_tokens_speech_rms_bitwise_exact']
                    assert proofs[kind]['overlap_all_batch_transfers_and_ordered_pcm_exact']
                if case.get('kind')=='natural_same_length_mixed':assert not proofs[kind]['text_reuse']
            pcm={str(row):sha(outputs['candidate']/f'{row:04d}.wav') for row in rows}
            if candidate!=source:
                left=load_file(str(outputs['control']/'selected-state.safetensors'))
                right=load_file(str(outputs['candidate']/'selected-state.safetensors'))
                assert left.keys()==right.keys()
                assert all(torch.equal(left[k],right[k]) for k in left),(batch,index,'Combined state/conditions differ')
                assert all(sha(outputs['control']/f'{int(row):04d}.wav')==digest for row,digest in pcm.items())
            report['functional'].append({'frames':case['workload']['total_frames'],'name':case.get('name','primary760'),'kind':case.get('kind','natural_primary'),'selected_state_and_pcm_exact':True if candidate!=source else None,'graph_direct_bitwise_guard':proofs['candidate']['graph_bitwise_guard'],'comparison':'same_configuration_direct_graph_guard' if candidate==source else 'independent_source_candidate_states_and_pcm','all_rows_pcm_count':batch,'reports':{k:str(v/'report.json') for k,v in outputs.items()}})
            fresh_cases.append({'condition':case['condition'],'workload':case['workload'],'seed':9102,'pcm_sha256':pcm,'mixed_rows':case.get('kind')=='natural_same_length_mixed'})
            save()
        dest=ROOT/f'outputs/zipvoice-benchmarks/b{batch}/combined-migration/sustained760'
        config=candidate
        data,_=invoke(batch,route,cases['primary760'],dest,application=config['application'],warmup=5,functional=False,
            extra=['--pcm-chunk',str(config['chunk']),'--pcm-workers',str(config['workers']),'--minimum-seconds','32'])
        duration=(data['results'][-1]['request_done_ns']-data['results'][0]['request_start_ns'])/1e9
        assert duration>=30
        report.update(status='combined_migration_mapping_performance_power_complete_review_pending',
            inventory=inventory(batch,route,cases['primary760']['workload']),
            sustained_power={'duration_s':duration,'telemetry':data['power_telemetry'],'report':str(dest/'report.json')})
        save();fresh['batches'][str(batch)]=fresh_cases
        (ROOT/'reports/sm89/zipvoice/a1007/combined-functional-expected.json').write_text(json.dumps(fresh,indent=2)+'\n')
        print(json.dumps({'batch':batch,'status':report['status']}),flush=True)
    fresh['status']='combined_functional_inputs_complete'
    (ROOT/'reports/sm89/zipvoice/a1007/combined-functional-expected.json').write_text(json.dumps(fresh,indent=2)+'\n')


if __name__=='__main__':main()
