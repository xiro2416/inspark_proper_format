"""Prove complete packed-operand model identity and reuse identical quality audio."""
import argparse,fcntl,json
from pathlib import Path
from run_zipvoice_validation import ROOT,invoke,contract_cases,sha
from zipvoice_selected_route import idle_gpu_preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=int,choices=(8,16,32,64),default=64);args=parser.parse_args();batch=args.batch
    review=json.loads((ROOT/'reports/sm89/zipvoice/a1007/model-graph-review.json').read_text())['batches'][str(batch)]
    lock=Path('/workspace/.cache/inspark/gpu-locks/1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);preflight=idle_gpu_preflight()
    import torch
    from safetensors.torch import load_file
    history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history';route='normtf32k16wp' if batch==64 else 'wp';control='normtf32k16' if batch==64 else 'inherited';app='a1007_delivery_graph' if batch==64 else review['application'];policy={'chunk':16,'workers':4} if batch==64 else review['pcm_policy'];extra=['--pcm-chunk',str(policy['chunk']),'--pcm-workers',str(policy['workers'])]
    proof=json.loads((history/('038-actual-weight-prepack.json' if batch==64 else '041-actual-weight-prepack.json')).read_text());assert proof['status']=='all12_actual_float32_ffn_weight_operands_gpu_rna_exact'
    manifest=json.loads((ROOT/'outputs/zipvoice-validation/cases/manifest.json').read_text());root=ROOT/f'outputs/zipvoice-validation/b{batch}/{route}-validation';evidence=history/f'018-{route}-application.json'
    report={'status':'running','batch':batch,'route':route,'control':control,'application':app,'pcm_policy':policy,'engine_sha256':json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{route}/build.json').read_text())['engine_sha256'],'cases':[],'matched_conditions':[],'contract_cases':[],'quality_inputs':[],'preflight':preflight}
    def save():evidence.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for index,case in enumerate(manifest['quality_cases']):
        source=ROOT/(f'outputs/zipvoice-validation/b64/{control}-validation/{control}/real-{index:03d}' if batch==64 else f'outputs/zipvoice-validation/b{batch}/model-graph/real-{index:03d}')
        # Previously reviewed source functional run uses the same application/PCM/seed.
        source_report=json.loads((source/'report.json').read_text());assert source_report['runner_sha256']==sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{app}.py')
        source_inventory=json.loads((source/'engines.json').read_text())
        assert source_report['engine_manifest_sha256']==sha(source/'engines.json')
        assert source_inventory['engines']['fm']['sha256']==json.loads((ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{control}/build.json').read_text())['engine_sha256']
        assert source_report['pcm_chunk']==policy['chunk'] and source_report['pcm_workers']==policy['workers'] and source_report['seed']==9102
        destination=root/route/f'real-{index:03d}'
        result,rows=invoke(batch,route,case,destination,application=app,extra=extra)
        assert result['graph_wave_bitwise_guard'] and result['input_sha256']==source_report['input_sha256']
        a=load_file(str(source/'selected-state.safetensors'));b=load_file(str(destination/'selected-state.safetensors'))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),('Complete source/candidate selected states differ',index)
        for row in rows:
            path=destination/f'{row:04d}.wav';assert sha(path)==sha(source/f'{row:04d}.wav')
            report['quality_inputs'].append({'batch':batch,'route':route,'case':index,'row':row,'path':str(path),'wav_sha256':sha(path),'target_text':case['text'],'reference_wav':case['reference_wav'],'reference_sha256':case['reference_sha256']})
        report['cases'].append({'route':route,'case':index,'frames':case['total_frames'],'report':str(destination/'report.json'),'all_pcm_items':batch,'graph_direct_exact':True,'source_selected_full_state_pcm_exact':True})
        report['matched_conditions'].append({'case':index,'original_noise_conditions_mask_grid_exact':True,'selected_state_relative_l2':0.0});save();print(f'Actual source/packed state andPCM case{index} exact',flush=True)
    for case in contract_cases(batch,manifest):
        destination=root/route/case['name'];result,_=invoke(batch,route,case,destination,application=app,extra=extra)
        assert result['graph_wave_bitwise_guard']
        if case['kind']=='natural_same_length_mixed':assert not result['text_reuse']
        report['contract_cases'].append({'name':case['name'],'kind':case['kind'],'report':str(destination/'report.json'),'passed':True});save()
    fulltext=root/route/'fulltext-control'
    full_result,_=invoke(batch,route,manifest['primary_760'],fulltext,application=app,extra=[*extra,'--disable-text-reuse'])
    assert not full_result['text_reuse'] and full_result['graph_wave_bitwise_guard']
    source_quality=history/('019-normtf32k16-quality.json' if batch==64 else '004-quality.json');q=json.loads(source_quality.read_text());assert q['status']=='complete'
    values=q['by_route'][control];inputs=root/'quality-inputs.json';inputs.write_text(json.dumps(report['quality_inputs'],indent=2)+'\n')
    quality={'status':'complete','count':33,'input_inventory_sha256':sha(inputs),'by_route':{route:values},'reused_from':str(source_quality.relative_to(ROOT)),'source_quality_sha256':sha(source_quality),'reuse_basis':'All33selectedrealWAVs byte-identical plusactual12operandGPU RNA andselectedfullmodel states exact; complete directstate/wave guard. No fabricated rescore.'}
    (history/f'019-{route}-quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    report.update(status='variant_full_application_mapping_passed_quality_pending',fulltext_control=str(fulltext/'report.json'))
    save();print('Complete model identity andboundary/mixed pass; identicalquality reused',flush=True)


if __name__=='__main__':main()
