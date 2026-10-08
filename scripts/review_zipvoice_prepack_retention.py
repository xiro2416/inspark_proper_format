"""Audit the final B64 packed operand artifact and matched application evidence."""
import argparse,json,statistics
from pathlib import Path
from run_zipvoice_validation import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=int,choices=(8,16,32,64),default=64);parser.add_argument('--shape-tradeoff-reason', help='Explicit evidence-based review when a non-primary length regresses');args=parser.parse_args();batch=args.batch
    base=json.loads((ROOT/'reports/sm89/zipvoice/a1007/model-graph-review.json').read_text())['batches'][str(batch)]
    history=ROOT/f'reports/sm89/zipvoice/a1007/b{batch}/history';route='normtf32k16wp' if batch==64 else 'wp';control='normtf32k16' if batch==64 else 'inherited';runner='a1007_delivery_graph' if batch==64 else base['application']
    performance_path=history/f'020-{route}-performance.json';performance=json.loads(performance_path.read_text())
    assert performance['status']=='formal_candidate_e2e_quality_power_complete_review_pending'
    if batch==64:assert 'application_evidence_before_fulltext_extension_sha256' in performance,'Wait for explicit fulltext extension and transparent rebinding'
    application_path=history/f'018-{route}-application.json';app=json.loads(application_path.read_text());quality_path=history/f'019-{route}-quality.json';q=json.loads(quality_path.read_text())
    assert performance['application_evidence_sha256']==sha(application_path) and performance['quality_evidence_sha256']==sha(quality_path)
    assert app['status']=='variant_full_application_mapping_passed_quality_pending' and len(app['cases'])==11 and len(app['contract_cases'])==7
    assert all(x['source_selected_full_state_pcm_exact'] for x in app['cases'])
    inputs_path=ROOT/f'outputs/zipvoice-validation/b{batch}/{route}-validation/quality-inputs.json';assert q['status']=='complete' and q['input_inventory_sha256']==sha(inputs_path)
    inputs=json.loads(inputs_path.read_text());assert len(inputs)==33
    for row in inputs:assert sha(Path(row['path']))==row['wav_sha256']
    full=json.loads(Path(app['fulltext_control']).read_text());assert not full['text_reuse'] and full['graph_wave_bitwise_guard'] and full['graph_bitwise_guard']
    build_path=ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{route}/build.json';build=json.loads(build_path.read_text());assert build['engine_sha256']==app['engine_sha256']
    maps=json.loads((history/('038-actual-weight-prepack.json' if batch==64 else '041-actual-weight-prepack.json')).read_text());expected={x['module']:x for x in maps['rows']}
    for row in build['f32_replacements']:
        assert row['source_weight_sha256']==expected[row['module']]['source_weight_sha256'] and row['weight_sha256']==expected[row['module']]['packed_weight_sha256']
    engine_hashes={}
    for kind in (control,route):
        directory=ROOT/f'artifacts/zipvoice/a1007/b{batch}/fm-{kind}'
        metadata=json.loads((directory/'build.json').read_text())
        assert sha(directory/'engine.plan')==metadata['engine_sha256']
        engine_hashes[kind]=metadata['engine_sha256']
    def verify_inventory(report_path,kind):
        path=Path(report_path);data=json.loads(path.read_text());inventory_path=path.parent/'engines.json'
        assert data['engine_manifest_sha256']==sha(inventory_path)
        inv=json.loads(inventory_path.read_text());assert inv['engines']['fm']['sha256']==engine_hashes[kind]
        for source,digest in inv['engines']['fm'].get('plugin_sources',{}).items():assert sha(Path(source))==digest
        return data
    for name,shape in performance['shapes'].items():
        samples={control:[],route:[]}
        for block in shape['blocks']:
            data=verify_inventory(block['report'],block['route']);assert data['runner_sha256']==sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{runner}.py')
            assert data['graph_bitwise_guard'] and data['graph_wave_bitwise_guard'] and data['power_limits_before_w']==data['power_limits_after_w']=='400.00, 400.00'
            assert all(x['pcm_items']==batch for x in data['results'])
            samples[block['route']]+=[x['all_pcm_s'] for x in data['results']]
        for kind,values in samples.items():assert len(values)==20 and statistics.median(values)==shape['statistics'][kind]['median_s']
        if not args.shape_tradeoff_reason:assert shape['gain_percent']>0
    assert performance['shapes']['primary760']['gain_percent']>0
    if args.shape_tradeoff_reason:assert any(x['gain_percent']<=0 for x in performance['shapes'].values()),'Use ordinary retention when all shapes improve'
    assert performance['sustained_power']['duration_s']>=30
    power=verify_inventory(performance['sustained_power']['report'],route)
    assert power['power_telemetry']==performance['sustained_power']['telemetry']
    assert power['runner_sha256']==sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{runner}.py')
    assert power['power_limits_before_w']==power['power_limits_after_w']=='400.00, 400.00'
    assert all(x['pcm_items']==batch for x in power['results'])
    record={'status':'f32_operand_prepack_retained_remaining_transfer_review_pending','route':route,'application':runner,'pcm_policy':{'chunk':16,'workers':4} if batch==64 else base['pcm_policy'],'performance_evidence_sha256':sha(performance_path),'application_evidence_sha256':sha(application_path),'quality_evidence_sha256':sha(quality_path),'build_evidence_sha256':sha(build_path),'gains_percent':{k:v['gain_percent'] for k,v in performance['shapes'].items()},'primary_median_ms':performance['shapes']['primary760']['statistics'][route]['median_s']*1000,'power_w':performance['sustained_power']['telemetry']['board_power_w'],'decision':'Retain simple existing-math staticoperand prepack: actual12weights matchGPU RNA,33WAVs/selectedstates unchanged, directwholegraph checks andboundary/mixed/fulltext pass;20matchedsamples/route/shape improve. Original sourcecheckpoints/I8scales/1:3 untouched.','optimization_review_complete':False,'remaining':'Finish applicability screens forB4/8/16/32, finalall7execution-cost review and release/source-bound checks'}
    if args.shape_tradeoff_reason:
        record['shape_tradeoff_review']=args.shape_tradeoff_reason
        record['decision']='Retain for agreed primary760 objective after complete mapping/quality validation; other-length tradeoff explicitly disclosed. '+args.shape_tradeoff_reason
    (history/('040-f32-prepack-retention.json' if batch==64 else '042-f32-prepack-retention.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'median_ms':record['primary_median_ms'],'gains_percent':record['gains_percent']}))


if __name__=='__main__':main()
