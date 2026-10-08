"""Audit completed full-model graph evidence without accepting unfinished batches."""
import json,statistics
from pathlib import Path
from run_zipvoice_validation import ROOT,sha as file_sha
from functools import lru_cache

@lru_cache(maxsize=None)
def cached_sha(path, size, modified):
    return file_sha(path)

def sha(path):
    path=Path(path);info=path.stat()
    return cached_sha(path.resolve(),info.st_size,info.st_mtime_ns)
REPORTS=ROOT/'reports/sm89/zipvoice/a1007'
COMPLETE='model_graph_full_mapping_quality_unchanged_e2e_power_complete_review_pending'


def main():
    baseline_path=REPORTS/'migration-baseline.json'
    baseline=json.loads(baseline_path.read_text())
    reviewed={};pending=[]
    for batch in (1,2,4,8,16,32,64):
        source=baseline['batches'][str(batch)]
        path=REPORTS/f'b{batch}/history/027-model-graph-validation.json'
        if not path.exists():pending.append(batch);continue
        d=json.loads(path.read_text())
        if d['status']!=COMPLETE:pending.append(batch);continue
        assert d['baseline_sha256']==sha(baseline_path)
        assert d['route']==source['route'] and d['pcm_policy']==source['pcm_policy']
        assert d['control']==source['application'] and d['candidate']==d['control']+'_graph'
        runner=ROOT/f"src/inspark_infer/runtime/zipvoice/routes/{d['candidate']}.py"
        assert sha(runner)==d['candidate_runner_sha256']
        hashes={str(path.relative_to(ROOT)):sha(path)}
        def check(report_path,app):
            p=Path(report_path);p.relative_to(ROOT)
            r=json.loads(p.read_text());inv_path=p.parent/'engines.json'
            inv=json.loads(inv_path.read_text())
            assert r['engine_manifest_sha256']==sha(inv_path)
            assert r['runner_sha256']==sha(ROOT/f'src/inspark_infer/runtime/zipvoice/routes/{app}.py')
            assert r['shape'][0]==batch and r['pcm_chunk']==source['pcm_policy']['chunk'] and r['pcm_workers']==source['pcm_policy']['workers']
            assert r['graph_bitwise_guard'] and r['graph_state_relative_l2']==0
            if app.endswith('_graph'):assert r['graph_wave_bitwise_guard']
            assert r['power_limits_before_w']==r['power_limits_after_w']=='400.00, 400.00'
            assert all(x['pcm_items']==batch for x in r['results'])
            command=json.loads((p.parent/'command.json').read_text())
            assert sha(Path(command[command.index('--inputs')+1]))==r['input_sha256']
            for key,engine in inv['engines'].items():
                assert engine['sha256']==source['inventory']['engines'][key]['sha256']
                assert sha(Path(engine['path']))==engine['sha256']
                for plugin,digest in engine.get('plugin_sources',{}).items():assert sha(Path(plugin))==digest
            hashes[str(p.relative_to(ROOT))]=sha(p)
            return r
        assert len(d['functional'])==(17 if batch==1 else 18)
        quality_path=ROOT/f'outputs/zipvoice-validation/b{batch}'/('quality-inputs.json' if source['route'] in ('native','inherited') else source['route']+'-validation/quality-inputs.json')
        quality={(x['case'],x['row']):x['wav_sha256'] for x in json.loads(quality_path.read_text()) if x['route']==source['route']}
        for row in d['functional']:
            r=check(row['report'],d['candidate'])
            if row['kind']=='natural_same_length_mixed':assert not r['text_reuse']
            if row['kind']=='accepted_quality_case':
                for index in sorted({0,min(1,batch-1),batch-1}):
                    assert sha(Path(row['report']).parent/f'{index:04d}.wav')==quality[row['case'],index]
        for shape in d['shapes'].values():
            samples={'control':[],'candidate':[]}
            assert len(shape['blocks'])==8
            for block in shape['blocks']:
                r=check(block['report'],d[block['kind']]);samples[block['kind']]+=[x['all_pcm_s'] for x in r['results']]
            for kind,values in samples.items():
                assert len(values)==20 and statistics.median(values)==shape['statistics'][kind]['median_s']
        power=check(d['sustained_power']['report'],d['candidate'])
        assert d['sustained_power']['duration_s']>=30 and power['power_telemetry']==d['sustained_power']['telemetry']
        positive=all(s['gain_percent']>0 for s in d['shapes'].values())
        reviewed[str(batch)]={'decision':'retain_graph' if positive else 'requires_shape_tradeoff_review','application':d['candidate'] if positive else d['control'],'route':source['route'],'pcm_policy':source['pcm_policy'],'gains_percent':{k:s['gain_percent'] for k,s in d['shapes'].items()},'primary_median_ms':d['shapes']['primary760']['statistics']['candidate']['median_s']*1000,'power_w':power['power_telemetry']['board_power_w'],'evidence':hashes,'reason':'Matched 20 samples per route at each real shape improve, all accepted selected WAVs byte-identical, complete direct state/wave guards and boundary/mixed checks pass' if positive else 'Complete checks pass; gains require shape-specific decision','optimization_review_complete':False}
    out={'status':'graph_candidates_reviewed_other_optimization_pending','batches':reviewed,'pending':pending,'limits':'No conclusion about remaining operator opportunities or publication acceptance; complete captured-state/wave comparison is executed by runtime, selected WAV hashes independently rechecked here'}
    (REPORTS/'model-graph-review.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'reviewed':list(reviewed),'pending':pending,'decisions':{k:v['decision'] for k,v in reviewed.items()}}))


if __name__=='__main__':main()
