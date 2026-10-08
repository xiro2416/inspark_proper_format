"""Bind fresh-download PCM expectations to the retained, validated application."""
import argparse,json,sys
from pathlib import Path
from run_zipvoice_validation import ROOT,sha
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.build.zipvoice import BATCHES,safe_path,validate_bundle,check_workload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();selection_path=args.selection.resolve();selection_path.relative_to(ROOT)
    selected=json.loads(selection_path.read_text())
    assert selected['status']=='all_batches_migrated_optimized_validated'
    registry=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
    assert set(selected['batches'])==set(registry['bundles'])==set(map(str,BATCHES))
    result={'status':'source_bound_final_pcm_expectations','selection_sha256':sha(selection_path),'batches':{}}
    for batch in BATCHES:
        chosen=selected['batches'][str(batch)]
        assert chosen['migration_accepted'] and chosen['optimization_review_complete']
        bundle=safe_path(ROOT,registry['bundles'][str(batch)]['local_path']);manifest=validate_bundle(bundle,batch)
        cases=[]
        for report_name in chosen['fresh_validation_reports']:
            path=safe_path(ROOT,report_name);proof=json.loads(path.read_text())
            assert proof['runner_sha256']==manifest['runner_sha256']
            assert proof['pcm_chunk']==manifest['pcm_policy']['chunk'] and proof['pcm_workers']==manifest['pcm_policy']['workers']
            assert proof['graph_bitwise_guard'] and proof['graph_state_relative_l2']==0
            if manifest['application'].endswith('_graph'):assert proof['graph_wave_bitwise_guard']
            assert all(row['pcm_items']==batch for row in proof['results'])
            assert proof['power_limits_before_w']==proof['power_limits_after_w']=='400.00, 400.00'
            command=json.loads((path.parent/'command.json').read_text())
            condition=Path(command[command.index('--inputs')+1]).resolve();condition.relative_to(ROOT)
            assert sha(condition)==proof['input_sha256']
            engine_manifest=Path(command[command.index('--engine-manifest')+1]).resolve();engine_manifest.relative_to(ROOT)
            assert sha(engine_manifest)==proof['engine_manifest_sha256']
            inventory=json.loads(engine_manifest.read_text())
            for key,engine in inventory['engines'].items():
                assert engine['sha256']==manifest['engines'][key]['sha256']
                assert sha(Path(engine['path']))==engine['sha256']
            check_workload(manifest,proof['workload'])
            rows=sorted({0,min(1,batch-1),batch-1})
            hashes={str(row):sha(path.parent/f'{row:04d}.wav') for row in rows}
            from safetensors.numpy import load_file
            tokens=load_file(str(condition))['token_ids']
            mixed=batch>1 and tokens.shape[0]==batch and any((tokens[0]!=tokens[i]).any() for i in range(1,batch))
            if mixed:assert not proof['text_reuse']
            case={'condition':str(condition),'input_sha256':proof['input_sha256'],'workload':proof['workload'],'seed':proof['seed'],'pcm_sha256':hashes,'mixed_rows':bool(mixed),'source_report':report_name,'source_report_sha256':sha(path),'bundle_id':manifest['bundle_id']}
            cases.append(case)
        assert {case['workload']['total_frames'] for case in cases}>={600,760,920}
        assert {case['workload']['padded_tokens'] for case in cases}>={52,78,141}
        assert batch==1 or any(case['mixed_rows'] for case in cases)
        result['batches'][str(batch)]=cases
    output=args.output.resolve();output.relative_to(ROOT);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'batches':list(result['batches']),'cases':sum(len(x) for x in result['batches'].values())}))


if __name__=='__main__':main()
