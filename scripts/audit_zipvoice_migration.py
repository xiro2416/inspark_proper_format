"""Audit actual seven-batch migration evidence before optimization may start."""
import argparse
import json
from pathlib import Path
import sys

from run_zipvoice_validation import ROOT, sha, inventory
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.build.zipvoice import BATCHES, check_workload

REPORTS=ROOT/'reports/sm89/zipvoice/a1007'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,help='Explicit final seven-batch decisions; omitted performs a partial audit')
    args=p.parse_args()
    choice_path=REPORTS/'reviewed-migration-combinations.json'
    choice=json.loads(choice_path.read_text())
    assert choice['status']=='migration_scheduling_combination_reviewed'
    review=None
    if args.review:
        args.review=args.review.resolve();args.review.relative_to(ROOT)
        review=json.loads(args.review.read_text())
        assert review['status']=='all_seven_migration_combinations_reviewed'
        assert set(review['batches'])==set(map(str,BATCHES))
    result={'status':'incomplete','batches':{},'missing':[],'selection_sha256':sha(choice_path)}
    for batch in BATCHES:
        h=REPORTS/f'b{batch}/history';combined=h/'024-combined-migration.json'
        if not combined.exists():result['missing'].append(batch);continue
        data=json.loads(combined.read_text())
        if data['status']!='combined_migration_mapping_performance_power_complete_review_pending':
            result['missing'].append(batch);continue
        scheme=choice['batches'][str(batch)];route=scheme['route']
        assert data['selection_sha256']==sha(choice_path)
        assert data['candidate']=={k:scheme[k] for k in ('application','chunk','workers')}
        actual=inventory(batch,route,data['inventory']['workload'])
        assert actual==data['inventory'],(batch,'Current engine/source differs from tested inventory')
        engines=actual['engines']
        assert engines['fm']['shape_profile']['x']==[[batch,t,100] for t in (600,760,920)]
        assert engines['text']['shape_profile']['token_ids']==[[batch,l] for l in (52,78,141)]
        assert engines['unique']['shape_profile']['token_ids']==[[1,l] for l in (52,78,141)]
        assert engines['vocos']['shape_profile']['mel']==[[batch,100,t] for t in (225,385,545)]
        check_workload({'batch':batch,'engines':engines},actual['workload'])
        assert data['preflight']['power_cap_w']==400
        minimum=h/('002-minimal-compute.json' if route in ('native','inherited') else f'016-{route}-minimal.json')
        m=json.loads(minimum.read_text())
        key='fm-native' if route=='native' else 'fm-inherited'
        assert m['engine_sha256'][key]==engines['fm']['sha256']
        assert {t['frames'] for t in m['tests']}=={600,601,759,760,761,919,920}
        assert all(t['finite_all'] and t['all_rows_written'] for t in m['tests'])
        quality=h/('004-quality.json' if route in ('native','inherited') else f'019-{route}-quality.json')
        q=json.loads(quality.read_text());assert q['status']=='complete' and route in q['by_route']
        inputs=ROOT/f'outputs/zipvoice-validation/b{batch}'/('quality-inputs.json' if route in ('native','inherited') else f'{route}-validation/quality-inputs.json')
        assert q['input_inventory_sha256']==sha(inputs)
        for row in q['results']:
            path=Path(row['path']).resolve();path.relative_to(ROOT)
            assert sha(path)==row['wav_sha256'],(batch,'Quality audio changed')
        assert data['compute_evidence']['quality_evidence_sha256']==sha(quality)
        import torch
        from safetensors.torch import load_file
        for case in data['functional']:
            if case['comparison']!='same_configuration_direct_graph_guard':
                left_dir=Path(case['reports']['control']).parent
                right_dir=Path(case['reports']['candidate']).parent
                left=load_file(str(left_dir/'selected-state.safetensors'))
                right=load_file(str(right_dir/'selected-state.safetensors'))
                assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
                proof=json.loads((right_dir/'report.json').read_text())
                for row in proof['diagnostic_selected_state_rows']:
                    assert sha(left_dir/f'{row:04d}.wav')==sha(right_dir/f'{row:04d}.wav')
            assert case['all_rows_pcm_count']==batch and case['graph_direct_bitwise_guard']
            if case['comparison']!='same_configuration_direct_graph_guard':assert case['selected_state_and_pcm_exact']
            for name,path in case['reports'].items():
                path=Path(path).resolve();path.relative_to(ROOT)
                proof=json.loads(path.read_text());manifest=path.parent/'engines.json'
                assert proof['runner_sha256']==sha(ROOT/f"src/inspark_infer/runtime/zipvoice/routes/{data['candidate' if name=='candidate' else 'source']['application']}.py")
                assert proof['engine_manifest_sha256']==sha(manifest)
                record=json.loads(manifest.read_text())
                assert record['engines']['fm']['sha256']==engines['fm']['sha256']
                assert proof['graph_bitwise_guard'] and all(item['pcm_items']==batch for item in proof['results'])
                assert proof['power_limits_before_w']==proof['power_limits_after_w']=='400.00, 400.00'
                if case['kind']=='natural_same_length_mixed':assert not proof['text_reuse']
        assert {c['frames'] for c in data['functional']}>={600,760,920}
        assert batch==1 or any(c['kind']=='natural_same_length_mixed' for c in data['functional'])
        assert set(data['shapes'])=={'short','primary760','long'}
        assert all(shape['statistics']['candidate']['count']==20 for shape in data['shapes'].values())
        power=data['sustained_power'];assert power['duration_s']>=30
        power_path=Path(power['report']);power_proof=json.loads(power_path.read_text())
        assert power_proof['runner_sha256']==sha(ROOT/f"src/inspark_infer/runtime/zipvoice/routes/{scheme['application']}.py")
        assert power_proof['power_limits_before_w']==power_proof['power_limits_after_w']=='400.00, 400.00'
        assert power_proof['power_telemetry']==power['telemetry']
        evidence={str(path.relative_to(ROOT)):sha(path) for path in (combined,minimum,quality,inputs,power_path)}
        result['batches'][str(batch)]={'route':route,'application':scheme['application'],
            'pcm_policy':{'chunk':scheme['chunk'],'workers':scheme['workers']},
            'inventory':actual,'evidence':evidence,'quality':q['by_route'][route],
            'primary_median_s':data['shapes']['primary760']['statistics']['candidate']['median_s'],
            'power_w':power['telemetry']['board_power_w'],'migration_accepted':False}
        if review:
            decision=review['batches'][str(batch)]
            assert decision['decision']=='retain_tested_combination' and decision['reason']
            assert decision['quality_tradeoffs_disclosed'] and decision['other_lengths_disclosed']
            result['batches'][str(batch)].update(migration_accepted=True,decision=decision)
    if not result['missing']:result['status']='all_seven_migration_evidence_complete_review_pending'
    if review:
        assert not result['missing'],'Cannot accept missing batches'
        result.update(status='all_seven_migrations_accepted_optimization_pending',review_sha256=sha(args.review))
    (REPORTS/'migration-baseline-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if review:(REPORTS/'migration-baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'verified_batches':list(result['batches']),'missing':result['missing']}))


if __name__=='__main__':main()
