"""Retire explicit legacy ZipVoice paths only after current fresh-download acceptance."""
import argparse,json,re,subprocess,sys
from pathlib import Path
from run_zipvoice_validation import ROOT,sha
REPORTS=ROOT/'reports/sm89/zipvoice/a1007'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fresh-validation',type=Path);p.add_argument('--apply',action='store_true');args=p.parse_args()
    plan=json.loads((REPORTS/'local-retirement-plan.json').read_text())
    paths=sorted(set(plan['retired_tracked_paths']))
    exact={'docs/zipvoice-int8.md','reports/sm89/zipvoice/acceptance.json','reports/sm89/zipvoice/download-acceptance.json','reports/sm89/zipvoice/source-migration.json','configs/hardware/sm89/zipvoice_int8_b24.json'}
    legacy=re.compile(r'^src/inspark_infer/(?:ops/tensorrt/zipvoice/b(?:16|24|32|64)/.+\.py|runtime/zipvoice/(?:routes/b|telemetry_b)(?:16|24|32|64)\.py)$')
    assert all(name in exact or legacy.fullmatch(name) for name in paths)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    present=[name for name in paths if name in tracked and (ROOT/name).exists()]
    assert not subprocess.check_output(['git','diff','--name-only','HEAD','--',*present],cwd=ROOT,text=True).strip(),'Legacy paths have unreviewed local edits'
    dependencies=[]
    references=re.compile(r'inspark_infer\.(?:ops\.tensorrt\.zipvoice\.b(?:16|24|32|64)|runtime\.zipvoice\.(?:routes\.b|telemetry_b)(?:16|24|32|64))\b')
    for directory in ('src','scripts','tests','configs'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.toml') and str(path.relative_to(ROOT)) not in paths:
                if references.search(path.read_text()):dependencies.append(str(path.relative_to(ROOT)))
    assert not dependencies,dependencies
    record={'status':'prepared_not_applied','paths':present,'before_sha256':{name:sha(ROOT/name) for name in present},'retained_dependency_scan_clear':True,'gate':'Accepted optimized registry and current private revision freshly downloaded through all7 public workers; no old source-tree deletion in this stage'}
    output=REPORTS/'legacy-code-retirement.json'
    if args.apply:
        assert args.fresh_validation,'--apply requires a completed fresh validation report'
        fresh=args.fresh_validation.resolve();fresh.relative_to(ROOT)
        validation=json.loads(fresh.read_text());registry=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
        assert validation['status']=='all_batches_and_weights_fresh_download_validated'
        assert set(validation['batches'])==set(registry['bundles'])=={'1','2','4','8','16','32','64'}
        assert validation['revision']==registry['revision'] and len(registry['revision'])==40
        sys.path.insert(0,str(ROOT/'src'))
        from inspark_infer.build.zipvoice import validate_bundle,safe_path
        for batch,entry in registry['bundles'].items():
            manifest=validate_bundle(safe_path(ROOT,entry['local_path']),int(batch))
            assert manifest['integration_validation']=='accepted_target_evidence_bound'
            assert not set(paths).intersection(manifest['runtime_sources'])
        record['fresh_validation_sha256']=sha(fresh)
        output.write_text(json.dumps(record,indent=2)+'\n')
        for name in present:(ROOT/name).unlink()
        assert all(not (ROOT/name).exists() for name in paths)
        record['status']='legacy_zipvoice_tracked_paths_removed_local_trees_preserved'
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'paths':len(present)}))


if __name__=='__main__':main()
