"""Publish accepted A_1007 tree with reviewed Index ancestry and an explicit lease."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
from run_zipvoice_validation import ROOT,sha
REPORTS=ROOT/'reports/sm89/zipvoice/a1007'
INDEX_ROOTS=('configs/current','reports/current','api/release','runtime/unified_deployment','benchmarks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-main',required=True);p.add_argument('--fresh-validation',type=Path);p.add_argument('--apply',action='store_true');args=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.expected_main)
    env={**os.environ,'GIT_CONFIG_GLOBAL':str(ROOT/'.cache/gitconfig'),'GIT_ASKPASS':str(ROOT/'.cache/github/askpass.py'),'GIT_TERMINAL_PROMPT':'0'}
    def git(*command):return subprocess.check_output(['git',*command],cwd=ROOT,env=env,text=True).strip()
    current=git('ls-remote','origin','refs/heads/main').split()[0];assert current==args.expected_main,'Remote moved; reconcile latest Index before publishing'
    parent=json.loads((REPORTS/'github-history-preparation.json').read_text())['rewritten_index_commit']
    assert not git('ls-tree','-r','--name-only',parent,'--','src/inspark_infer/ops/tensorrt/zipvoice','src/inspark_infer/runtime/zipvoice')
    subprocess.run(['git','merge-base','--is-ancestor','412039fed2ec20c0cc450437719ff11ea7d1bcad',parent],cwd=ROOT,env=env,check=True)
    subprocess.run(['git','diff','--exit-code',current,'HEAD','--',*INDEX_ROOTS],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
    record={'status':'prepared_not_pushed','expected_previous_main':current,'clean_index_parent':parent,'index_preserved_roots':list(INDEX_ROOTS),'local_head':git('rev-parse','HEAD'),'local_tree':git('rev-parse','HEAD^{tree}'),'gate':'All7 accepted optimized bundles and current private asset fresh validation, retired103legacy paths, clean committed tree'}
    out=ROOT/'outputs/publication/github-publication.json';out.parent.mkdir(parents=True,exist_ok=True)
    if not args.apply:
        out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'previous_main':current}));return
    assert args.fresh_validation,'--apply requires actual fresh private asset validation'
    fresh=args.fresh_validation.resolve();fresh.relative_to(ROOT);v=json.loads(fresh.read_text())
    reg=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text());publication=json.loads((REPORTS/'publication-assets.json').read_text())
    assert v['status']=='all_batches_and_weights_fresh_download_validated' and set(v['batches'])==set(reg['bundles'])=={'1','2','4','8','16','32','64'}
    assert v['revision']==publication['revision']
    if v['revision']!=reg['revision']:
        cleanup=json.loads((REPORTS/'hub-cleanup.json').read_text())
        assert cleanup['status']=='obsolete_paths_history_lfs_removed_final_download_pending' and cleanup['final_revision']==reg['revision'] and cleanup['fresh_validation_sha256']==sha(fresh)
    assert not git('status','--porcelain'),'Commit reviewed code/reports before publishing'
    retired=json.loads((REPORTS/'local-retirement-plan.json').read_text())['retired_tracked_paths'];assert all(not (ROOT/name).exists() for name in retired)
    sys.path.insert(0,str(ROOT/'src'));from inspark_infer.build.zipvoice import safe_path,validate_bundle
    for batch,entry in reg['bundles'].items():
        manifest=validate_bundle(safe_path(ROOT,entry['local_path']),int(batch));assert v['batches'][batch]['bundle_id']==manifest['bundle_id']
    secret=re.compile(r'(?:hf_|ghp_)[A-Za-z0-9]{20,}')
    for name in git('ls-files').splitlines():
        path=ROOT/name
        if path.is_file():assert not secret.search(path.read_text(errors='ignore')),f'Credential-shaped content found in tracked {name}'
    commit=subprocess.check_output(['git','commit-tree',record['local_tree'],'-p',parent],input='Publish validated ZipVoice A_1007 INT8 profiles and preserve current IndexTTS2\n',cwd=ROOT,env=env,text=True).strip()
    record.update(status='clean_release_commit_prepared',release_commit=commit,fresh_validation_sha256=sha(fresh));out.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run(['git','push',f'--force-with-lease=refs/heads/main:{current}','origin',f'{commit}:refs/heads/main'],cwd=ROOT,env=env,check=True)
    assert git('ls-remote','origin','refs/heads/main').split()[0]==commit
    record['status']='clean_release_main_pushed_fresh_github_validation_pending';out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'release_commit':commit,'receipt':str(out)}))


if __name__=='__main__':main()
