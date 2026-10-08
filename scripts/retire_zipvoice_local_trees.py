"""Delete explicit old ZipVoice trees after final published source/assets validate."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from run_zipvoice_validation import ROOT,sha
TARGETS=(Path('/workspace/zvoice_temp'),Path('/workspace/inspark_proper_format'),Path('/workspace/zip_pipeline'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--published-validation',type=Path,required=True);p.add_argument('--apply',action='store_true');args=p.parse_args()
    proof=args.published_validation.resolve();proof.relative_to(ROOT);validation=json.loads(proof.read_text())
    assert validation['status']=='published_github_source_and_final_private_hub_revision_fresh_validated'
    registry=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text());assert registry['revision']==validation['hub_revision']
    assert set(validation['batches'])==set(registry['bundles'])=={'1','2','4','8','16','32','64'}
    env={**os.environ,'GIT_CONFIG_GLOBAL':str(ROOT/'.cache/gitconfig'),'GIT_ASKPASS':str(ROOT/'.cache/github/askpass.py'),'GIT_TERMINAL_PROMPT':'0'}
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,env=env,text=True).split()[0]
    # A later receipt-only commit may follow the tested source; the caller must revalidate identity.
    assert remote==validation['github_commit'],'Reconcile the actual remote before deleting originals'
    plan=json.loads((ROOT/'reports/sm89/zipvoice/a1007/local-retirement-plan.json').read_text());assert set(plan['local_old_trees'])==set(map(str,TARGETS))
    processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines();busy=[]
    own=os.getpid()
    for line in processes:
        fields=line.strip().split(None,1)
        if len(fields)==2 and int(fields[0])!=own and any(str(root) in fields[1] for root in TARGETS):busy.append({'pid':int(fields[0]),'reason':'Command references old tree'})
    pids=[line.strip().split(None,1)[0] for line in processes if line.strip()]
    observed=subprocess.run(['pwdx',*pids],capture_output=True,text=True)
    for line in observed.stdout.splitlines():
        pid,_,cwd=line.partition(': ')
        if pid.isdigit() and any(cwd==str(root) or cwd.startswith(str(root)+'/') for root in TARGETS):busy.append({'pid':int(pid),'reason':'Current directory under old tree'})
    assert not busy,busy
    # The old code checkout's binary artifact tree was only ZipVoice; recheck rather than trust age.
    old=Path('/workspace/inspark_proper_format/artifacts')
    manifests=[]
    if old.exists():
        for path in old.rglob('manifest.json'):
            if path.is_symlink():continue
            data=json.loads(path.read_text());assert data.get('model')=='zipvoice',f'Preserve non-ZipVoice assets at {path}'
            manifests.append(str(path))
    record={'status':'prepared_not_deleted','targets':[str(x) for x in TARGETS if x.exists()],'published_validation_sha256':sha(proof),'github_commit':remote,'hub_revision':registry['revision'],'observed_old_tree_processes':busy,'old_binary_manifests':manifests,'preserved':['/workspace/A_1007','/workspace/index-tts','/workspace/.codex','xirr/index_pipeline','current IndexTTS2 source and retained ancestry'],'scope_limit':'Current PID namespace only; no kill/reset or external context changes'}
    report=ROOT/'outputs/publication/local-tree-retirement.json';report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(record,indent=2)+'\n')
    if args.apply:
        for target in TARGETS:
            assert target.parent==Path('/workspace') and target!=ROOT and not target.is_symlink()
            if target.exists():shutil.rmtree(target)
        assert all(not root.exists() for root in TARGETS)
        record['status']='explicit_old_zipvoice_local_trees_removed';report.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'targets':record['targets']}))


if __name__=='__main__':main()
