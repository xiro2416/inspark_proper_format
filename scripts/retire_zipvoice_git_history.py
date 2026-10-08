"""Align local main with the verified clean release and prune replaced Zip history."""
import argparse,json,os,subprocess
from pathlib import Path
from run_zipvoice_validation import ROOT,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--published-validation',type=Path,required=True);p.add_argument('--apply',action='store_true');args=p.parse_args()
    proof=args.published_validation.resolve();proof.relative_to(ROOT);v=json.loads(proof.read_text());assert v['status']=='published_github_source_and_final_private_hub_revision_fresh_validated'
    env={**os.environ,'GIT_CONFIG_GLOBAL':str(ROOT/'.cache/gitconfig'),'GIT_ASKPASS':str(ROOT/'.cache/github/askpass.py'),'GIT_TERMINAL_PROMPT':'0'}
    def git(*command):return subprocess.check_output(['git',*command],cwd=ROOT,env=env,text=True).strip()
    release=v['github_commit'];assert git('ls-remote','origin','refs/heads/main').split()[0]==release
    assert not git('status','--porcelain'),'Commit or reconcile metadata before history retirement'
    assert git('rev-parse','HEAD^{tree}')==git('rev-parse',release+'^{tree}'),'Preserve all current files; reconcile receipt-only commits separately'
    subprocess.run(['git','merge-base','--is-ancestor','412039fed2ec20c0cc450437719ff11ea7d1bcad',release],cwd=ROOT,env=env,check=True)
    old=['bb54a18','bd0d4b7','af3ace5','0282228'];remove=[]
    for line in git('for-each-ref','--format=%(refname) %(objectname)').splitlines():
        ref,commit=line.split()
        if ref in ('refs/heads/main','refs/remotes/origin/main','refs/remotes/origin/HEAD'):continue
        linked=any(subprocess.run(['git','merge-base','--is-ancestor',item,commit],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for item in old)
        if linked:
            assert ref.startswith('refs/heads/a1007-'),'Preserve unrelated branch; reconcile its Index content before cleanup'
            remove.append((ref,commit))
    output=ROOT/'outputs/publication/local-history-retirement.json';output.parent.mkdir(parents=True,exist_ok=True)
    record={'status':'prepared_not_applied','release_commit':release,'old_main':git('rev-parse','refs/heads/main'),'delete_task_refs':[r for r,_ in remove],'published_validation_sha256':sha(proof),'preserved_ancestor':'412039fed2ec20c0cc450437719ff11ea7d1bcad'}
    output.write_text(json.dumps(record,indent=2)+'\n')
    if args.apply:
        subprocess.run(['git','update-ref','refs/heads/main',release,record['old_main']],cwd=ROOT,env=env,check=True)
        subprocess.run(['git','fetch','--prune','origin'],cwd=ROOT,env=env,check=True)
        for ref,commit in remove:subprocess.run(['git','update-ref','-d',ref,commit],cwd=ROOT,env=env,check=True)
        subprocess.run(['git','reflog','expire','--expire=now','--expire-unreachable=now','--all'],cwd=ROOT,env=env,check=True)
        subprocess.run(['git','gc','--prune=now'],cwd=ROOT,env=env,check=True)
        reachable=git('rev-list','--all').splitlines()
        assert not any(any(commit.startswith(item) for item in old) for commit in reachable)
        assert git('rev-parse','HEAD')==release and not git('status','--porcelain')
        record['status']='local_main_clean_index_ancestry_preserved_old_zip_history_unreachable_pruned';output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'release_commit':release}))


if __name__=='__main__':main()
