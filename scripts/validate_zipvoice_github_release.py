"""Validate actual published GitHub source with a fresh pinned private Hub download."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
from run_zipvoice_validation import ROOT,environment,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--expected-runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',args.expected_commit)
    output=args.output.resolve();output.relative_to(ROOT);assert not output.exists(),'Use a fresh checkout and asset cache'
    expected_runs=args.expected_runs.resolve();expected_runs.relative_to(ROOT)
    output.mkdir(parents=True);checkout=output/'checkout';report=output/'validation.json'
    env=environment();env.update(GIT_CONFIG_GLOBAL=str(ROOT/'.cache/gitconfig'),GIT_ASKPASS=str(ROOT/'.cache/github/askpass.py'),GIT_TERMINAL_PROMPT='0')
    result={'status':'cloning_published_source','github_repo':'xiro2416/inspark_proper_format','expected_commit':args.expected_commit,'expected_runs_sha256':sha(expected_runs)}
    def save():report.write_text(json.dumps(result,indent=2)+'\n')
    save()
    with (output/'clone.log').open('w') as log:
        subprocess.run(['git','clone','--single-branch','--branch','main','https://github.com/xiro2416/inspark_proper_format.git',str(checkout)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,env=env,text=True).strip()
    assert commit==args.expected_commit,'Published main changed; reconcile before source validation'
    current=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
    published=json.loads((checkout/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
    assert current==published and len(published['revision'])==40
    result.update(status='fresh_published_code_hub_worker_validation_running',github_commit=commit,hub_revision=published['revision']);save()
    # Credentials are passed only in the subprocess environment, never argv/log/report.
    child={**env,'INSPARK_REPO_ROOT':str(checkout),'PYTHONPATH':str(checkout/'src'),'HF_TOKEN':(ROOT/'.cache/huggingface/token').read_text().strip(),
           'HF_HOME':str(checkout/'.cache/huggingface'),'XDG_CACHE_HOME':str(checkout/'.cache'),'TRITON_CACHE_DIR':str(checkout/'.cache/triton'),
           'CUDA_CACHE_PATH':str(checkout/'.cache/cuda'),'TORCH_HOME':str(checkout/'.cache/torch'),'TMPDIR':str(checkout/'.cache/tmp')}
    for folder in ('.cache/tmp','.cache/huggingface','.cache/triton','.cache/cuda'):(checkout/folder).mkdir(parents=True,exist_ok=True)
    imported=Path(subprocess.check_output([sys.executable,'-c','import inspark_infer; print(inspark_infer.__file__)'],cwd=checkout,env=child,text=True).strip()).resolve()
    assert imported.is_relative_to(checkout/'src'),'Python resolved the original checkout instead of fresh published code'
    result['imported_package_path']=str(imported);save()
    fresh=checkout/'outputs/final-private-download'
    with (output/'worker-validation.log').open('w') as log:
        subprocess.run([sys.executable,str(checkout/'scripts/validate_zipvoice_download.py'),'--expected-runs',str(expected_runs),'--output',str(fresh)],cwd=checkout,env=child,stdout=log,stderr=subprocess.STDOUT,check=True)
    validation=json.loads((fresh/'validation.json').read_text())
    assert validation['status']=='all_batches_and_weights_fresh_download_validated' and validation['revision']==published['revision']
    assert set(validation['batches'])=={'1','2','4','8','16','32','64'}
    result.update(status='published_github_source_and_final_private_hub_revision_fresh_validated',fresh_validation=str(fresh/'validation.json'),fresh_validation_sha256=sha(fresh/'validation.json'),batches=validation['batches'])
    save();print(json.dumps({'status':result['status'],'github_commit':commit,'hub_revision':published['revision'],'report':str(report)}))


if __name__=='__main__':main()
