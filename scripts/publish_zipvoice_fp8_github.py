"""Normal fast-forward GitHub publication after accepted private fresh downloads."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh-validation',type=Path,required=True)
    p.add_argument('--expected-main',required=True)
    args=p.parse_args()
    proof=json.loads(args.fresh_validation.read_text())
    reg=json.loads((ROOT/'configs/hardware/sm120/zipvoice_fp8_registry.json').read_text())
    if proof['status']!='all_seven_fresh_fp8_bundles_validated' or proof['revision']!=reg['revision']:
        raise ValueError('Accepted current private fresh-download validation required')
    credentials=Path('/workspace/.codex/github-credentials').read_text()
    match=re.search(r'(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+)',credentials)
    if not match:raise ValueError('Workspace GitHub credential format not recognized')
    token=match.group(0)
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'InSpark-FP8-Publisher'}
    with urllib.request.urlopen(urllib.request.Request('https://api.github.com/user',headers=headers),timeout=30) as response:
        user=json.load(response)
    with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/xiro2416/inspark_proper_format',headers=headers),timeout=30) as response:
        repo=json.load(response)
    if not repo.get('permissions',{}).get('push'):raise ValueError('No repository write permission')
    # Askpass contains only a path reader, never a credential literal.
    askpass=ROOT/'.cache/github/askpass.py';askpass.parent.mkdir(parents=True,exist_ok=True)
    askpass.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport re,sys\nif 'Username' in sys.argv[1]: print('x-access-token')\nelse:\n s=Path('/workspace/.codex/github-credentials').read_text();m=re.search(r'(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+)',s);print(m.group(0))\n")
    askpass.chmod(0o700)
    env={**os.environ,'GIT_ASKPASS':str(askpass),'GIT_TERMINAL_PROMPT':'0'}
    def git(*values):return subprocess.check_output(['git',*values],cwd=ROOT,env=env,text=True).strip()
    remote=git('ls-remote','origin','refs/heads/main').split()[0]
    if remote!=args.expected_main:raise RuntimeError('Remote main changed; reconcile concurrent edits before publishing')
    if git('status','--porcelain'):raise RuntimeError('Commit all reviewed source/docs/public reports before publication')
    subprocess.run(['git','merge-base','--is-ancestor',remote,'HEAD'],cwd=ROOT,env=env,check=True)
    subprocess.run(['git','diff','--exit-code',remote,'HEAD','--','configs/current','reports/current',
                    'src/inspark_infer/models/indextts2','src/inspark_infer/runtime/indextts2'],cwd=ROOT,env=env,check=True)
    # The main dispatcher has intentional FP8-only additions; legacy closures
    # themselves must remain unchanged for their existing hash-bound bundles.
    subprocess.run(['git','diff','--exit-code',remote,'HEAD','--','src/inspark_infer/runtime/zipvoice',
                    'src/inspark_infer/build/zipvoice.py','configs/hardware/sm89'],cwd=ROOT,env=env,check=True)
    secret=re.compile(r'(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')
    for name in git('ls-files').splitlines():
        path=ROOT/name
        if path.is_file() and secret.search(path.read_text(errors='ignore')):raise ValueError('Credential-shaped content in tracked file: '+name)
    head=git('rev-parse','HEAD')
    subprocess.run(['git','push','origin','HEAD:refs/heads/main'],cwd=ROOT,env=env,check=True)
    if git('ls-remote','origin','refs/heads/main').split()[0]!=head:raise RuntimeError('Published GitHub head differs')
    out=ROOT/'outputs/fp8/publication/github-publication.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status='normal_main_commit_published_fresh_checkout_pending',commit=head,
                   previous_main=remote,hf_revision=reg['revision'],authenticated_user=user['login'],history_rewritten=False),indent=2)+'\n')
    print(json.dumps(dict(status='normal_main_commit_published_fresh_checkout_pending',commit=head)))


if __name__=='__main__':main()
