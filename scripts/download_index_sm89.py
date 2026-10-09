"""Download immutable private SM89 runtime closure into a fresh checkout."""
import argparse,concurrent.futures,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--token-file',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    from huggingface_hub import hf_hub_download
    reg=json.loads((ROOT/'configs/hardware/sm89/indextts/assets.json').read_text())
    token=os.environ.get('HF_TOKEN') or (a.token_file.read_text().strip() if a.token_file else (ROOT.parent/'.cache/huggingface/token').read_text().strip())
    common=dict(repo_id=reg['repo'],revision=reg['revision'],token=token,endpoint='https://huggingface.co',cache_dir=str(ROOT/'.cache/huggingface'))
    manifest=Path(hf_hub_download(filename=reg['prefix']+'/manifest.json',**common))
    if sha(manifest)!=reg['manifest_sha256']:raise ValueError('Registry/manifest mismatch')
    data=json.loads(manifest.read_text());assert data['source_weights_revision']==reg['weights_revision']
    def one(item):
        name,record=item;target=ROOT/name
        if not target.resolve().is_relative_to(ROOT):raise ValueError('Bundle path traversal')
        if target.exists():
            if target.stat().st_size==record['bytes'] and sha(target)==record['sha256']:return dict(name=name, downloaded=False)
            raise ValueError('Refuse to replace different existing asset: '+name)
        downloaded=Path(hf_hub_download(filename=reg['prefix']+'/'+name,**common)).resolve(strict=True)
        if downloaded.stat().st_size!=record['bytes'] or sha(downloaded)!=record['sha256']:raise ValueError('Downloaded hash mismatch: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(downloaded,target)
        except OSError:shutil.copy2(downloaded,target)
        if target.stat().st_size!=record['bytes'] or sha(target)!=record['sha256']:raise ValueError('Materialized hash mismatch: '+name)
        return dict(name=name, downloaded=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(one,data['files'].items()))
    destination=ROOT/'deployment/publication/runtime-manifest.json';destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(manifest,destination)
    out=ROOT/'outputs/publication/index-download.json';out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='all_runtime_files_hash_verified', downloaded_files=sum(r['downloaded'] for r in results), downloaded_engines=sum(r['downloaded'] and r['name'].endswith('.engine') for r in results),repo=reg['repo'],revision=reg['revision'],files=len(data['files']),bytes=sum(x['bytes'] for x in data['files'].values()),weights_revision=reg['weights_revision'])
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
