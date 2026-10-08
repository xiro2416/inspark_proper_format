"""Download and verify the source weights/graphs pinned by the A_1007 registry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'models/zipvoice')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'.cache/huggingface/hub')
    args=parser.parse_args()
    from inspark_infer.build.zipvoice import safe_path
    from huggingface_hub import hf_hub_download
    registry=json.loads((ROOT/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())
    revision=registry.get('revision')
    if not revision or len(revision)!=40:raise RuntimeError('No accepted published source revision is pinned yet')
    output=args.output.resolve()
    if not output.is_relative_to('/workspace'):raise ValueError('Source assets must stay under /workspace')
    cache=args.cache_dir.resolve()
    if not cache.is_relative_to('/workspace'):raise ValueError('Download cache must stay under /workspace')
    tokenfile=Path(os.getenv('HF_HOME',str(ROOT/'.cache/huggingface')))/'token'
    token=os.getenv('HF_TOKEN') or (tokenfile.read_text().strip() if tokenfile.is_file() else None)
    if not token:raise RuntimeError('Private repository credentials are required')
    endpoint=os.getenv('HF_ENDPOINT','https://hf-mirror.com')
    def download(name):
        nonlocal endpoint
        try:
            return Path(hf_hub_download(repo_id=registry['repo_id'],revision=revision,filename=name,token=token,endpoint=endpoint,cache_dir=cache))
        except Exception:
            if endpoint=='https://huggingface.co':raise
            endpoint='https://huggingface.co'
            return Path(hf_hub_download(repo_id=registry['repo_id'],revision=revision,filename=name,token=token,endpoint=endpoint,cache_dir=cache))
    manifest=json.loads(download('weights-manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    for item in manifest['files']:
        path=safe_path(output,item['path'])
        if path.is_file() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']:continue
        source=download(item['path'])
        if source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:raise ValueError(f'Weight integrity failure: {item["path"]}')
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,path)
    (output/'weights-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'source_assets_verified','revision':revision,'files':len(manifest['files']),'endpoint':endpoint}))

if __name__=='__main__':main()
