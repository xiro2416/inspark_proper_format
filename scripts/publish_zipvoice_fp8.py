"""Add accepted FP8 assets to private HF; preserve existing files and history."""
import argparse
import hashlib
import logging
import re
import json
import os
from pathlib import Path
import sys

def redact_urls(value):
    return re.sub(r'(https?://[^\s"<>?]+)\?[^\s"<>]*',r'\1?[redacted]',value)

class PrivateTransferLogFilter(logging.Filter):
    def filter(self,record):
        record.msg=redact_urls(record.getMessage());record.args=()
        return True

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,sha,write
from inspark_infer.build.zipvoice_fp8 import validate_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-card',type=Path,required=True);p.add_argument('--verify-revision')
    args=p.parse_args()
    from huggingface_hub import HfApi,CommitOperationAdd
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    for handler in logging.getLogger('huggingface_hub').handlers:handler.addFilter(PrivateTransferLogFilter())
    token=os.getenv('HF_TOKEN')
    if not token:raise ValueError('Inject authorized private HF credential via HF_TOKEN')
    api=HfApi(endpoint='https://huggingface.co',token=token)
    reg_path=ROOT/'configs/hardware/sm120/zipvoice_fp8_registry.json'
    reg=json.loads(reg_path.read_text())
    if reg['status']!='accepted_local_publication_pending' or set(reg['bundles'])!=set(map(str,BATCHES)):
        raise ValueError('Seven accepted packaged targets required before upload')
    receipt=ROOT/'outputs/fp8/publication/hf-publication.json'
    parent=json.loads(receipt.read_text())['parent_revision'] if args.verify_revision else None
    info=api.model_info(reg['repo_id'],revision=parent,files_metadata=True)
    if not info.private:raise ValueError('Asset repository must be private')
    assets={}
    for batch,entry in reg['bundles'].items():
        bundle=ROOT/entry['local_path'];m=validate_bundle(bundle,int(batch))
        for name in [*m['files'],'manifest.json']:assets[entry['bundle_path']+'/'+name]=bundle/name
    for name in ['model.safetensors','quantization.json']:
        assets['fp8/sm120/'+name]=ROOT/'models/zipvoice/fp8'/name
    graph=ROOT/'models/zipvoice/fp8/fm.onnx'
    import onnx
    metadata=onnx.load(graph,load_external_data=False)
    locations={x.value for t in metadata.graph.initializer for x in t.external_data if x.key=='location'}
    assets['onnx/sm120/fp8/fm.onnx']=graph
    for name in locations:
        path=(graph.parent/name).resolve()
        if path.parent!=graph.parent.resolve():raise ValueError('Unsafe external graph path')
        assets['onnx/sm120/fp8/'+name]=path
    # README supplied by the caller must integrate, rather than erase, the
    # current SM89 card. The upload is additive and uses the actual parent SHA.
    card=args.model_card.resolve()
    if not card.is_relative_to(ROOT):raise ValueError('Model card must be a checkout-owned file')
    assets['README.md']=card
    expected={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in assets.items()}
    old_metadata={x.rfilename:x for x in info.siblings}
    for name in assets:
        if name in old_metadata and name!='README.md':
            remote=old_metadata[name]
            lfs=remote.lfs
            digest=(lfs['sha256'] if isinstance(lfs,dict) else lfs.sha256) if lfs else None
            if lfs:
                identical=digest==expected[name]['sha256']
            else:
                data=assets[name].read_bytes();identical=remote.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if not identical:raise ValueError('Additive publication would overwrite existing asset: '+name)
    operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in assets.items()]
    receipt=ROOT/'outputs/fp8/publication/hf-publication.json'
    write(receipt,dict(status='prepared',repo_id=reg['repo_id'],parent_revision=info.sha,files=expected,deletions=[]))
    if args.verify_revision:
        revision=args.verify_revision
    else:
        commit=api.create_commit(reg['repo_id'],operations=operations,parent_commit=info.sha,num_threads=4,
                                commit_message='Add validated ZipVoice SM120 FP8 batches 1 2 4 8 16 32 64')
        revision=commit.oid
    write(receipt,dict(status='uploaded_verification_pending',repo_id=reg['repo_id'],parent_revision=info.sha,
                       revision=revision,files=expected,deletions=[]))
    after=api.model_info(reg['repo_id'],revision=revision,files_metadata=True)
    actual={x.rfilename:x for x in after.siblings}
    old={x.rfilename for x in info.siblings}
    if not old<=set(actual) or not after.private:raise RuntimeError('Existing assets or private status changed')
    from huggingface_hub import hf_hub_download
    def attributes(revision):
        return Path(hf_hub_download(reg['repo_id'],'.gitattributes',revision=revision,token=token,
                                   endpoint='https://huggingface.co')).read_text().splitlines()
    before_attributes=attributes(info.sha);after_attributes=attributes(revision)
    if after_attributes[:len(before_attributes)]!=before_attributes:
        raise RuntimeError('Existing LFS attributes changed')
    added_attributes=after_attributes[len(before_attributes):]
    for line in added_attributes:
        if not line.strip():continue
        path,rule=line.split(maxsplit=1)
        if path not in assets or path in old_metadata or rule!='filter=lfs diff=lfs merge=lfs -text':
            raise RuntimeError('Unexpected LFS attribute addition')
    for name,previous in old_metadata.items():
        if name in assets or name=='.gitattributes':continue
        if actual[name].blob_id!=previous.blob_id or actual[name].size!=previous.size:
            raise RuntimeError('Existing asset identity changed: '+name)
    for name,item in expected.items():
        remote=actual[name]
        if remote.size!=item['bytes']:raise ValueError('Remote size mismatch: '+name)
        if remote.lfs:
            digest=remote.lfs['sha256'] if isinstance(remote.lfs,dict) else remote.lfs.sha256
            if digest!=item['sha256']:raise ValueError('Remote LFS hash mismatch: '+name)
        else:
            data=assets[name].read_bytes();digest=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            if digest!=remote.blob_id:raise ValueError('Remote Git blob mismatch: '+name)
    reg.update(revision=revision,status='remote_hashes_verified_fresh_download_pending');write(reg_path,reg)
    write(receipt,dict(status=reg['status'],repo_id=reg['repo_id'],parent_revision=info.sha,revision=revision,
                       files=expected,deletions=[],existing_paths_preserved=True,gitattributes_additions=added_attributes))
    print(json.dumps(dict(status=reg['status'],revision=revision,files=len(expected))))


if __name__=='__main__':
    try:main()
    except Exception as e:raise RuntimeError(redact_urls(str(e))) from None
