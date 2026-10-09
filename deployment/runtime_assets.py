"""Verify hash-pinned runtime-only assets without weakening build-input reuse."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_engine(directory,batch):
    directory=Path(directory)
    if (directory/'model.onnx').is_file():
        from deployment.multibatch.matrix import verified_engine
        return verified_engine(directory,batch)
    registry=json.loads((ROOT/'configs/hardware/sm89/indextts/assets.json').read_text())
    manifest=ROOT/'deployment/publication/runtime-manifest.json'
    if digest(manifest)!=registry['manifest_sha256']:raise ValueError('Published asset manifest hash mismatch')
    data=json.loads(manifest.read_text())
    if data.get('kind')!='index_sm89_verified_runtime_assets' or not data.get('runtime_only'):raise ValueError('Expected verified runtime-only bundle')
    for name in ('model.plan.json','model.engine','model.inspector.json'):
        path=directory/name;record=data['files'][path.relative_to(ROOT).as_posix()]
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:raise ValueError('Published runtime asset mismatch: '+name)
    plan=json.loads((directory/'model.plan.json').read_text())
    if plan['batch']!=batch or plan['sm']!=89 or plan['optimization_level']!=5 or plan['tiling_optimization_level']!='full' or plan['max_num_tactics']!=2147483646:raise ValueError('Published build policy mismatch')
    record=next(r for r in data['validated_builds'] if r['plan']==(directory/'model.plan.json').relative_to(ROOT).as_posix())
    if not record['build_input_bindings_verified'] or record['engine_sha256']!=plan['sha256']:raise ValueError('Published source validation receipt mismatch')
    return True
