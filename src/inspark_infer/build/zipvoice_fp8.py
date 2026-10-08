"""Independent, hash-bound SM120 FP8 bundles; legacy SM89 remains untouched."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import uuid

from inspark_infer.build.zipvoice import safe_path, gpu_info, check_workload
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES, root, sha


def registry():
    return json.loads((root()/'configs/hardware/sm120/zipvoice_fp8_registry.json').read_text())


def validate_bundle(bundle, batch=None):
    bundle=Path(bundle).resolve()
    m=json.loads((bundle/'manifest.json').read_text())
    if m.get('schema')!=1 or m.get('model')!='zipvoice' or m.get('precision')!='fp8' or m.get('batch') not in BATCHES:
        raise ValueError('Unsupported ZipVoice FP8 bundle')
    if batch is not None and m['batch']!=batch:raise ValueError('Wrong FP8 bundle batch')
    if set(m.get('engines',{}))!={'fm','text','unique','vocos'}:raise ValueError('Incomplete FP8 components')
    if m['hardware']['sm']!=120 or m['quantization']['format']!='E4M3FN':raise ValueError('Wrong FP8 recipe/hardware')
    for name,item in m['files'].items():
        p=safe_path(bundle,name)
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('FP8 bundle integrity failure: '+name)
    for key,e in m['engines'].items():
        if e['path'] not in m['files'] or m['files'][e['path']]['sha256']!=e['sha256'] or not e.get('shape_profile'):
            raise ValueError('Unbound FP8 engine: '+key)
    for name,digest in m['runtime_sources'].items():
        if sha(safe_path(root(),name))!=digest:raise ValueError('FP8 runtime source mismatch: '+name)
    application=m.get('application','route')
    packages=m.get('plugin_packages',[])
    prefix=f"inspark_infer.ops.tensorrt.zipvoice_fp8.b{m['batch']}."
    allowed=[prefix+x for x in ('normal_tf32_plugin','online_nonlinear_rna_plugin')]
    geometry=[prefix+'geo.'+x for x in ('normal_tf32_plugin','online_nonlinear_rna_plugin')]
    if application not in ('route','attention_route') or (application=='route' and packages) or (application=='attention_route' and packages not in (allowed,geometry)):
        raise ValueError('Unrecognized FP8 plugin application')
    for package in packages:
        name='src/'+package.replace('.','/')+'.py'
        if name not in m['runtime_sources']:raise ValueError('Unbound FP8 plugin source')
    if m['component_precisions']['text']!='fp32' or m['component_precisions']['vocos']!='fp32':
        raise ValueError('Unexpected text/Vocos precision')
    return m


def compatible(m,g):
    if any(g[k]!=m['hardware'][k] for k in ('name','sm','memory_total_mib')):
        raise ValueError('FP8 engine hardware mismatch; rebuild and validate for this GPU')


def ensure(batch,gpu=None,output_root=None):
    if batch not in BATCHES:raise ValueError(f'Supported FP8 batches: {BATCHES}')
    reg=registry();entry=reg['bundles'][str(batch)]
    bundle=safe_path(root(),entry['local_path']) if output_root is None else Path(output_root).resolve()/f'b{batch}'/entry['bundle_id']
    if not bundle.is_relative_to('/workspace'):raise ValueError('Storage must remain in /workspace')
    if gpu is not None:
        g=gpu_info(gpu)
        if g['sm']!=120:raise ValueError('FP8 SM120 route requires SM120 hardware')
    if not bundle.exists():
        from huggingface_hub import HfApi,hf_hub_download
        revision=reg.get('revision')
        if not isinstance(revision,str) or len(revision)!=40:raise FileNotFoundError('No published pinned FP8 revision')
        token=os.getenv('HF_TOKEN')
        token_file=Path(os.getenv('HF_HOME',str(root()/'.cache/huggingface')))/'token'
        if not token and token_file.is_file():token=token_file.read_text().strip()
        if not token:raise ValueError('Private HF credentials required')
        if HfApi(endpoint='https://huggingface.co',token=token).model_info(reg['repo_id'],revision=revision).private is not True:
            raise ValueError('FP8 assets must remain private')
        endpoint=os.getenv('HF_ENDPOINT','https://hf-mirror.com')
        def download(name):
            nonlocal endpoint
            try:
                p=hf_hub_download(repo_id=reg['repo_id'],revision=revision,filename=entry['bundle_path']+'/'+name,token=token,endpoint=endpoint)
            except Exception:
                if endpoint=='https://huggingface.co':raise
                endpoint='https://huggingface.co'
                p=hf_hub_download(repo_id=reg['repo_id'],revision=revision,filename=entry['bundle_path']+'/'+name,token=token,endpoint=endpoint)
            return Path(p)
        stage=bundle.parent/('.staging-'+uuid.uuid4().hex);stage.mkdir(parents=True)
        shutil.copyfile(download('manifest.json'),stage/'manifest.json')
        m=json.loads((stage/'manifest.json').read_text())
        if m['bundle_id']!=entry['bundle_id'] or m['batch']!=batch:raise ValueError('Pinned FP8 bundle identity mismatch')
        for name in m['files']:
            dst=safe_path(stage,name);dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(download(name),dst)
        validate_bundle(stage,batch);stage.rename(bundle)
    m=validate_bundle(bundle,batch)
    if m['bundle_id']!=entry['bundle_id']:raise ValueError('FP8 registry identity mismatch')
    if gpu is not None:compatible(m,g)
    return bundle,m
