"""Additive private runtime publication; weights and existing releases preserved."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--token-file',type=Path);ap.add_argument('--upload',action='store_true');a=ap.parse_args()
    source=a.source.resolve();stage=a.stage.resolve();stage.mkdir(parents=True,exist_ok=True)
    prefix='engines/sm89/int8_smoothquant/draft900_cfm800/v1';files={};references={};receipts=[]
    sys.path[:0]=[str(source/'src'),str(source/'scripts'),str(source)]
    from deployment.multibatch.matrix import verified_engine
    def add(path,name=None,raw=False):
        path=Path(path);name=name or path.relative_to(source).as_posix()
        if name in files:return
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.json' and not raw:
            data=json.loads(path.read_text())
            def portable(v):
                if isinstance(v,str) and v.startswith(str(source)+'/'):return 'bundle://'+v[len(str(source))+1:]
                if isinstance(v,list):return [portable(x) for x in v]
                if isinstance(v,dict):return {k:portable(x) for k,x in v.items()}
                return v
            target.write_text(json.dumps(portable(data),indent=2)+'\n')
        else:
            if target.exists():target.unlink()
            try:os.link(path,target)
            except OSError:shutil.copy2(path,target)
        files[name]={'sha256':sha(target),'bytes':target.stat().st_size}
    plans=set()
    for b in (1,2,4,8,16,32,64,128):
        p=source/f'configs/hardware/sm89/indextts/int8_b{b}_selected.json';data=json.loads(p.read_text());add(p)
        for k,v in data.items():
            if k.endswith('_plan') and isinstance(v,str):plans.add(Path(v))
        for path in [data['calibration'],*data.get('component_calibrations',{}).values()]:add(Path(path),raw=True)
        for path in Path(data['official_sources']).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json'):add(path)
    for c in ('target','draft'):plans.add(source/f'artifacts/sm89/int8_smoothquant/b48/{c}/model.plan.json')
    for p in sorted(plans):
        d=json.loads(p.read_text());assert verified_engine(p.parent,d['batch'])
        engine=Path(d['engine']);engine=engine if engine.is_absolute() else p.parent/engine
        add(p);add(engine);add(p.with_name('model.inspector.json'))
        receipts.append({'plan':p.relative_to(source).as_posix(),'batch':d['batch'],'engine_sha256':d['sha256'],'build_input_bindings_verified':True,'onnx_binding':d['provenance']['onnx_binding']})
    private=['deployment/history/validation-manifest.json','deployment/multibatch/history/validation-manifest-128.json',
        'deployment/ready_pipeline/history/confirmation.json','deployment/ready_pipeline/history/retained-final.json',
        'deployment/ready_pipeline/history/delivery-check.json','deployment/ready_pipeline/history/paired-CD.json',
        'deployment/ready_pipeline/history/priority-CD.json','deployment/ready_pipeline/history/trace-lifecycle.json']
    for name in private:add(source/name,raw=True)
    for path in (source/'deployment').rglob('history/**/*.json'):
        add(path, raw=True)
    for name in private[:2]:
        for ref in json.loads((source/name).read_text())['references']:references[ref['voice_id']]=ref
    for name,ref in references.items():
        assert sha(ref['path'])==ref['sha256'];add(Path(ref['path']),'deployment/publication/references/'+name,raw=True)
    marker=stage/'.bundle_root';marker.write_text('Index SM89 runtime bundle v1\n');files['.bundle_root']={'sha256':sha(marker),'bytes':marker.stat().st_size}
    manifest={'schema':1,'kind':'index_sm89_verified_runtime_assets','runtime_only':True,'batches':[1,2,4,8,16,32,64,128],'internal_ar_batches':[48],
        'source_weights_revision':'96933af87262eec17d685050723d4f768d2f8778','source_calibration_revision':'2edb087d47eefc95c0c008c71e27c21a1158c896','hardware':{'sm':89,'gpu_name':'NVIDIA GeForce RTX 4090'},'files':files,'validated_builds':receipts,
        'scope':'Selected runtime engine/plan/inspector/calibration/source closure; original ONNX bindings validated before packaging. ONNX rebuild inputs not included; no relaxed build reuse.'}
    m=stage/'manifest.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    code=Path(__file__).resolve().parents[1];out=code/'deployment/publication/assets-publication.json'
    record={'status':'prepared','repo':'xirr/index_pipeline','prefix':prefix,'files':len(files)+1,'bytes':sum(x['bytes'] for x in files.values()),'manifest_sha256':sha(m),'weights_changed':False,'deletions':[]}
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    if not a.upload:return
    from huggingface_hub import HfApi,CommitOperationAdd
    token=os.environ.get('HF_TOKEN') or a.token_file.read_text().strip();api=HfApi(endpoint='https://huggingface.co',token=token)
    before=api.model_info(record['repo'],files_metadata=True);assert before.private
    expected={prefix+'/'+name:item for name,item in files.items()};expected[prefix+'/manifest.json']={'sha256':sha(m),'bytes':m.stat().st_size}
    remote={x.rfilename:x for x in before.siblings}
    operations=[]
    for name,item in expected.items():
        existing=remote.get(name)
        if existing is not None:
            digest=(existing.lfs.sha256 if hasattr(existing.lfs,'sha256') else existing.lfs.get('sha256')) if existing.lfs else None
            if existing.size==item['bytes'] and digest==item['sha256']:continue
        operations.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(stage/name[len(prefix)+1:])))
    record.update(previous_revision=before.sha,upload_files=len(operations));out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'event':'upload_start','files':len(operations),'GB':record['bytes']/1e9}),flush=True)
    commit=api.create_commit(record['repo'],operations=operations,parent_commit=before.sha,num_threads=4,commit_message='Add validated SM89 Index INT8 batches and readiness-pipeline B48 AR assets')
    record.update(status='uploaded_remote_verification_pending',revision=commit.oid);out.write_text(json.dumps(record,indent=2)+'\n')
    info=api.model_info(record['repo'],revision=commit.oid,files_metadata=True);assert info.private;remote={x.rfilename:x for x in info.siblings}
    for name,item in expected.items():
        r=remote[name];assert r.size==item['bytes'],name
        if r.lfs:
            digest=r.lfs.sha256 if hasattr(r.lfs,'sha256') else r.lfs['sha256'];assert digest==item['sha256'],name
        else:
            body=(stage/name[len(prefix)+1:]).read_bytes();assert r.blob_id==hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest(),name
    record['status']='private_runtime_assets_uploaded_and_remote_hash_verified';out.write_text(json.dumps(record,indent=2)+'\n')
    registry={k:record[k] for k in ('repo','prefix','revision','manifest_sha256')};registry.update(schema=1,status='private_runtime_assets_remote_verified',weights_revision=manifest['source_weights_revision'],source_revision=manifest['source_calibration_revision'],batches=manifest['batches'],internal_ar_batches=[48],runtime_only=True)
    (code/'configs/hardware/sm89/indextts/assets.json').write_text(json.dumps(registry,indent=2)+'\n');print(json.dumps({'status':record['status'],'revision':commit.oid}),flush=True)

if __name__=='__main__':main()
