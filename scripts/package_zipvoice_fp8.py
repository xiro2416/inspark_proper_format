"""Package only seven accepted FP8 targets with immutable source/evidence hashes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8.common import BATCHES,private_report,sha,write
from inspark_infer.build.zipvoice_fp8 import validate_bundle
from validate_zipvoice_fp8 import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    args=p.parse_args();selection=json.loads(args.selection.read_text())
    if selection['status']!='all_seven_fp8_migrated_optimized_validated' or set(selection['batches'])!=set(map(str,BATCHES)):
        raise ValueError('Explicit all-seven accepted selection required')
    import torch,tensorrt
    from inspark_infer.build.zipvoice import gpu_info
    hardware=gpu_info(3);hardware.pop('uuid');hardware.pop('physical_gpu')
    hardware.update(tensorrt=tensorrt.__version__,power_limit_w=600)
    recipe=json.loads((ROOT/'models/zipvoice/fp8/quantization.json').read_text())
    sources=[]
    for directory in ['runtime/zipvoice_fp8','models/zipvoice','ops/cuda/zipvoice','ops/tensorrt/zipvoice_fp8']:
        sources.extend((ROOT/'src/inspark_infer'/directory).rglob('*.py'))
    sources.extend(ROOT/'src/inspark_infer'/name for name in ['build/zipvoice_fp8.py','build/zipvoice.py',
        'runtime/device.py','runtime/zipvoice/telemetry.py','runtime/zipvoice/request_telemetry.py',
        'ops/tensorrt/zipvoice/engine.py'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))}
    registry=dict(schema=1,model='zipvoice',precision='fp8',repo_id='xirr/zip_pipeline',revision=None,
                  frame_profile=[600,760,920],token_profile=[52,78,141],status='accepted_local_publication_pending',bundles={})
    for batch in BATCHES:
        selected=selection['batches'][str(batch)]
        if not selected['migration_accepted'] or not selected['optimization_review_complete']:
            raise ValueError('Incomplete target acceptance')
        for name,digest in selected['evidence'].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Accepted evidence identity mismatch')
        inv=json.loads((ROOT/selected['inventory']).read_text())
        identity=dict(model='zipvoice',precision='fp8',batch=batch,hardware=hardware,
                      runtime_sources=hashes,application=inv.get('application','attention_route' if inv.get('plugin_packages') else 'route'),
                      plugin_packages=inv.get('plugin_packages',[]),quantization_sha256=sha(ROOT/'models/zipvoice/fp8/quantization.json'),
                      engines={k:v['sha256'] for k,v in inv['engines'].items()},pcm_policy=selected['pcm_policy'])
        import hashlib
        bundle_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
        bundle=ROOT/f'artifacts/trt113_bundles/zipvoice/sm120/fp8/b{batch}/{bundle_id}'
        files={};engines={}
        def copy(src,name,expected=None):
            src=Path(src).resolve()
            if not src.is_relative_to(ROOT) or (expected and sha(src)!=expected):raise ValueError('Asset source identity mismatch')
            dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            files[name]=dict(sha256=sha(dst),bytes=dst.stat().st_size)
        for key,engine in inv['engines'].items():
            copy(engine['path'],f'{key}/engine.plan',engine['sha256'])
            engines[key]=dict(path=f'{key}/engine.plan',sha256=engine['sha256'],shape_profile=engine['shape_profile'],source_sha256=engine['source_sha256'])
        for name in ['config/model.json','config/tokens.txt']:
            copy(ROOT/'models/zipvoice'/name,name)
        copy(ROOT/'models/zipvoice/fp8/quantization.json','quantization.json')
        for name in ['LICENSE','THIRD_PARTY_NOTICES.md','licenses/ZipVoice.txt','licenses/Vocos.txt']:
            copy(ROOT/name,name)
        m=dict(schema=1,**{k:v for k,v in identity.items() if k!='engines'},bundle_id=bundle_id,engines=engines,files=files,
               runtime=dict(torch=torch.__version__),component_precisions=inv['component_precisions'],
               quantization=dict(format='E4M3FN',method='static_max',weight_axis=None,activation_axis=None,
                                 first_floating_layers=4,last_fp8_layers=12,linear_modules=len(recipe['modules']),
                                 floating_depthwise_modules=len(recipe['coverage']['floating_convolutions'])),
               mapping_source_sha256=inv['origin_mapping_source_sha256'],
               validation_evidence=selected['evidence'],certified_for_production=False,
               integration_validation='accepted_target_evidence_bound')
        write(bundle/'manifest.json',m);validate_bundle(bundle,batch)
        entry=dict(bundle_id=bundle_id,local_path=str(bundle.relative_to(ROOT)),
                   bundle_path=f'bundles/zipvoice/sm120/fp8/b{batch}/{bundle_id}')
        registry['bundles'][str(batch)]=entry
        write(ROOT/f'configs/hardware/sm120/zipvoice_fp8_b{batch}.json',
              dict(schema=1,model='zipvoice',precision='fp8',batch=batch,registry='zipvoice_fp8_registry.json',bundle_id=bundle_id))
    write(ROOT/'configs/hardware/sm120/zipvoice_fp8_registry.json',registry)
    print(json.dumps(dict(status='seven_source_bound_fp8_bundles_packaged')))


if __name__=='__main__':main()
