"""Hash-check and sequentially deserialize every published engine on GPU1.

This tests runtime compatibility and context creation; it does not claim inference
correctness. Full computation validation remains in the private migration records
and readiness runner.
"""
import argparse
import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('Physical GPU1 only')
    from inspark_infer.runtime.device import select_gpu, GPULease
    select_gpu(1)
    import torch
    from deployment.runtime_assets import verify_engine, digest
    from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
    registry = json.loads((ROOT / 'configs/hardware/sm89/indextts/assets.json').read_text())
    manifest_file = ROOT / 'deployment/publication/runtime-manifest.json'
    if digest(manifest_file) != registry['manifest_sha256']:
        raise ValueError('Registry/manifest mismatch')
    manifest = json.loads(manifest_file.read_text())
    report = dict(status='incomplete', revision=registry['revision'], engines=[], physical_gpu=1,
                  scope='Sequential deserialization and execution-context creation; no enqueue or performance claim')
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    with GPULease(1):
        for build in manifest['validated_builds']:
            path = ROOT / build['plan']
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError('Engine plan escapes root')
            verify_engine(path.parent, build['batch'])
            engine = StaticEngine(path, build['batch'])
            report['engines'].append(dict(plan=build['plan'], batch=build['batch'],
                engine_sha256=engine.engine_sha256, context_created=True))
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            save()
            print(json.dumps(dict(loaded=len(report['engines']), total=len(manifest['validated_builds']))), flush=True)
        report.update(status='all_published_engines_deserialized', engine_count=len(report['engines']))
        save()


if __name__ == '__main__':
    main()
