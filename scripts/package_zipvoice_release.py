"""Package explicitly accepted A_1007 artifacts; never infer acceptance from a build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspark_infer.build.zipvoice import BATCHES, safe_path, validate_bundle, plugin_package


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Accepted per-batch inventories, policies and evidence hashes')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    assert selection['status'] == 'all_batches_migrated_optimized_validated'
    assert set(selection['batches']) == set(map(str, BATCHES))
    registry = {'schema': 1, 'model': 'zipvoice', 'precision': 'int8',
                'repo_id': 'xirr/zip_pipeline', 'revision': None,
                'frame_profile': [600, 760, 920], 'token_profile': [52, 78, 141],
                'status': 'accepted_local_publication_pending', 'bundles': {}}
    hardware = {'name': 'NVIDIA GeForce RTX 4090', 'sm': 89,
                'memory_total_mib': 49140, 'tensorrt': '11.3.0.99', 'power_limit_w': 400}
    support = []
    for directory in ('models/zipvoice', 'ops/cuda/zipvoice'):
        support.extend((ROOT / 'src/inspark_infer' / directory).rglob('*.py'))
    runtime_root = ROOT / 'src/inspark_infer/runtime/zipvoice'
    # Retired frame-specific routes and telemetry are not A_1007 dependencies.
    support.extend(runtime_root / name for name in (
        '__init__.py', 'cli.py', 'worker.py', 'telemetry.py', 'request_telemetry.py',
        'routes/__init__.py'))
    support += [ROOT / 'src/inspark_infer/ops/tensorrt/zipvoice/engine.py',
                ROOT / 'src/inspark_infer/build/zipvoice.py',
                ROOT / 'src/inspark_infer/runtime/device.py']
    for batch in BATCHES:
        chosen = selection['batches'][str(batch)]
        assert chosen['migration_accepted'] and chosen['optimization_review_complete']
        for path, expected in chosen['evidence'].items():
            assert sha(safe_path(ROOT, path)) == expected, path
        inventory = json.loads(safe_path(ROOT, chosen['inventory']).read_text())
        runner = safe_path(ROOT, chosen['runner'])
        assert runner.stem in ('a1007', 'a1007_delivery', 'a1007_graph', 'a1007_delivery_graph')
        package=plugin_package(batch,inventory.get('plugin_package'))
        plugins = {str(p.relative_to(ROOT)): sha(p) for p in sorted(
            (ROOT / 'src' / Path(*package.split('.'))).glob('*.py'))}
        support_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(support + [runner]))}
        runtime = {**plugins, **support_hashes, str(runner.relative_to(ROOT)): sha(runner)}
        identity = {'model': 'zipvoice', 'precision': 'int8', 'batch': batch,
                    'hardware': hardware, 'runtime_sources': runtime,
                    'application': runner.stem, 'runner': str(runner.relative_to(ROOT)),
                    'plugin_package':package,
                    'engines': {k: v['sha256'] for k, v in inventory['engines'].items()},
                    'pcm_policy': chosen['pcm_policy']}
        bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        bundle = ROOT / f'artifacts/trt113_bundles/zipvoice/sm89/int8/b{batch}/{bundle_id}'
        files, engines = {}, {}

        def copy(source, name, expected=None):
            source = source.resolve()
            assert source.is_relative_to(ROOT), source
            digest = sha(source)
            if expected is not None:
                assert digest == expected, source
            destination = safe_path(bundle, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha(destination) != digest:
                shutil.copyfile(source, destination)
            files[name] = {'sha256': digest, 'bytes': destination.stat().st_size}

        for key, engine in inventory['engines'].items():
            source = Path(engine['path'])
            if not source.is_absolute():
                source = ROOT / source
            copy(source, f'{key}/engine.plan', engine['sha256'])
            engines[key] = {'path': f'{key}/engine.plan', 'sha256': engine['sha256'],
                            'shape_profile': engine['shape_profile'],
                            'origin_graph_sha256': engine.get('source_sha256')}
        assert engines['fm']['shape_profile']['x'] == [[batch, n, 100] for n in (600, 760, 920)]
        assert engines['text']['shape_profile']['token_ids'] == [[batch, n] for n in (52, 78, 141)]
        for name in ('config/tokens.txt', 'config/model.json', 'LICENSE'):
            copy(ROOT / 'models/zipvoice' / name, name)
        for name in ('licenses/ZipVoice.txt', 'licenses/Vocos.txt', 'THIRD_PARTY_NOTICES.md'):
            copy(ROOT / name, name)
        manifest = {'schema': 1, **identity, 'bundle_id': bundle_id, 'engines': engines,
                    'files': files, 'plugin_sources': plugins, 'support_sources': support_hashes,
                    'runner': str(runner.relative_to(ROOT)), 'runner_sha256': sha(runner),
                    'application': runner.stem,
                    'origin_mapping_source_sha256': inventory['origin_mapping_source_sha256'],
                    'certified_for_production': False,
                    'integration_validation': 'accepted_target_evidence_bound',
                    'validation_evidence': chosen['evidence'],
                    'quantization': {'method': 'SmoothQuant', 'alpha': .5, 'floating_layers': 4,
                                     'int8_layers': 12, 'quantized_modules': 180}}
        write(bundle / 'manifest.json', manifest)
        validate_bundle(bundle, batch)
        registry['bundles'][str(batch)] = {
            'bundle_id': bundle_id, 'local_path': str(bundle.relative_to(ROOT)),
            'bundle_path': f'bundles/zipvoice/sm89/int8/b{batch}/{bundle_id}'}
        write(ROOT / f'configs/hardware/sm89/zipvoice_int8_b{batch}.json', {
            'schema': 1, 'model': 'zipvoice', 'precision': 'int8', 'batch': batch,
            'registry': 'zipvoice_int8_registry.json', 'bundle_id': bundle_id})
    write(ROOT / 'configs/hardware/sm89/zipvoice_int8_registry.json', registry)
    assets = []
    base = ROOT / 'models/zipvoice'
    for directory in ('eager', 'int8', 'config', 'vocos', 'onnx'):
        assert (base / directory).is_dir(), directory
        assets.extend(p for p in (base / directory).rglob('*') if p.is_file())
    assets.append(base / 'LICENSE')
    write(ROOT / 'reports/sm89/zipvoice/a1007/weights-manifest.json', {
        'schema': 1, 'model': 'zipvoice', 'precisions': ['eager', 'int8'],
        'files': [{'path': str(p.relative_to(base)), 'bytes': p.stat().st_size,
                   'sha256': sha(p)} for p in sorted(assets)]})
    print(json.dumps({'status': 'source_bound_bundles_prepared', 'batches': list(BATCHES)}))


if __name__ == '__main__':
    main()
