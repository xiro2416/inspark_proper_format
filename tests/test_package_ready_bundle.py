"""CPU coverage for portable ready manifests with recursive shared assets."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.package_ready_bundle import package
from inspark_infer.api.release import verify_bundle
from inspark_infer.runtime.bundle_paths import read_json


def fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    def write(name, data):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + '\n')
        return path
    old = write('old_calibration.json', {'scheme': 'nvfp4', 'role_specs': {}})
    mixed = write('mixed_calibration.json', {'scheme': 'nvfp4_fp8', 'role_specs': {}})
    engine = source / 'original.engine'; engine.write_bytes(b'engine-data')
    digest = hashlib.sha256(engine.read_bytes()).hexdigest()
    def plan(name, calibration):
        path = write(name + '/model.plan.json', {
            'component': 'target', 'batch': 16, 'engine': '../original.engine',
            'bytes': engine.stat().st_size, 'sha256': digest,
            'quantization_recipe': {'calibration': {
                'path': str(calibration),
                'sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
                'bytes': calibration.stat().st_size}}})
        write(name + '/model.inspector.json', {'Layers': [{'TacticName': 'x' * 2000}]})
        return path
    original = plan('old', old); newer = plan('mixed', mixed)
    (source / 'official').mkdir(); (source / 'official/source.py').write_text('source = 1\n')
    (source / 'official/metadata.json').write_text('{"source": "official"}\n')
    raw = source / 'raw'; raw.mkdir()
    for name in ('s2mel.pth', 'bigvgan_generator.pt'):
        (raw / name).write_bytes(name.encode())
    acoustic = write('acoustic.json', {'batch': 16, 'calibration': str(mixed),
        'target_plan': str(newer), 'vocoder_plan': str(newer)})
    bucket = write('bucket.json', {'batch': 16, 'calibration': str(mixed),
        'target_plan': str(newer), 'draft_plan': str(newer)})
    ready = write('ready.json', {'schema': 1, 'admission_batch': 32,
        'acoustic_batch': 16, 'ar_buckets': {'16': str(bucket)},
        'acoustic_deployment': str(acoustic)})
    root = write('deployment.json', {'schema': 9, 'precision': 'nvfp4_fp8',
        'batch': 32, 'first_chunk_scheduler': 'ready_c',
        'target_plan': str(original), 'ready_scheduler_plan': str(ready),
        'calibration': str(mixed), 'component_calibrations': {'target': str(old)},
        'official_sources': str(source / 'official')})
    return source, root, raw, engine, old, mixed


def test_package_recursive_relocation_and_immutable_json(tmp_path):
    source, root, raw, engine, old, mixed = fixture(tmp_path)
    before = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    out = tmp_path / 'bundle'
    result = package(root, raw, out)
    assert result['assets'] > 10
    verify_bundle(out)
    deployment = read_json(result['deployment'])
    manifest = read_json(deployment['ready_scheduler_plan'])
    bucket = read_json(manifest['ar_buckets']['16'])
    acoustic = read_json(manifest['acoustic_deployment'])
    assert bucket['target_plan'] == acoustic['target_plan']
    assert acoustic['target_plan'] == acoustic['vocoder_plan']
    target = read_json(deployment['target_plan'])
    assert Path(target['engine']).read_bytes() == engine.read_bytes()
    assert len(list((out / 'artifacts/current_release/ready_c/engine_blobs').glob('*'))) == 1
    for original in (old, mixed):
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        copy = out / 'artifacts/current_release/calibration' / (digest + '.json')
        assert copy.read_bytes() == original.read_bytes()
        assert copy.stat().st_ino != original.stat().st_ino
    assert Path(deployment['target_plan']).stat().st_ino != (source / 'old/model.plan.json').stat().st_ino
    for path in out.rglob('*.json'):
        if path.name not in ('manifest.json', 'metadata.json'):
            assert str(source) not in path.read_text()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_missing_dependency_is_explicit_and_no_bundle_marker(tmp_path):
    source, root, raw, _, _, _ = fixture(tmp_path)
    (source / 'bucket.json').unlink()
    out = tmp_path / 'bundle'
    with pytest.raises(FileNotFoundError):
        package(root, raw, out)
    assert not (out / '.bundle_root').exists()


def test_changed_engine_identity_rejected(tmp_path):
    _, root, raw, engine, _, _ = fixture(tmp_path)
    engine.write_bytes(b'changed-engine')
    with pytest.raises(ValueError, match='Engine identity mismatch'):
        package(root, raw, tmp_path / 'bundle')


def test_nonempty_destination_rejected(tmp_path):
    _, root, raw, _, _, _ = fixture(tmp_path)
    out = tmp_path / 'bundle'; out.mkdir(); (out / 'existing').write_text('preserve')
    with pytest.raises(ValueError, match='empty package'):
        package(root, raw, out)
    assert (out / 'existing').read_text() == 'preserve'
