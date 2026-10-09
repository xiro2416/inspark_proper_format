import hashlib
import json

import pytest

from deployment.ready_pipeline.run import plan_identity
from deployment import runtime_assets


def test_relocated_plan_identity_still_checks_recipe_and_engine_choice():
    original = {'target_plan': '/old/artifacts/b64/target/model.plan.json',
                'calibration': '/old/local_assets/calibration.json', 'round_burst': 2,
                'hardware': {'sm': 89}, 'draft_plan': '/old/artifacts/b64/draft/model.plan.json'}
    relocated = json.loads(json.dumps(original).replace('/old/', '/new/'))
    assert plan_identity(original) == plan_identity(relocated)
    relocated['round_burst'] = 1
    assert plan_identity(original) != plan_identity(relocated)
    relocated['round_burst'] = 2
    relocated['draft_plan'] = '/new/artifacts/b32/draft/model.plan.json'
    assert plan_identity(original) != plan_identity(relocated)


def test_runtime_bundle_rejects_tampered_engine_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_assets, 'ROOT', tmp_path)
    directory = tmp_path / 'artifacts/b48/target'
    directory.mkdir(parents=True)
    (directory / 'model.engine').write_bytes(b'validated engine')
    (directory / 'model.inspector.json').write_text('{}')
    plan = dict(batch=48, sm=89, optimization_level=5, tiling_optimization_level='full',
                max_num_tactics=2147483646, sha256=runtime_assets.digest(directory / 'model.engine'))
    (directory / 'model.plan.json').write_text(json.dumps(plan))
    files = {p.relative_to(tmp_path).as_posix(): dict(sha256=runtime_assets.digest(p), bytes=p.stat().st_size)
             for p in directory.iterdir()}
    manifest = tmp_path / 'deployment/publication/runtime-manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(kind='index_sm89_verified_runtime_assets', runtime_only=True, files=files,
        validated_builds=[dict(plan='artifacts/b48/target/model.plan.json', batch=48,
                              build_input_bindings_verified=True, engine_sha256=plan['sha256'])])))
    registry = tmp_path / 'configs/hardware/sm89/indextts/assets.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps(dict(manifest_sha256=runtime_assets.digest(manifest))))
    assert runtime_assets.verify_engine(directory, 48)
    with pytest.raises(ValueError, match='policy'):
        runtime_assets.verify_engine(directory, 64)
    (directory / 'model.engine').write_bytes(b'changed engine')
    with pytest.raises(ValueError, match='runtime asset'):
        runtime_assets.verify_engine(directory, 48)
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest hash'):
        runtime_assets.verify_engine(directory, 48)


def test_reference_relocation_preserves_canonical_identity_and_rejects_changes(tmp_path, monkeypatch):
    from benchmarks.unified_first_chunk import canonical_hash, load_manifest
    references = tmp_path / 'references'
    references.mkdir()
    audio = references / 'voice.wav'
    audio.write_bytes(b'unit-test reference bytes')
    payload = dict(cfm_steps=4, reference_seconds=3,
                   references=[dict(voice_id='voice.wav', path='/previous/voice.wav',
                                    sha256=hashlib.sha256(audio.read_bytes()).hexdigest())],
                   splits={'evaluation': [{'text': 'fixture', 'seed': 1}]})
    manifest = dict(payload, manifest_sha256=canonical_hash(payload))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    monkeypatch.setenv('INSPARK_REFERENCE_ROOT', str(references))
    relocated = load_manifest(path)
    assert relocated['manifest_sha256'] == manifest['manifest_sha256']
    assert relocated['references'][0]['path'] == str(audio)
    assert path.read_bytes() == original
    audio.write_bytes(b'changed unit-test reference')
    with pytest.raises(ValueError, match='Reference hash'):
        load_manifest(path)
    manifest['splits']['evaluation'][0]['seed'] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Manifest content hash'):
        load_manifest(path, verify_references=False)


def test_reference_root_rejects_symlink_escape(tmp_path, monkeypatch):
    from benchmarks.unified_first_chunk import canonical_hash, load_manifest
    references = tmp_path / 'references'
    references.mkdir()
    outside = tmp_path / 'outside.wav'
    outside.write_bytes(b'unit-test bytes')
    (references / 'voice.wav').symlink_to(outside)
    payload = dict(cfm_steps=4, reference_seconds=3,
                   references=[dict(voice_id='voice.wav', path='/previous/voice.wav', sha256='unused')])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(payload, manifest_sha256=canonical_hash(payload))))
    monkeypatch.setenv('INSPARK_REFERENCE_ROOT', str(references))
    with pytest.raises(ValueError, match='escapes explicit root'):
        load_manifest(path, verify_references=False)


def test_download_materializes_cache_symlink_as_verified_file(tmp_path, monkeypatch):
    import sys
    import huggingface_hub
    from scripts import download_index_sm89 as downloader
    monkeypatch.setattr(downloader, 'ROOT', tmp_path)
    blob = tmp_path / 'cache/blobs/blob'
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b'unit-test engine bytes')
    snapshot = tmp_path / 'cache/snapshots/rev/engine'
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to('../../blobs/blob')
    manifest = tmp_path / 'cache/manifest.json'
    manifest.write_text(json.dumps(dict(source_weights_revision='weights-pin',
        files={'artifacts/model.engine': dict(bytes=blob.stat().st_size, sha256=downloader.sha(blob))})))
    registry = tmp_path / 'configs/hardware/sm89/indextts/assets.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps(dict(repo='unit-test/repo', revision='engine-pin', prefix='bundle',
        weights_revision='weights-pin', manifest_sha256=downloader.sha(manifest))))
    token = tmp_path / 'token'
    token.write_text('unit-test credential')
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.setattr(sys, 'argv', ['download', '--token-file', str(token)])
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download',
                        lambda filename, **kwargs: str(manifest if filename.endswith('manifest.json') else snapshot))
    downloader.main()
    target = tmp_path / 'artifacts/model.engine'
    assert target.is_file() and not target.is_symlink()
    assert target.read_bytes() == blob.read_bytes()
    # An idempotent retry retains the verified file.
    downloader.main()
    assert target.read_bytes() == blob.read_bytes()
    target.write_bytes(b'tampered engine')
    with pytest.raises(ValueError, match='Refuse to replace'):
        downloader.main()
