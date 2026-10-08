"""Upload accepted A_1007 assets; historical cleanup is a later verified step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspark_infer.build.zipvoice import BATCHES, safe_path, validate_bundle


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-card', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, CommitOperationAdd
    token = (ROOT / '.cache/huggingface/token').read_text().strip()
    api = HfApi(endpoint='https://huggingface.co', token=token)
    registry_path = ROOT / 'configs/hardware/sm89/zipvoice_int8_registry.json'
    registry = json.loads(registry_path.read_text())
    assert registry['status'] == 'accepted_local_publication_pending'
    assert set(registry['bundles']) == set(map(str, BATCHES))
    assert registry['repo_id'] == 'xirr/zip_pipeline'
    info = api.model_info(registry['repo_id'], files_metadata=True)
    assert info.private and api.whoami()['name'] == 'xirr'
    assets = {}
    for batch, entry in registry['bundles'].items():
        bundle = safe_path(ROOT, entry['local_path'])
        manifest = validate_bundle(bundle, int(batch))
        assert manifest['integration_validation'] == 'accepted_target_evidence_bound'
        for name in [*manifest['files'], 'manifest.json']:
            assets[entry['bundle_path'] + '/' + name] = safe_path(bundle, name)
    weights_path = ROOT / 'reports/sm89/zipvoice/a1007/weights-manifest.json'
    weights = json.loads(weights_path.read_text())
    assert weights['precisions'] == ['eager', 'int8']
    for item in weights['files']:
        path = safe_path(ROOT / 'models/zipvoice', item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        assets[item['path']] = path
    assets['weights-manifest.json'] = weights_path
    assets['README.md'] = args.model_card.resolve()
    assert assets['README.md'].is_relative_to(ROOT)
    expected = {name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                for name, path in assets.items()}
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                  for name, path in assets.items()]
    report_path = ROOT / 'reports/sm89/zipvoice/a1007/publication-assets.json'
    report = {'status': 'prepared', 'repo_id': registry['repo_id'],
              'previous_revision': info.sha, 'files': expected,
              'deletions': [], 'history_cleanup': 'not_started'}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    commit = api.create_commit(registry['repo_id'], operations=operations,
                              parent_commit=info.sha, num_threads=4,
                              commit_message='Publish validated A_1007 INT8 batches 1 2 4 8 16 32 64 and source weights')
    report.update(status='uploaded_remote_hash_verification_pending', revision=commit.oid)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    after = api.model_info(registry['repo_id'], revision=commit.oid, files_metadata=True)
    assert after.private
    remote = {x.rfilename: x for x in after.siblings}
    for name, entry in expected.items():
        item = remote[name]
        assert item.size == entry['bytes'], name
        if item.lfs:
            digest = item.lfs['sha256'] if isinstance(item.lfs, dict) else item.lfs.sha256
            assert digest == entry['sha256'], name
        else:
            body = assets[name].read_bytes()
            digest = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            assert digest == item.blob_id, name
    registry['revision'] = commit.oid
    registry['status'] = 'uploaded_fresh_download_validation_pending'
    registry_path.write_text(json.dumps(registry, indent=2) + '\n')
    report.update(status='private_assets_remote_hash_verified_fresh_download_pending')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'revision': commit.oid, 'files': len(expected)}))


if __name__ == '__main__':
    main()
