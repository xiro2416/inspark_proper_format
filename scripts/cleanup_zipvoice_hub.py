"""Retire recorded obsolete Hub assets after a fresh-download release validation."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/sm89/zipvoice/a1007'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-validation', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Apply authorized cleanup; default prepares the concrete plan only')
    args = parser.parse_args()
    validation_path = args.fresh_validation.resolve()
    validation_path.relative_to(ROOT)
    validation = read(validation_path)
    registry_path = ROOT / 'configs/hardware/sm89/zipvoice_int8_registry.json'
    registry = read(registry_path)
    publication = read(REPORTS / 'publication-assets.json')
    assert validation['status'] == 'all_batches_and_weights_fresh_download_validated'
    assert set(validation['batches']) == set(registry['bundles']) == {'1','2','4','8','16','32','64'}
    assert validation['revision'] == registry['revision'] == publication['revision']
    assert publication['status'] == 'private_assets_remote_hash_verified_fresh_download_pending'
    assert registry['repo_id'] == 'xirr/zip_pipeline'
    from huggingface_hub import HfApi, CommitOperationDelete
    api = HfApi(endpoint='https://huggingface.co', token=(ROOT / '.cache/huggingface/token').read_text().strip())
    assert api.whoami()['name'] == 'xirr'
    info = api.model_info(registry['repo_id'], files_metadata=True)
    assert info.private and info.sha == registry['revision'], 'Remote changed; reconcile before cleanup'
    expected = publication['files']
    current = {item.rfilename: item for item in info.siblings}
    for name, item in expected.items():
        remote = current[name]
        assert remote.size == item['bytes'], name
        if remote.lfs:
            assert remote.lfs.sha256 == item['sha256'], name
    obsolete = sorted(set(current) - set(expected) - {'.gitattributes'})
    # Only paths observed before publication may be removed. Concurrent additions survive.
    assert set(obsolete) <= set(read(REPORTS / 'hf-existing-paths.json')['paths'])
    keep_hashes = {item['sha256'] for item in expected.values()}
    old_lfs = [item for item in api.list_lfs_files(registry['repo_id']) if item.file_oid not in keep_hashes]
    # Deleted content can have several historical names; retain by object identity.
    retained = {item.file_oid for item in api.list_lfs_files(registry['repo_id']) if item.file_oid in keep_hashes}
    assert not retained.intersection(item.file_oid for item in old_lfs)
    plan = {'status':'prepared_not_applied', 'revision':info.sha, 'delete_paths':obsolete,
            'delete_lfs':[{'sha256':item.file_oid,'name':item.filename,'bytes':item.size} for item in old_lfs],
            'retained_object_count':len(retained), 'fresh_validation_sha256':hashlib.sha256(validation_path.read_bytes()).hexdigest()}
    report = REPORTS / 'hub-cleanup.json'
    write(report, plan)
    if not args.apply:
        print(json.dumps({'status':plan['status'],'paths':len(obsolete),'lfs_objects':len(old_lfs)}))
        return
    if obsolete:
        commit = api.create_commit(registry['repo_id'], parent_commit=info.sha,
            operations=[CommitOperationDelete(path_in_repo=name) for name in obsolete],
            commit_message='Retire replaced ZipVoice profiles and FP8 assets after A_1007 validation')
        plan.update(status='obsolete_paths_removed', deletion_revision=commit.oid)
        write(report,plan)
    api.super_squash_history(registry['repo_id'], branch='main', commit_message='Validated A_1007 INT8 release and source weights')
    plan['status']='history_squashed'; write(report,plan)
    if old_lfs:
        api.permanently_delete_lfs_files(registry['repo_id'],old_lfs,rewrite_history=True)
    final=api.model_info(registry['repo_id'],files_metadata=True)
    assert final.private
    assert {item.rfilename for item in final.siblings} == set(expected)|{'.gitattributes'}
    for item in final.siblings:
        if item.rfilename in expected:
            assert item.size==expected[item.rfilename]['bytes']
            if item.lfs: assert item.lfs.sha256==expected[item.rfilename]['sha256']
    registry.update(revision=final.sha,status='history_cleaned_final_fresh_download_validation_pending')
    write(registry_path,registry)
    plan.update(status='obsolete_paths_history_lfs_removed_final_download_pending',final_revision=final.sha)
    write(report,plan)
    print(json.dumps({'status':plan['status'],'revision':final.sha}))


if __name__ == '__main__':
    main()
