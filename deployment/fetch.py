"""Fetch pinned weights, source containers and INT8 calibration; no SM120 engines."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file', type=Path)
    args = parser.parse_args()
    registry = json.loads((ROOT / 'configs/current/release.json').read_text())
    sm89 = ROOT / 'configs/hardware/sm89/indextts/assets.json'
    if sm89.is_file():
        pinned = json.loads(sm89.read_text())
        registry.update(repo=pinned['repo'], weights_revision=pinned['weights_revision'], engine_revision=pinned['source_revision'])
    credential = args.token_file or ROOT.parent / '.cache/huggingface/token'
    token = os.environ.get('HF_TOKEN') or (credential.read_text().strip() if credential.is_file() else None)
    if not token:
        raise RuntimeError('HF_TOKEN or A_1007/.cache/huggingface/token is required')
    endpoint = 'https://huggingface.co' if registry['repo'] == 'xirr/index_pipeline' else os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    common = dict(repo_id=registry['repo'], token=token, endpoint=endpoint)
    assets = ROOT / 'local_assets'
    weights = assets / 'weights'
    snapshot_download(**common, revision=registry['weights_revision'], local_dir=weights,
                      allow_patterns=['unquantized/**', 'shared/**', 'manifest.json', 'training_provenance.json'], max_workers=4)
    from inspark_infer.api.release import verify_weights
    verify_weights(weights)
    prefix = registry['engine_prefix']
    wanted = ['raw_sources/s2mel.pth', 'raw_sources/bigvgan_generator.pt',
              'artifacts/current_release/calibration/int8_smoothquant.json',
              'artifacts/current_release/calibration/int8_smoothquant_target_vocoder.json']
    download = assets / 'download'
    manifest_file = hf_hub_download(**common, revision=registry['engine_revision'],
                                   filename=prefix + '/manifest.json', local_dir=download)
    records = {r['path']: r for r in json.loads(Path(manifest_file).read_text())}
    wanted.extend(sorted(name for name in records if name.startswith('artifacts/official_trtllm_dspark/')))
    for name in wanted:
        filename = hf_hub_download(**common, revision=registry['engine_revision'], filename=prefix + '/' + name, local_dir=download)
        path = Path(filename)
        record = records[name]
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != record['bytes'] or digest != record['sha256']:
            raise ValueError('Asset identity mismatch: ' + name)
    from inspark_infer.api.release import materialize
    config, _ = materialize(download / prefix, weights, assets / 'runtime', 'fp32', 1)
    receipt = dict(repo=registry['repo'], weights_revision=registry['weights_revision'],
                   engine_revision=registry['engine_revision'], verified_sources=wanted,
                   config=str(config), downloaded_sm120_engines=False)
    (assets / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
