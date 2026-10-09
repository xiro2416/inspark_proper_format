"""Package a selected ready-C deployment and its recursive static dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package(deployment, raw_sources, out):
    deployment, raw_sources, out = map(lambda p: Path(p).resolve(),
                                     (deployment, raw_sources, out))
    root = json.loads(deployment.read_text())
    if root.get('schema') != 9 or root.get('first_chunk_scheduler') != 'ready_c':
        raise ValueError('Expected a schema9 ready_c deployment')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty package destination')
    mapping, pending, discovered = {}, {}, set()
    asset_root = Path('artifacts/current_release/ready_c')

    def source(value, owner):
        value = str(value)
        if value.startswith('bundle://'):
            bundle = next((p for p in (owner.parent, *owner.parents)
                           if (p / '.bundle_root').is_file()), None)
            if bundle is None:
                raise ValueError('Source bundle marker missing: ' + str(owner))
            resolved = (bundle / value[len('bundle://'):]).resolve()
            if not resolved.is_relative_to(bundle):
                raise ValueError('Source bundle path escapes root')
            return resolved
        path = Path(value)
        return (path if path.is_absolute() else owner.parent / path).resolve()

    def remember(path, relative):
        mapping[str(path)] = 'bundle://' + relative.as_posix()

    def copy_file(path, relative, binary=False):
        path = Path(path).resolve()
        if str(path) in mapping:
            return
        if not path.is_file():
            raise ValueError('Missing runtime dependency: ' + str(path))
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(path) != sha256(target):
                raise ValueError('Package destination collision: ' + str(relative))
        elif binary:
            try:
                target.hardlink_to(path)
            except OSError:
                shutil.copyfile(path, target)
        else:
            # Calibration JSON must retain exact bytes for engine identities,
            # but must never share a writable inode with its source.
            shutil.copyfile(path, target)
        remember(path, relative)

    def calibration(value, owner, record=None):
        path = source(value, owner)
        digest = sha256(path)
        if record and (record.get('sha256') != digest
                       or record.get('bytes', path.stat().st_size) != path.stat().st_size):
            raise ValueError('Calibration identity mismatch: ' + str(path))
        copy_file(path, Path('artifacts/current_release/calibration') / (digest + '.json'))

    def official(value, owner):
        path = source(value, owner)
        if not path.is_dir() or out.is_relative_to(path):
            raise ValueError('Missing/invalid official source directory: ' + str(path))
        relative = Path('artifacts/official_trtllm_dspark')
        remember(path, relative)
        for file in sorted(path.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:
                copy_file(file, relative / file.relative_to(path))

    def document(value, owner, preferred=None):
        path = source(value, owner)
        if str(path) in discovered:
            return
        data = json.loads(path.read_text())
        is_engine = 'engine' in data and 'sha256' in data
        if preferred is None:
            suffix = sha256(path)[:16]
            if is_engine:
                relative = asset_root / 'engines' / (
                    str(data.get('component', 'engine')) + '_b' + str(data.get('batch', 'static'))
                    + '_' + suffix) / 'model.plan.json'
            else:
                relative = asset_root / 'configs' / (path.stem + '_' + suffix + '.json')
        else:
            relative = preferred
        discovered.add(str(path)); remember(path, relative)
        pending[relative] = (path, data, is_engine)
        if is_engine:
            engine = source(data['engine'], path)
            digest = sha256(engine)
            if digest != data['sha256'] or data.get('bytes', engine.stat().st_size) != engine.stat().st_size:
                raise ValueError('Engine identity mismatch: ' + str(engine))
            copy_file(engine, asset_root / 'engine_blobs' / (digest + '.engine'), binary=True)
            record = data.get('quantization_recipe', {}).get('calibration')
            if record:
                calibration(record['path'], path, record)
            inspector = path.with_name(path.name.replace('.plan.json', '.inspector.json'))
            if inspector != path and inspector.is_file():
                # Inspector JSON may contain original diagnostic provenance;
                # it is separately written, never hardlinked.
                inspector_relative = relative.with_name('model.inspector.json')
                remember(inspector, inspector_relative)
                pending[inspector_relative] = (inspector, json.loads(inspector.read_text()), False)
            return
        for key, item in data.items():
            if key == 'calibration':
                calibration(item, path)
            elif key == 'component_calibrations':
                for entry in item.values():
                    calibration(entry, path)
            elif key == 'official_sources':
                official(item, path)
            elif key == 'ar_buckets':
                for entry in item.values():
                    document(entry, path)
            elif key == 'acoustic_deployment' or key.endswith('_plan'):
                if not isinstance(item, str) or not item:
                    raise ValueError('Invalid runtime plan path: ' + key)
                document(item, path)

    standard = Path('artifacts/current_release/deployments') / (
        f"{root['precision']}_b{root['batch']}.json")
    document(str(deployment), deployment, standard)
    for name in ('s2mel.pth', 'bigvgan_generator.pt'):
        copy_file(raw_sources / name, Path('raw_sources') / name, binary=True)

    def portable(item, owner):
        if isinstance(item, str):
            if item in mapping:
                return mapping[item]
            # Resolve relative runtime paths, retaining ordinary labels and
            # historic build provenance which are not runtime dependencies.
            if (item.startswith(('/', 'bundle://', './', '../'))
                    or item.endswith(('.engine', '.json', '.pth', '.pt'))):
                candidate = str(source(item, owner))
                return mapping.get(candidate, item)
            return item
        if isinstance(item, list):
            return [portable(value, owner) for value in item]
        if isinstance(item, dict):
            return {key: portable(value, owner) for key, value in item.items()}
        return item

    for relative, (owner, data, _) in pending.items():
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(portable(data, owner), indent=2) + '\n')
    (out / '.bundle_root').write_text('Selected InSpark ready-C static deployment\n')
    manifest = [{'path': file.relative_to(out).as_posix(), 'bytes': file.stat().st_size,
                 'sha256': sha256(file)} for file in sorted(out.rglob('*')) if file.is_file()]
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'deployment': str(out / standard), 'assets': len(manifest),
            'bytes': sum(record['bytes'] for record in manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--raw-sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.deployment, args.raw_sources, args.out), indent=2))


if __name__ == '__main__':
    main()
