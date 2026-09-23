"""Fail-closed shadow attachment; V2 does not mutate kernels until validated."""
from __future__ import annotations

import hashlib
from pathlib import Path

from inspark_infer.ops.planning.v2.manifest import load, stable_hash
from inspark_infer.ops.planning.v2.model import HardwareProfile


def source_hash(root: str | Path) -> str:
    root = Path(root)
    values = []
    for path in sorted((root / 'src' / 'inspark_infer').rglob('*.py')):
        values.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return stable_hash(values)


def model_hash(config: dict) -> str:
    weights = Path(config['weights'])
    relative = (
        'index_tts2/gpt.pth', 'index_tts2/s2mel.pth',
        'draft_onpolicy100/model.safetensors',
    )
    assets = []
    for name in relative:
        path = weights / name
        assets.append((name, path.stat().st_size, path.stat().st_mtime_ns))
    return stable_hash(dict(student=config['student_sha256'], assets=assets))


def toolchain() -> dict[str, str]:
    import torch
    import triton
    return {'torch': torch.__version__, 'triton': triton.__version__, 'cuda': str(torch.version.cuda)}


def runtime_identity(engine) -> dict:
    root = Path(__file__).resolve().parents[5]
    return dict(
        hardware=HardwareProfile.current(), model_hash=model_hash(engine.config),
        source_hash=source_hash(root), toolchain=toolchain(),
    )


def prepare(engine, path: str | Path, *, apply: bool = False) -> dict:
    if engine.sessions or engine.head_graphs is not None:
        raise RuntimeError('Planner shadow must attach before capture/admission')
    manifest = load(path)
    maximum = int(engine.config['max_batch'])
    if maximum not in manifest.supported_batches:
        raise ValueError(f'Planner manifest does not support configured max_batch={maximum}')
    identity = runtime_identity(engine)
    manifest.validate_runtime(**identity)
    exceptions = {
        role: policy.exception_reason for role, policy in manifest.policies.items()
        if policy.legacy_exception
    }
    from inspark_infer.ops.planning.v2.runtime import ScheduleRegistry
    registry = ScheduleRegistry(manifest, apply=apply)
    engine.planner_v2_registry = registry
    changes_schedules = any(not policy.legacy_exception for policy in manifest.policies.values())
    return dict(
        status='apply' if apply else 'shadow', manifest=str(Path(path).resolve()),
        manifest_hash=manifest.manifest_hash, roles=len(manifest.policies),
        signatures=len(manifest.signatures), exceptions=exceptions,
        supported_batches=list(manifest.supported_batches), max_batch=maximum,
        mutates_runtime=bool(apply and changes_schedules), online_tuning=False,
    )


def prepare_shadow(engine, path: str | Path) -> dict:
    return prepare(engine, path, apply=False)
