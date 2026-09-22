"""Versioned, fail-closed Planner V2 deployment manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from acc_infer_clear.ops.planning.v2.model import HardwareProfile, OperatorSignature, ScheduleSpec


SCHEMA_VERSION = 2


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class RolePolicy:
    backend: str
    global_layout: str
    schedules: dict[str, ScheduleSpec]
    legacy_exception: bool = False
    exception_reason: str = ''
    remove_when: str = ''

    def __post_init__(self):
        if not self.backend:
            raise ValueError('RolePolicy requires a backend')
        if not self.schedules and not self.legacy_exception:
            raise ValueError('A non-legacy RolePolicy requires schedules')
        if self.legacy_exception and (not self.exception_reason or not self.remove_when):
            raise ValueError('Legacy exceptions require reason and removal criterion')
        if any(schedule.backend != self.backend for schedule in self.schedules.values()):
            raise ValueError('A role may not switch backend by shape')

    def as_dict(self) -> dict:
        return dict(
            backend=self.backend, global_layout=self.global_layout,
            schedules={key: value.as_dict() for key, value in self.schedules.items()},
            legacy_exception=self.legacy_exception,
            exception_reason=self.exception_reason, remove_when=self.remove_when,
        )


@dataclass(frozen=True)
class DeploymentManifest:
    hardware: HardwareProfile
    model_hash: str
    source_hash: str
    toolchain: dict[str, str]
    policies: dict[str, RolePolicy]
    signatures: dict[str, OperatorSignature]
    supported_batches: tuple[int, ...] = tuple(range(1, 9)) + (16,)
    calibration: dict = field(default_factory=dict)
    status: str = 'shadow'
    schema: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema != SCHEMA_VERSION:
            raise ValueError('Unknown Planner V2 schema')
        if self.status not in ('shadow', 'candidate', 'validated'):
            raise ValueError('Invalid manifest status')
        if (not self.supported_batches or tuple(sorted(set(self.supported_batches))) != self.supported_batches
                or any(batch <= 0 for batch in self.supported_batches)):
            raise ValueError('supported_batches must be positive, unique and sorted')
        for role, policy in self.policies.items():
            if role not in {signature.role_key for signature in self.signatures.values()}:
                raise ValueError(f'Policy without signature role: {role}')
            known = {signature.shape_key for signature in self.signatures.values() if signature.role_key == role}
            if not set(policy.schedules) <= known | {'generic'}:
                raise ValueError(f'Unknown shape schedule for {role}')

    @property
    def device_fingerprint(self) -> str:
        payload = self.hardware.as_dict()
        payload['rates'] = None  # Rates may be refreshed without reidentifying physical compatibility.
        return stable_hash(payload)

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.as_dict(include_hash=False))

    def as_dict(self, *, include_hash: bool = True) -> dict:
        result = dict(
            schema=self.schema, status=self.status,
            hardware=self.hardware.as_dict(), device_fingerprint=self.device_fingerprint,
            model_hash=self.model_hash, source_hash=self.source_hash, toolchain=self.toolchain,
            supported_batches=list(self.supported_batches),
            policies={key: value.as_dict() for key, value in self.policies.items()},
            signatures={key: value.as_dict() for key, value in self.signatures.items()},
            calibration=self.calibration,
        )
        if include_hash:
            result['manifest_hash'] = stable_hash(result)
        return result

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))

    def validate_runtime(self, hardware: HardwareProfile, *, model_hash: str,
                         source_hash: str, toolchain: dict[str, str]) -> None:
        runtime = DeploymentManifest(
            hardware=hardware, model_hash=model_hash, source_hash=source_hash,
            toolchain=toolchain, policies=self.policies, signatures=self.signatures,
            supported_batches=self.supported_batches, calibration=self.calibration, status=self.status,
        )
        if runtime.device_fingerprint != self.device_fingerprint:
            raise ValueError('Planner manifest device mismatch')
        if model_hash != self.model_hash or source_hash != self.source_hash:
            raise ValueError('Planner manifest model/source mismatch')
        if toolchain != self.toolchain:
            raise ValueError('Planner manifest toolchain mismatch')


def schedule_from_dict(value: dict) -> ScheduleSpec:
    value = dict(value)
    value['fused_epilogue'] = tuple(value.get('fused_epilogue', ()))
    return ScheduleSpec(**value)


def signature_from_dict(value: dict) -> OperatorSignature:
    value = dict(value)
    value['epilogue'] = tuple(value.get('epilogue', ()))
    value['metadata'] = tuple(sorted(value.get('metadata', {}).items()))
    return OperatorSignature(**value)


def load(path: str | Path) -> DeploymentManifest:
    data = json.loads(Path(path).read_text())
    expected = {'schema', 'status', 'hardware', 'device_fingerprint', 'model_hash', 'source_hash',
                'supported_batches',
                'toolchain', 'policies', 'signatures', 'calibration', 'manifest_hash'}
    if set(data) != expected:
        raise ValueError(f'Planner manifest fields mismatch: {set(data) ^ expected}')
    supplied_hash = data.pop('manifest_hash')
    if stable_hash(data) != supplied_hash:
        raise ValueError('Planner manifest hash mismatch')
    rates = data['hardware'].get('rates')
    from acc_infer_clear.ops.planning.v2.model import MeasuredRates
    profile_data = dict(data['hardware'])
    profile_data['rates'] = None if rates is None else MeasuredRates(**rates)
    hardware = HardwareProfile(**profile_data)
    signatures = {key: signature_from_dict(value) for key, value in data['signatures'].items()}
    policies = {}
    for key, value in data['policies'].items():
        policies[key] = RolePolicy(
            backend=value['backend'], global_layout=value['global_layout'],
            schedules={shape: schedule_from_dict(schedule) for shape, schedule in value['schedules'].items()},
            legacy_exception=value.get('legacy_exception', False),
            exception_reason=value.get('exception_reason', ''), remove_when=value.get('remove_when', ''),
        )
    result = DeploymentManifest(
        hardware=hardware, model_hash=data['model_hash'], source_hash=data['source_hash'],
        toolchain=data['toolchain'], policies=policies, signatures=signatures,
        supported_batches=tuple(data['supported_batches']), calibration=data['calibration'],
        status=data['status'], schema=data['schema'],
    )
    if result.device_fingerprint != data['device_fingerprint']:
        raise ValueError('Planner manifest fingerprint mismatch')
    return result
