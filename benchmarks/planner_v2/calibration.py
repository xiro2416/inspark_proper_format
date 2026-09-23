"""Bounded single-GPU offline calibration; never imported by serving hot paths."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from inspark_infer.ops.planning.v2.manifest import RolePolicy
from inspark_infer.ops.planning.v2.model import OperatorSignature, ScheduleSpec


@dataclass(frozen=True)
class Measurement:
    role_key: str
    shape_key: str
    schedule: ScheduleSpec
    median_s: float
    p95_s: float
    correct: bool
    max_abs: float = 0.0
    relative_l2: float = 0.0

    def __post_init__(self):
        if min(self.median_s, self.p95_s) <= 0:
            raise ValueError('Positive measured latency required')


class Runner(Protocol):
    def measure(self, signature: OperatorSignature, schedule: ScheduleSpec) -> Measurement: ...


def select_role_policy(signatures: list[OperatorSignature], measurements: list[Measurement],
                       *, regression_limit: float = 0.05, legacy_backend: str = 'legacy') -> RolePolicy:
    if not signatures:
        raise ValueError('Role needs signatures')
    role = signatures[0].role_key
    if any(signature.role_key != role for signature in signatures):
        raise ValueError('Mixed roles')
    rows = [row for row in measurements if row.role_key == role and row.correct]
    shapes = {signature.shape_key: signature for signature in signatures}
    if any(not any(row.shape_key == shape for row in rows) for shape in shapes):
        raise ValueError(f'Missing correct measurement for {role}')
    best = {shape: min(row.median_s for row in rows if row.shape_key == shape) for shape in shapes}
    backends = sorted({row.schedule.backend for row in rows})
    choices = []
    for backend in backends:
        selected = {}
        regrets = []
        weighted = 0.0
        valid = True
        for shape, signature in shapes.items():
            candidates = [row for row in rows if row.shape_key == shape and row.schedule.backend == backend]
            if not candidates:
                valid = False
                break
            winner = min(candidates, key=lambda row: row.median_s)
            selected[shape] = winner.schedule
            regret = winner.median_s / best[shape] - 1
            regrets.append(regret)
            weighted += winner.median_s * signature.calls * signature.critical_weight
        if valid:
            choices.append((max(regrets), weighted, backend, selected))
    passing = [choice for choice in choices if choice[0] <= regression_limit]
    if passing:
        worst, _, backend, selected = min(passing, key=lambda item: (item[1], item[0]))
        return RolePolicy(backend=backend, global_layout=next(iter(selected.values())).global_layout,
                          schedules=selected)
    legacy = next((choice for choice in choices if choice[2] == legacy_backend), None)
    chosen = legacy or min(choices, key=lambda item: (item[0], item[1]))
    worst, _, backend, selected = chosen
    return RolePolicy(
        backend=backend, global_layout=next(iter(selected.values())).global_layout,
        schedules=selected, legacy_exception=True,
        exception_reason=f'No fixed backend met {regression_limit:.1%}; best worst-shape regret={worst:.1%}',
        remove_when=f'A fixed backend passes all shapes at <= {regression_limit:.1%} regression',
    )


def calibrate(signatures: list[OperatorSignature], candidates: dict[str, list[ScheduleSpec]],
              runner: Runner, *, budget_seconds: float = 7200, regression_limit: float = 0.05,
              legacy_backend: str = 'legacy') -> tuple[dict[str, RolePolicy], list[Measurement]]:
    started = time.monotonic()
    measurements = []
    for signature in signatures:
        options = candidates.get(signature.shape_key)
        if not options:
            raise ValueError(f'No candidates for {signature.shape_key}')
        for schedule in options:
            if time.monotonic() - started >= budget_seconds:
                raise TimeoutError('Calibration budget exhausted before role coverage completed')
            measurements.append(runner.measure(signature, schedule))
    policies = {}
    for role in sorted({signature.role_key for signature in signatures}):
        role_signatures = [signature for signature in signatures if signature.role_key == role]
        policies[role] = select_role_policy(
            role_signatures, measurements, regression_limit=regression_limit,
            legacy_backend=legacy_backend,
        )
    return policies, measurements
