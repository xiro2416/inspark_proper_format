"""Read-only schedule resolver used only by validated Planner V2 manifests."""
from __future__ import annotations

from .manifest import DeploymentManifest
from .model import ScheduleSpec


class ScheduleRegistry:
    def __init__(self, manifest: DeploymentManifest, *, apply: bool):
        if apply and manifest.status != 'validated':
            raise ValueError('Only a validated Planner V2 manifest may mutate runtime schedules')
        self.manifest = manifest
        self.apply = bool(apply)
        self.hits = 0
        self.generic_hits = 0
        self.legacy_hits = 0
        self.misses = 0

    @classmethod
    def offline_candidate(cls, manifest: DeploymentManifest) -> 'ScheduleRegistry':
        """Explicit test-only resolver; production deployment never calls this."""
        if manifest.status not in ('candidate', 'validated'):
            raise ValueError('Offline application requires candidate or validated status')
        result = cls(manifest, apply=False); result.apply = True
        return result

    def resolve(self, model: str, role: str, *, batch: int, m: int, n: int, k: int) -> ScheduleSpec | None:
        policy = self.manifest.policies.get(f'{model}:{role}')
        if policy is None:
            self.misses += 1
            return None
        if policy.legacy_exception:
            self.legacy_hits += 1
            return None
        shape = f'b{batch}_m{m}_n{n}_k{k}'
        schedule = policy.schedules.get(shape)
        if schedule is not None:
            self.hits += 1
            return schedule
        schedule = policy.schedules.get('generic')
        if schedule is not None:
            self.generic_hits += 1
            return schedule
        self.misses += 1
        return None

    def stats(self) -> dict:
        return dict(apply=self.apply, hits=self.hits, generic_hits=self.generic_hits,
                    legacy_hits=self.legacy_hits,
                    misses=self.misses, manifest_hash=self.manifest.manifest_hash)
