"""Reusable CUDA-event runner for bounded offline candidate calibration."""
from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Callable, Protocol

from .calibration import Measurement
from .model import OperatorSignature, ScheduleSpec


class CandidateFactory(Protocol):
    def __call__(self, signature: OperatorSignature, schedule: ScheduleSpec) -> tuple[Callable, Callable]:
        """Return candidate callable and numerical-reference callable."""


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    relative_l2: float


class CudaRunner:
    """One candidate at a time; compile/warmup is explicitly outside timing."""
    def __init__(self, factory: CandidateFactory, *, warmup: int = 10, inner: int = 100,
                 repeats: int = 7, tolerance: Tolerance = Tolerance(1e-3, 1e-3, 1e-3)):
        if min(warmup, inner, repeats) <= 0:
            raise ValueError('Positive benchmark counts required')
        self.factory, self.warmup, self.inner, self.repeats = factory, warmup, inner, repeats
        self.tolerance = tolerance

    @staticmethod
    def _flatten(value):
        import torch
        if isinstance(value, torch.Tensor): return [value]
        if isinstance(value, (tuple, list)):
            return [tensor for item in value for tensor in CudaRunner._flatten(item)]
        if isinstance(value, dict):
            return [tensor for key in sorted(value) for tensor in CudaRunner._flatten(value[key])]
        raise TypeError(f'Unsupported benchmark output {type(value)!r}')

    def measure(self, signature: OperatorSignature, schedule: ScheduleSpec) -> Measurement:
        import torch
        candidate, reference = self.factory(signature, schedule)
        # Compilation and allocator stabilization happen before numerical or timing samples.
        for _ in range(self.warmup): candidate()
        torch.cuda.synchronize(); actual = self._flatten(candidate()); expected = self._flatten(reference())
        if len(actual) != len(expected): raise ValueError('Candidate/reference output arity mismatch')
        max_abs = 0.0; numerator = 0.0; denominator = 0.0; correct = True
        for got, want in zip(actual, expected):
            g, w = got.float(), want.float(); delta = g - w
            max_abs = max(max_abs, float(delta.abs().max()))
            numerator += float((delta * delta).sum()); denominator += float((w * w).sum())
            correct &= bool(torch.allclose(g, w, atol=self.tolerance.atol, rtol=self.tolerance.rtol))
        relative_l2 = (numerator / max(denominator, 1e-30)) ** .5
        correct &= relative_l2 <= self.tolerance.relative_l2
        values = []
        for _ in range(self.repeats):
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
            for _ in range(self.inner): candidate()
            end.record(); end.synchronize(); values.append(begin.elapsed_time(end) * 1e-3 / self.inner)
        ordered = sorted(values); p95 = ordered[min(len(ordered)-1, int(.95*len(ordered)))]
        return Measurement(signature.role_key, signature.shape_key, schedule,
                           statistics.median(values), p95, correct, max_abs, relative_l2)
