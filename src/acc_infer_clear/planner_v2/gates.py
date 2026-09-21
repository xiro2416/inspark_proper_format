"""Hard release gates for latency, throughput and speech quality."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceMetrics:
    first_packet_p50_ms: dict[int, float]
    first_packet_p95_ms: dict[int, float]
    throughput_req_s: dict[int, float]
    component_ms: dict[str, float]
    peak_memory_bytes: int


@dataclass(frozen=True)
class QualityMetrics:
    utmos: float
    cer: float


def release_gate(baseline: ServiceMetrics, candidate: ServiceMetrics,
                 baseline_quality: QualityMetrics, candidate_quality: QualityMetrics,
                 *, regression_limit: float = .05, utmos_limit: float = .03,
                 cer_limit: float = .02) -> dict:
    checks = []
    for name, lower_is_better, base, new in (
        ('first_packet_p50_ms', True, baseline.first_packet_p50_ms, candidate.first_packet_p50_ms),
        ('first_packet_p95_ms', True, baseline.first_packet_p95_ms, candidate.first_packet_p95_ms),
        ('throughput_req_s', False, baseline.throughput_req_s, candidate.throughput_req_s),
        ('component_ms', True, baseline.component_ms, candidate.component_ms),
    ):
        if set(base) != set(new):
            raise ValueError(f'{name} coverage mismatch')
        for key in base:
            regression = new[key] / base[key] - 1 if lower_is_better else base[key] / new[key] - 1
            checks.append(dict(metric=name, key=str(key), regression=regression,
                               limit=regression_limit, pass_gate=regression <= regression_limit))
    utmos_change = candidate_quality.utmos / baseline_quality.utmos - 1
    cer_change = candidate_quality.cer - baseline_quality.cer
    checks.extend((
        dict(metric='utmos', key='aggregate', regression=-utmos_change,
             limit=utmos_limit, pass_gate=utmos_change >= -utmos_limit),
        dict(metric='cer', key='aggregate', regression=cer_change,
             limit=cer_limit, pass_gate=cer_change <= cer_limit),
        dict(metric='peak_memory', key='no_oom', regression=0., limit=0.,
             pass_gate=candidate.peak_memory_bytes > 0),
    ))
    return dict(pass_gate=all(item['pass_gate'] for item in checks), checks=checks,
                policy=dict(performance_regression=regression_limit,
                            utmos_relative_drop=utmos_limit, cer_absolute_increase=cer_limit))


def component_graph_gate(rows: list[dict], *, regression_limit: float = .05,
                         relative_l2_limit: float = 1e-6, max_abs_limit: float = 1e-5) -> dict:
    checks=[]
    for row in rows:
        regression=row['candidate_ms']/row['baseline_ms']-1
        numerical=row.get('relative_l2',0.)<=relative_l2_limit and row.get('max_abs',0.)<=max_abs_limit
        checks.append(dict(batch=row['batch'],regression=regression,numerical=numerical,
                           pass_gate=regression<=regression_limit and numerical))
    passed=all(row['pass_gate'] for row in checks)
    return dict(pass_gate=passed,checks=checks,action='promote_candidate' if passed else 'retain_legacy_component',
                regression_limit=regression_limit,relative_l2_limit=relative_l2_limit,max_abs_limit=max_abs_limit)
