"""Small, explicit conversions from V2 schedules to existing kernel contracts."""
from .model import ScheduleSpec


def full_m_plan(schedule: ScheduleSpec, *, m: int) -> dict:
    if schedule.schedule != 'full_m':
        raise ValueError('Expected a Full-M schedule')
    return dict(
        mode='tiled', M=m, bm=schedule.bm, bn=schedule.bn, bk=schedule.bk,
        stages=schedule.stages, warps=schedule.warps,
        wm=schedule.warp_m, wn=schedule.warp_n, schedule='full',
        swizzle=schedule.shared_layout_a.startswith('xor'),
        activation_eviction=schedule.activation_eviction,
        weight_eviction=schedule.weight_eviction,
    )


def tiled_plan(schedule: ScheduleSpec) -> dict:
    if schedule.schedule != 'tiled' or schedule.split_k != 1:
        raise ValueError('Expected a tiled schedule')
    return dict(
        bm=schedule.bm, bn=schedule.bn, bk=schedule.bk,
        stages=schedule.stages, warps=schedule.warps,
        split_k=schedule.split_k, schedule='tiled',
        swizzle=schedule.shared_layout_a.startswith('xor'),
    )


def conv_plan(schedule: ScheduleSpec, *, explicit: bool) -> dict:
    result = tiled_plan(schedule)
    if explicit:
        result.update(
            swizzle=schedule.shared_layout_a.startswith('xor'),
            double=False, inner_double=False,
        )
    return result
