"""Explainable candidate generation and calibrated latency equations."""
from __future__ import annotations

import math
from dataclasses import asdict

from .model import DTYPE_BYTES, HardwareProfile, MeasuredRates, OperatorSignature, ScheduleSpec


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def shared_bytes(op: OperatorSignature, s: ScheduleSpec) -> int:
    a = DTYPE_BYTES[op.input_dtype]
    b = DTYPE_BYTES[op.weight_dtype]
    epilogue = s.bm * s.bn * 4 if 'workspace' in s.fused_epilogue else 0
    return s.stages * (s.bm * s.bk * a + s.bk * s.bn * b) + epilogue


def register_lower_bound(op: OperatorSignature, s: ScheduleSpec) -> int:
    accumulator = ceildiv(s.bm * s.bn, 32 * s.warps)
    operands = ceildiv(s.bm * s.bk + s.bk * s.bn, 32 * s.warps * max(1, s.stages))
    epilogue = 12 + 4 * len(s.fused_epilogue)
    return accumulator + min(32, operands) + epilogue


def jobs(op: OperatorSignature, s: ScheduleSpec) -> int:
    if s.schedule in ('full_m', 'persistent'):
        return ceildiv(op.n, s.bn) * s.split_k
    return ceildiv(op.m, s.bm) * ceildiv(op.n, s.bn) * s.split_k


def bank_conflict_degree(rows: int, cols: int, element_bytes: int, *, xor_shift: int | None,
                         bank_width: int = 4, banks: int = 32) -> int:
    """Worst multiplicity for a warp reading one column from consecutive rows."""
    if min(rows, cols, element_bytes) <= 0:
        raise ValueError('Positive layout dimensions required')
    degree = 1
    for col in range(min(cols, 32)):
        counts = [0] * banks
        for lane in range(min(rows, 32)):
            logical_col = col ^ (lane >> xor_shift) if xor_shift is not None else col
            address = (lane * cols + logical_col) * element_bytes
            counts[(address // bank_width) % banks] += 1
        degree = max(degree, max(counts))
    return degree


def best_xor_shift(rows: int, cols: int, element_bytes: int) -> tuple[int | None, int]:
    choices = [(None, bank_conflict_degree(rows, cols, element_bytes, xor_shift=None))]
    choices.extend((shift, bank_conflict_degree(rows, cols, element_bytes, xor_shift=shift)) for shift in range(0, 6))
    return min(choices, key=lambda item: (item[1], 99 if item[0] is None else item[0]))


def estimate(profile: HardwareProfile, op: OperatorSignature, s: ScheduleSpec,
             *, rates: MeasuredRates | None = None, l2_hit_rate: float = 0.8,
             compiled_registers: int | None = None, compiled_shared: int | None = None,
             dependency_s: float | None = None, quant_s: float = 0.0,
             reduction_s: float = 0.0, layout_s: float = 0.0) -> dict:
    if not 0 <= l2_hit_rate <= 1:
        raise ValueError('l2_hit_rate must be in [0,1]')
    reasons = []
    if (op.input_dtype == 'fp8' or op.weight_dtype == 'fp8') and not profile.native_fp8:
        reasons.append('no_native_fp8')
    if s.bk % (32 if 'fp8' in (op.input_dtype, op.weight_dtype) else 16):
        reasons.append('mma_k_alignment')
    if s.split_k > ceildiv(op.k, s.bk):
        reasons.append('empty_split_k_partition')
    if s.schedule == 'full_m' and s.bm < op.m:
        reasons.append('full_m_does_not_cover_m')
    smem = compiled_shared if compiled_shared is not None else shared_bytes(op, s)
    registers = compiled_registers if compiled_registers is not None else register_lower_bound(op, s)
    if smem > profile.shared_per_cta:
        reasons.append('shared_per_cta')
    if registers > profile.max_registers_per_thread:
        reasons.append('registers_per_thread')
    resident = min(
        profile.shared_per_sm // max(1, smem),
        profile.registers_per_sm // max(1, registers * 32 * s.warps),
        profile.threads_per_sm // (32 * s.warps),
        profile.max_ctas_per_sm,
    )
    if resident < 1:
        reasons.append('no_resident_cta')
    count = jobs(op, s)
    capacity = profile.sms * max(1, resident)
    waves = ceildiv(count, capacity)
    wave_efficiency = count / max(1, waves * capacity)
    mt = 1 if s.schedule in ('full_m', 'persistent') else ceildiv(op.m, s.bm)
    nt, kt = ceildiv(op.n, s.bn), ceildiv(op.k, s.bk)
    executed_m = s.bm if s.schedule in ('full_m', 'persistent') else mt * s.bm
    executed = 2 * executed_m * nt * s.bn * kt * s.bk
    useful = 2 * op.m * op.n * op.k
    tile_efficiency = useful / max(1, executed)
    a_bytes, b_bytes = DTYPE_BYTES[op.input_dtype], DTYPE_BYTES[op.weight_dtype]
    a_requested = nt * op.m * op.k * a_bytes
    b_requested = mt * op.n * op.k * b_bytes
    unique = op.m * op.k * a_bytes + op.n * op.k * b_bytes
    output = op.m * op.n * DTYPE_BYTES[op.accumulator_dtype]
    workspace = s.split_k * output if s.split_k > 1 else 0
    dram = unique + (1 - l2_hit_rate) * max(0, a_requested + b_requested - unique) + output + workspace
    l2 = a_requested + b_requested + output + workspace
    shared = kt * count * (s.bm * s.bk * a_bytes + s.bk * s.bn * b_bytes)
    result = dict(
        legal=not reasons, reasons=reasons, shared_per_cta=smem,
        registers_per_thread=registers, resident_ctas=resident, jobs=count,
        waves=waves, wave_efficiency=wave_efficiency,
        tile_efficiency=tile_efficiency, useful_flops=useful,
        executed_flops=executed, estimated_dram_bytes=dram,
        estimated_l2_bytes=l2, estimated_shared_bytes=shared,
        compiled=compiled_registers is not None and compiled_shared is not None,
    )
    selected_rates = rates or profile.rates
    if selected_rates is None or reasons:
        result['ranking_proxy'] = tile_efficiency * wave_efficiency / (1 + 0.12 * (s.split_k - 1))
        result['missing_calibration'] = selected_rates is None
        return result
    terms = dict(
        tensor=executed / (selected_rates.tensor_flops_s * max(wave_efficiency, 1e-6)),
        dram=dram / selected_rates.dram_bytes_s,
        l2=l2 / selected_rates.l2_bytes_s,
        shared=shared / selected_rates.shared_bytes_s,
        dependency=dependency_s,
    )
    known = [value for value in terms.values() if value is not None]
    result.update(
        estimate_s=selected_rates.launch_s + quant_s + max(known) + reduction_s + layout_s,
        terms_s=terms, missing_terms=[key for key, value in terms.items() if value is None],
        not_an_optimality_guarantee=True,
    )
    return result


def _stage_candidates(profile: HardwareProfile, op: OperatorSignature, bm: int, bn: int, bk: int) -> tuple[int, ...]:
    per_stage = bm * bk * DTYPE_BYTES[op.input_dtype] + bk * bn * DTYPE_BYTES[op.weight_dtype]
    maximum = min(6, ceildiv(op.k, bk), profile.shared_per_cta // max(1, per_stage))
    if maximum < 1:
        return ()
    if profile.rates is None:
        center = min(3, maximum)
    else:
        mma_s = 2 * bm * bn * bk / profile.rates.tensor_flops_s
        center = max(1, min(maximum, math.ceil(profile.rates.global_latency_s / max(mma_s, 1e-12))))
    return tuple(sorted({max(1, center - 1), center, min(maximum, center + 1), maximum}))


def generate_candidates(profile: HardwareProfile, op: OperatorSignature, *, limit: int = 8,
                        allow_split_k: bool = True, backends: tuple[str, ...] = ('explicit',)) -> list[tuple[ScheduleSpec, dict]]:
    dtype_bk = (32, 64, 128) if 'fp8' in (op.input_dtype, op.weight_dtype) else (16, 32, 64, 128)
    bm_values = {16, 32, 64, 128}
    full_bm = max(16, 1 << (op.m - 1).bit_length())
    if full_bm <= 128:
        bm_values.add(full_bm)
    metadata = dict(op.metadata)
    if 'allowed_schedules' in metadata:
        allowed = tuple(metadata['allowed_schedules'].split(','))
    elif op.model in ('target', 'draft') and op.role in ('qkv', 'up'):
        allowed = ('full_m',) if (1 << (op.m - 1).bit_length()) <= 128 else ('tiled',)
    elif op.model in ('target', 'draft') and op.role in ('out', 'down'):
        allowed = ('tiled',)
    else:
        allowed = ('tiled', 'full_m')
    rows = []
    for backend in backends:
        for schedule in allowed:
            if schedule == 'full_m' and full_bm > 128:
                continue
            for bm in sorted(bm_values):
                if schedule == 'full_m' and bm != full_bm:
                    continue
                for bn in (32, 64, 128):
                    for bk in dtype_bk:
                        for warps in (4, 8):
                            warp_m, warp_n = ((2, warps // 2) if schedule == 'full_m' else (1, warps))
                            for stages in _stage_candidates(profile, op, bm, bn, bk):
                                spec = ScheduleSpec(
                                    backend=backend, schedule=schedule, bm=bm, bn=bn, bk=bk,
                                    warps=warps, stages=stages, warp_m=warp_m, warp_n=warp_n,
                                    fused_epilogue=op.epilogue,
                                )
                                info = estimate(profile, op, spec)
                                if info['legal']:
                                    rows.append((spec, info))
        if allow_split_k and metadata.get('allow_split_k') == 'true' and op.k >= 2048:
            for split in (2, 4):
                spec = ScheduleSpec(
                    backend=backend, schedule='split_k', bm=32, bn=64, bk=128,
                    warps=4, stages=2, split_k=split, warp_m=1, warp_n=4,
                    fused_epilogue=op.epilogue + ('workspace',),
                )
                info = estimate(profile, op, spec)
                if info['legal']:
                    rows.append((spec, info))
    score = lambda item: item[1].get('estimate_s', -item[1].get('ranking_proxy', 0))
    if profile.rates is None:
        ordered = sorted(rows, key=lambda item: item[1]['ranking_proxy'], reverse=True)
    else:
        ordered = sorted(rows, key=score)
    selected = []
    # Preserve schedule diversity before filling by score.
    for schedule in ('full_m', 'tiled', 'split_k'):
        match = next((item for item in ordered if item[0].schedule == schedule), None)
        if match is not None and match not in selected:
            selected.append(match)
    for item in ordered:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected[:limit]
