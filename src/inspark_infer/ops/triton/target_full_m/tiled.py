"""Experimental full-logical-M CTA W8A8 GEMM, isolated from deployed kernels.

Every CTA covers all logical M and one output-N tile. K is never split among
CTAs or reduced from partial sums. Physical BM masks next-power-of-two padding.
The optional m16_ilp schedule interleaves independent output-row accumulators,
not independent partial-K accumulators. No graph/tuning/weight packing here.
"""
import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as ac
from inspark_infer.ops.triton.acoustic_pipeline.conv import mma
from inspark_infer.ops.triton.fp8 import _quantize


@g.jit
def enqueue(A, W, SA, SB, part, M: gl.constexpr, N: gl.constexpr,
            K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
            BK: gl.constexpr, WARPS: gl.constexpr, WE: gl.constexpr, AE: gl.constexpr):
    AL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [WARPS, 1], [1, 0])
    BL: gl.constexpr = gl.BlockedLayout([16, 1], [8, 4], [1, WARPS], [0, 1])
    mm = gl.arange(0, BM, layout=gl.SliceLayout(1, AL))
    ak = part*BK + gl.arange(0, BK, layout=gl.SliceLayout(0, AL))
    nn = gl.program_id(0)*BN + gl.arange(0, BN, layout=gl.SliceLayout(0, BL))
    bk = part*BK + gl.arange(0, BK, layout=gl.SliceLayout(1, BL))
    ac.async_copy_global_to_shared(SA, A+mm[:, None]*K+ak[None, :],
                                   (mm[:, None]<M)&(ak[None, :]<K), eviction_policy=AE)
    ac.async_copy_global_to_shared(SB, W+nn[None, :]*K+bk[:, None],
                                   (nn[None, :]<N)&(bk[:, None]<K), eviction_policy=WE)
    ac.commit_group()


@g.jit
def consume(SA, SB, accs, BM: gl.constexpr, BK: gl.constexpr,
            MM: gl.constexpr, ILP: gl.constexpr):
    DA: gl.constexpr = gl.DotOperandLayout(0, MM, 4)
    DB: gl.constexpr = gl.DotOperandLayout(1, MM, 4)
    if ILP:
        # K32 order is increasing for EACH output accumulator. B fragment is
        # loaded once and reused across independent output M16 accumulators.
        for sub in gl.static_range(BK//32):
            b = SB.slice(sub*32, 32, 0).load(DB)
            if ILP == 2:
                # Same fragments/read volume as mode1, deliberately increase
                # load-consumer distance and register live range as a contrast.
                preloaded = ()
                for fragment in gl.static_range(BM//16):
                    preloaded += (SA.slice(fragment*16, 16, 0).slice(sub*32, 32, 1).load(DA),)
            updated = ()
            for fragment in gl.static_range(BM//16):
                if ILP == 2:
                    a = preloaded[fragment]
                else:
                    a = SA.slice(fragment*16, 16, 0).slice(sub*32, 32, 1).load(DA)
                updated += (mma(a, b, accs[fragment]),)
            accs = updated
    else:
        accs = (mma(SA.load(DA), SB.load(DB), accs[0]),)
    return accs


@g.jit
def output(ACC, AS, WS, BIAS, Y, row_start, M: gl.constexpr,
           N: gl.constexpr, ROWS: gl.constexpr, BN: gl.constexpr,
           WARPS: gl.constexpr, HAS_BIAS: gl.constexpr):
    O: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [WARPS, 1], [1, 0])
    acc = gl.convert_layout(ACC, O)
    mm = row_start + gl.arange(0, ROWS, layout=gl.SliceLayout(1, O))
    nn = gl.program_id(0)*BN + gl.arange(0, BN, layout=gl.SliceLayout(0, O))
    result = acc*gl.load(AS+mm, mm<M, 0)[:, None]*gl.load(WS+nn, nn<N, 0)[None, :]
    if HAS_BIAS:
        result = result + gl.load(BIAS+nn, nn<N, 0)[None, :]
    gl.store(Y+mm[:, None]*N+nn[None, :], result, (mm[:, None]<M)&(nn[None, :]<N))


@g.jit
def full_m_gemm(A, W, AS, WS, BIAS, Y, M: gl.constexpr, N: gl.constexpr,
                K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                BK: gl.constexpr, P: gl.constexpr, WARPS: gl.constexpr,
                WM: gl.constexpr, WN: gl.constexpr, ILP: gl.constexpr,
                SWIZZLE: gl.constexpr, HAS_BIAS: gl.constexpr,
                WE: gl.constexpr, AE: gl.constexpr):
    gl.static_assert(WM*WN == WARPS)
    gl.static_assert(not ILP or (WM == 1 and WN == WARPS and BN >= 8*WN))
    gl.static_assert(P >= 1 and P <= 6)
    gl.static_assert(BM >= M and BM % 16 == 0 and BK % 32 == 0)
    MM: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0],
                         warps_per_cta=[WM, WN], instr_shape=[16, 8])
    PER: gl.constexpr = 128//BK if BK <= 128 else 1
    MAX: gl.constexpr = (BK//16 if BK <= 128 else 8) if SWIZZLE else 1
    AL: gl.constexpr = gl.SwizzledSharedLayout(16, PER, MAX, [1, 0])
    BL: gl.constexpr = gl.SwizzledSharedLayout(16, PER, MAX, [0, 1])
    # Independent rank-2 descriptors avoid indexed rank-3 swizzle limitations.
    sa = (gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL),
          gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL),
          gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL),
          gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL),
          gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL),
          gl.allocate_shared_memory(A.dtype.element_ty, [BM, BK], AL))
    sb = (gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL),
          gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL),
          gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL),
          gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL),
          gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL),
          gl.allocate_shared_memory(W.dtype.element_ty, [BK, BN], BL))
    accs = ()
    if ILP:
        for fragment in gl.static_range(BM//16):
            accs += (gl.full((16, BN), 0, gl.float32, MM),)
    else:
        accs = (gl.full((BM, BN), 0, gl.float32, MM),)
    ITER: gl.constexpr = (K+BK-1)//BK
    # P=1: one shared buffer, no look-ahead. P=2..6: genuine independent
    # shared slots; copy tile (step+P-1) while consuming tile step.
    for part in gl.static_range(P-1):
        enqueue(A, W, sa[part], sb[part], part, M, N, K, BM, BN, BK, WARPS, WE, AE)
    for cycle in range((ITER+P-1)//P):
        for slot in gl.static_range(P):
            step = cycle*P+slot
            if step < ITER:
                gl.thread_barrier()
                enqueue(A, W, sa[(slot+P-1)%P], sb[(slot+P-1)%P], step+P-1,
                        M, N, K, BM, BN, BK, WARPS, WE, AE)
                ac.wait_group(P-1)
                gl.thread_barrier()
                accs = consume(sa[slot], sb[slot], accs, BM, BK, MM, ILP)
    ac.wait_group(0)
    gl.thread_barrier()
    if ILP:
        for fragment in gl.static_range(BM//16):
            output(accs[fragment], AS, WS, BIAS, Y, fragment*16,
                   M, N, 16, BN, WARPS, HAS_BIAS)
    else:
        output(accs[0], AS, WS, BIAS, Y, 0, M, N, BM, BN, WARPS, HAS_BIAS)


def normalize_plan(m, plan):
    """CPU-only validation; hardware resource validity remains a compiler gate."""
    bm = max(16, triton.next_power_of_2(m))
    if plan.get('bm', bm) != bm or plan.get('M', plan.get('m', m)) != m:
        raise ValueError('Full-M plan must match logical M and minimal physical BM')
    bn, bk = plan.get('bn', 64), plan.get('bk', 128)
    warps = plan.get('warps', 4)
    wm, wn = plan.get('wm', 2), plan.get('wn', 2)
    if (warps, wm, wn) not in ((4, 4, 1), (4, 2, 2), (4, 1, 4), (8, 4, 2), (8, 2, 4), (8, 1, 8)):
        raise ValueError('Unsupported total-warps/MMA warp-layout combination')
    if bn < 8 or bn & (bn-1) or bk not in (32, 64, 128, 256, 512, 1024):
        raise ValueError('BN must be power-of-two >=8; BK in 32/64/128/256/512/1024')
    stages = plan.get('stages', 1)
    if stages not in (1, 2, 3, 4, 5, 6): raise ValueError('This experiment has exactly 1..6 shared stages')
    # FP8 shared-ring storage. The benchmark must also gate actual compiler
    # shared bytes/registers/spills: this arithmetic lower bound is not enough
    # to prove residency or register feasibility for large full-BK operands.
    shared_limit = plan.get('shared_limit_bytes', 101376)
    if stages*bk*(bm+bn) > shared_limit:
        raise ValueError('Shared operand ring exceeds per-CTA shared-memory limit')
    schedule = plan.get('schedule', 'full')
    if schedule not in ('full', 'm16_ilp', 'm16_preload'):
        raise ValueError('schedule is full, m16_ilp or m16_preload')
    if schedule != 'full' and (wm != 1 or wn != warps or bn < 8*wn):
        raise ValueError('M16 ILP requires WM=1, WN=WARPS and BN>=8*WN; use matched full control')
    if plan.get('split_k', 1) != 1: raise ValueError('K splitting is forbidden')
    return dict(M=m, bm=bm, bn=bn, bk=bk, warps=warps, wm=wm, wn=wn,
                stages=stages, schedule=schedule, swizzle=plan.get('swizzle', True))


def run(x, weight_col, scales, bias, plan, return_kernel=False):
    if x.dtype != torch.float32 or weight_col.dtype != torch.float8_e4m3fn:
        raise ValueError('Experiment preserves existing FP32 input/native W8A8 path')
    x = x.contiguous()
    k, n = weight_col.shape[1], scales.numel()
    flat = x.view(-1, k); m = flat.shape[0]
    p = normalize_plan(m, plan)
    if not weight_col.is_contiguous(): raise ValueError('Pack weight_col offline in N,K order')
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    sx = torch.empty(m, device=x.device, dtype=torch.float32)
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _quantize[(m,)](flat, q, sx, k, triton.next_power_of_2(k))
    kernel = full_m_gemm[(triton.cdiv(n, p['bn']),)](
        q, weight_col, sx, scales, bias if bias is not None else scales, y,
        m, n, k, p['bm'], p['bn'], p['bk'], p['stages'], p['warps'],
        p['wm'], p['wn'], {'full':0, 'm16_ilp':1, 'm16_preload':2}[p['schedule']], p['swizzle'], bias is not None,
        plan.get('weight_eviction', ''), plan.get('activation_eviction', ''),
        num_warps=p['warps'], ir_override=plan.get('ir_override'))
    result = y.view(*x.shape[:-1], n)
    return (result, kernel) if return_kernel else result


def run_prequantized(q, sx, shape, weight_col, scales, bias, plan, return_kernel=False):
    """Use the same BM64 kernel when an exact producer already emitted Q/S."""
    if q.dtype != torch.float8_e4m3fn or sx.dtype != torch.float32 or not q.is_contiguous():
        raise ValueError('Expected contiguous E4M3 activation and FP32 row scales')
    m, k = q.shape; n = scales.numel(); p = normalize_plan(m, plan)
    if tuple(weight_col.shape) != (n, k) or not weight_col.is_contiguous():
        raise ValueError('N,K weight/schedule mismatch')
    y = torch.empty((m, n), device=q.device, dtype=torch.float32)
    kernel = full_m_gemm[(triton.cdiv(n, p['bn']),)](
        q, weight_col, sx, scales, bias if bias is not None else scales, y,
        m, n, k, p['bm'], p['bn'], p['bk'], p['stages'], p['warps'],
        p['wm'], p['wn'], {'full':0, 'm16_ilp':1, 'm16_preload':2}[p['schedule']],
        p['swizzle'], bias is not None, plan.get('weight_eviction',''),
        plan.get('activation_eviction',''), num_warps=p['warps'],
        ir_override=plan.get('ir_override'))
    result = y.view(*shape[:-1], n)
    return (result, kernel) if return_kernel else result
