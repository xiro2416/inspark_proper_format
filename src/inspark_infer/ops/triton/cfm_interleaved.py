"""Experimental exact-recipe gate/up packing and SM120 TMA scheduling.

W[2*n] is gate and W[2*n+1] is up. Packing changes neither quantization
nor the reduction: both FP32 dot products consume the same FP8 operands.
"""
import triton
import triton.language as tl


@triton.jit
def _interleaved_raw(A, W, S, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr):
    A = tl.cast(A, tl.pointer_type(tl.float8e4nv))
    W = tl.cast(W, tl.pointer_type(tl.float8e4nv))
    _interleaved(A, W, S, OUT, M, N, K, SCALE, BM, BN, BK, False, 1)


@triton.jit
def _epilogue(acc, S, OUT, m, pair_n, N: tl.constexpr, M: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
    sg = tl.load(S + 2 * pair_n, pair_n < N, other=0.)
    su = tl.load(S + 2 * pair_n + 1, pair_n < N, other=0.)
    gate = gate * sg[None, :]
    up = up * su[None, :]
    value = gate * tl.sigmoid(gate) * up
    q = tl.minimum(tl.maximum(value * (1. / SCALE), -448.), 448.).to(tl.float8e4nv)
    tl.store(OUT + m[:, None] * N + pair_n[None, :], q,
             (m[:, None] < M) & (pair_n[None, :] < N))


@triton.jit
def _interleaved(A, W, S, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, PERSISTENT: tl.constexpr, SMS: tl.constexpr):
    mt = tl.cdiv(M, BM)
    nt = tl.cdiv(2 * N, BN)
    total = mt * nt
    first = tl.program_id(0)
    stride = SMS if PERSISTENT else total
    for tile in tl.range(first, total, stride):
        m = tile // nt * BM + tl.arange(0, BM)
        n = tile % nt * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for base in range(tl.cdiv(K, BK)):
            ki = base * BK + k
            a = tl.load(A + m[:, None] * K + ki[None, :],
                        (m[:, None] < M) & (ki[None, :] < K), other=0.)
            w = tl.load(W + n[None, :] * K + ki[:, None],
                        (n[None, :] < 2 * N) & (ki[:, None] < K), other=0.)
            acc = tl.dot(a, w, acc, max_num_imprecise_acc=0)
        pair_n = tile % nt * (BN // 2) + tl.arange(0, BN // 2)
        _epilogue(acc, S, OUT, m, pair_n, N, M, BM, BN, SCALE)


@triton.jit
def _interleaved_tma(A, W, S, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, SMS: tl.constexpr, WS: tl.constexpr):
    nt = tl.cdiv(2 * N, BN)
    total = tl.cdiv(M, BM) * nt
    for tile in tl.range(tl.program_id(0), total, SMS, warp_specialize=WS):
        mo = tile // nt * BM
        no = tile % nt * BN
        acc = tl.zeros((BM, BN), tl.float32)
        for base in range(tl.cdiv(K, BK)):
            a = A.load([mo, base * BK])
            w = W.load([no, base * BK])
            acc = tl.dot(a, tl.trans(w), acc, max_num_imprecise_acc=0)
        m = mo + tl.arange(0, BM)
        pair_n = no // 2 + tl.arange(0, BN // 2)
        _epilogue(acc, S, OUT, m, pair_n, N, M, BM, BN, SCALE)


@triton.jit
def _lookahead(A, W, S, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, SMS: tl.constexpr):
    """Mix previous tile SFU/quant with the next tile's first MMA.

    Both complete FP32 accumulators remain distinct. The epilogue never reads
    a partial accumulator. Extra live registers are deliberately measured.
    """
    nt = tl.cdiv(2*N,BN)
    total = tl.cdiv(M,BM)*nt
    prev = tl.zeros((BM,BN),tl.float32)
    prev_tile = tl.program_id(0)
    first = tl.program_id(0)
    k = tl.arange(0,BK)
    for tile in tl.range(first,total,SMS):
        m = tile//nt*BM+tl.arange(0,BM)
        n = tile%nt*BN+tl.arange(0,BN)
        acc = tl.zeros((BM,BN),tl.float32)
        for base in range(tl.cdiv(K,BK)):
            ki=base*BK+k
            a=tl.load(A+m[:,None]*K+ki[None,:],(m[:,None]<M)&(ki[None,:]<K),other=0.)
            w=tl.load(W+n[None,:]*K+ki[:,None],(n[None,:]<2*N)&(ki[:,None]<K),other=0.)
            acc=tl.dot(a,w,acc,max_num_imprecise_acc=0)
            if base==0 and tile!=first:
                pm=prev_tile//nt*BM+tl.arange(0,BM)
                pn=prev_tile%nt*(BN//2)+tl.arange(0,BN//2)
                _epilogue(prev,S,OUT,pm,pn,N,M,BM,BN,SCALE)
        prev=acc
        prev_tile=tile
    pm=prev_tile//nt*BM+tl.arange(0,BM)
    pn=prev_tile%nt*(BN//2)+tl.arange(0,BN//2)
    _epilogue(prev,S,OUT,pm,pn,N,M,BM,BN,SCALE)
