"""Float32 storage GEMM with explicitly selected existing TF32 or authorized TF32x3 math with original bias and stable activation epilogue."""
import triton
import triton.language as tl

@triton.jit
def linear_f32_activation(X, W, Bias, Y, M, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SHIFT: tl.constexpr, SLOPE: tl.constexpr, OFFSET: tl.constexpr, MATH_MODE: tl.constexpr):
    pid = tl.program_id(0)
    count_m = tl.cdiv(M, BM)
    count_n = tl.cdiv(N, BN)
    group = pid // (8 * count_n)
    first_m = group * 8
    size_m = tl.minimum(count_m - first_m, 8)
    pm = first_m + pid % (8 * count_n) % size_m
    pn = pid % (8 * count_n) // size_m
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for b in range(triton.cdiv(K, BK)):
        offset = b * BK + ks
        x = tl.load(X + rows[:, None] * K + offset[None, :], (rows[:, None] < M) & (offset[None, :] < K), 0)
        w = tl.load(W + offset[:, None] * N + cols[None, :], (offset[:, None] < K) & (cols[None, :] < N), 0)
        if MATH_MODE == 'tf32':
            x = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[x], dtype=tl.float32, is_pure=True, pack=1)
            w = tl.inline_asm_elementwise('cvt.rna.tf32.f32 $0, $1;', constraints='=f,f', args=[w], dtype=tl.float32, is_pure=True, pack=1)
        acc = tl.dot(x, w, acc, input_precision=MATH_MODE)
    z = acc + tl.load(Bias + cols, cols < N, 0)[None, :]
    u = z - SHIFT
    activated = tl.maximum(u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u))) - SLOPE * z - OFFSET
    tl.store(Y + rows[:, None] * N + cols[None, :], activated, (rows[:, None] < M) & (cols[None, :] < N))
