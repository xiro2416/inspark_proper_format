"""Original integer projection, Float32 epilogue, tanh(z0)*z1 value and z2 gate."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def int8_nonlinear_value(Q, W, AS, WS, Bias, V, G, T, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    a0 = tl.zeros((BM, BN), tl.int32)
    a1 = tl.zeros((BM, BN), tl.int32)
    a2 = tl.zeros((BM, BN), tl.int32)
    for block in range(tl.cdiv(512, BK)):
        k = block * BK + ks
        q = tl.load(Q + m[:, None] * 512 + k[None, :], (m[:, None] < T * 16) & (k[None, :] < 512), 0)
        w0 = tl.load(W + n[None, :] * 512 + k[:, None], (n[None, :] < 384) & (k[:, None] < 512), 0)
        w1 = tl.load(W + (n[None, :] + 384) * 512 + k[:, None], (n[None, :] < 384) & (k[:, None] < 512), 0)
        w2 = tl.load(W + (n[None, :] + 768) * 512 + k[:, None], (n[None, :] < 384) & (k[:, None] < 512), 0)
        a0 = tl.dot(q, w0, a0, out_dtype=tl.int32)
        a1 = tl.dot(q, w1, a1, out_dtype=tl.int32)
        a2 = tl.dot(q, w2, a2, out_dtype=tl.int32)
    scale = tl.load(AS)
    s0 = tl.load(WS + n, n < 384, 0) * scale
    s1 = tl.load(WS + n + 384, n < 384, 0) * scale
    s2 = tl.load(WS + n + 768, n < 384, 0) * scale
    z0 = a0.to(tl.float32) * s0[None, :] + tl.load(Bias + n, n < 384, 0)[None, :]
    z1 = a1.to(tl.float32) * s1[None, :] + tl.load(Bias + n + 384, n < 384, 0)[None, :]
    z2 = a2.to(tl.float32) * s2[None, :] + tl.load(Bias + n + 768, n < 384, 0)[None, :]
    value = libdevice.tanh(z0) * z1
    b = m % 16
    t = m // 16
    tl.store(V + ((b[:, None] * T + t[:, None]) * 384 + n[None, :]), value, (m[:, None] < T * 16) & (n[None, :] < 384))
    tl.store(G + m[:, None] * 384 + n[None, :], z2, (m[:, None] < T * 16) & (n[None, :] < 384))
