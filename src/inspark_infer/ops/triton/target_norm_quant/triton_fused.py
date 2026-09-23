"""Portable Triton LayerNorm+FP8 quantization candidate for SM89 and newer."""
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_quant(
    X, WEIGHT, BIAS, Q, SCALE, Y,
    N: tl.constexpr, EPS: tl.constexpr, EMIT_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N
    weight = tl.load(WEIGHT + col, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + col, mask=mask, other=0.0).to(tl.float32)
    y = centered * tl.rsqrt(variance + EPS) * weight + bias
    amax = tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0)
    scale = tl.maximum(amax / 448.0, 1.0e-12)
    tl.store(Q + row * N + col, y / scale, mask=mask)
    tl.store(SCALE + row, scale)
    if EMIT_Y:
        tl.store(Y + row * N + col, y, mask=mask)


def fused(x, weight, bias, eps=1e-5, emit_y=False):
    if x.device.type != "cuda" or weight.device != x.device or bias.device != x.device:
        raise ValueError("CUDA tensors on one device are required")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError("FP32 inputs are required")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("Contiguous inputs are required")
    n = x.shape[-1]
    if n != 1280 or weight.numel() != n or bias.numel() != n or x.numel() == 0:
        raise ValueError("Validated interface is non-empty N=1280 LayerNorm")
    flat = x.view(-1, n)
    q = torch.empty_like(flat, dtype=torch.float8_e4m3fn)
    scale = torch.empty(flat.shape[0], device=x.device, dtype=torch.float32)
    y = torch.empty_like(x) if emit_y else torch.empty(0, device=x.device, dtype=x.dtype)
    _layernorm_quant[(flat.shape[0],)](
        flat, weight, bias, q, scale, y,
        N=n, EPS=float(eps), EMIT_Y=emit_y,
        BLOCK=triton.next_power_of_2(n), num_warps=8,
        enable_fp_fusion=False,
    )
    return q, scale, y if emit_y else None
