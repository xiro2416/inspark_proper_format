"""Offline weight conversions; no compute-kernel or Triton dependency."""
import torch


def pack_weight(weight):
    """N,K FP32/BF16 -> padded K,N E4M3 weights and per-output FP32 scales.

    This layout is consumed by the explicit FP8 Triton kernels. Packing does
    not establish hardware support or model-quality acceptance.
    """
    w = weight.detach().float()
    scale = (w.abs().amax(1) / 448).clamp_min(1e-12)
    n, k = w.shape
    packed = torch.zeros(k, ((n + 31) // 32) * 32,
                         device=w.device, dtype=torch.float8_e4m3fn)
    packed[:, :n].copy_((w / scale[:, None]).to(torch.float8_e4m3fn).t())
    return packed, scale
