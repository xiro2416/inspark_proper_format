"""AOT ABI wrapper for original INT8 DW boundary."""
import triton
import triton.language as tl
from .dw_int8_probe_kernel import dw_int8

@triton.jit
def dw_int8_aot(X, W, WS, Bias, T, Out, XS: tl.constexpr, K: tl.constexpr):
    dw_int8(X, W, WS, Bias, Out, T, XS, K, 512, 256)
