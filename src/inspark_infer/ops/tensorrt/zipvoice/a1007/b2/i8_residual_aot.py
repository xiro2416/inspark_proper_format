import triton
from .i8_residual_runtime_kernel import i8_residual

@triton.jit
def residual_64x64(Q, W, AS, WS, Bias, R, T, Y):
    i8_residual(Q, W, AS, WS, Bias, R, Y, T * 2, 48, 512, 64, 64, 64)

@triton.jit
def residual_128x64(Q, W, AS, WS, Bias, R, T, Y):
    i8_residual(Q, W, AS, WS, Bias, R, Y, T * 2, 48, 512, 128, 64, 64)
