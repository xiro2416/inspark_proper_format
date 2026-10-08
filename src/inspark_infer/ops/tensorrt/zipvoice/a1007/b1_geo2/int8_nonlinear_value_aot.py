import triton
from .int8_nonlinear_value_runtime_kernel import int8_nonlinear_value

@triton.jit
def nonlinear_value_64x32(Q, W, AS, WS, Bias, T, V, G):
    int8_nonlinear_value(Q, W, AS, WS, Bias, V, G, T, 64, 32, 64)

@triton.jit
def nonlinear_value_64x64(Q, W, AS, WS, Bias, T, V, G):
    int8_nonlinear_value(Q, W, AS, WS, Bias, V, G, T, 64, 64, 128)
