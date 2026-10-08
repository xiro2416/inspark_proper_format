import triton
from .online_nonlinear_rna_runtime_kernel import online_nonlinear_rna

@triton.jit
def nonlinear_rna_aot(Q, K, PQ, E, Mask, V, T, O):
    online_nonlinear_rna(Q, K, PQ, E, Mask, V, O, T, True, 32, 16, 'tf32', True)

@triton.jit
def nonlinear_rna_quarter_aot(Q, K, PQ, E, Mask, V, T, O):
    online_nonlinear_rna(Q, K, PQ, E, Mask, V, O, T, True, 32, 16, 'tf32', True)
