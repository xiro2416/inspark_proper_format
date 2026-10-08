import triton
from .online_branch_stats_runtime_kernel import online_branch_stats

@triton.jit
def normal_tile_aot(Q, K, PQ, E, Mask, V, T, O):
    online_branch_stats(Q, K, PQ, E, Mask, V, O, O, T, False, 64, 32, 'ieee', 0)

@triton.jit
def normal_stats_write_aot(Q, K, PQ, E, Mask, V, T, O, Stats):
    online_branch_stats(Q, K, PQ, E, Mask, V, O, Stats, T, False, 64, 32, 'ieee', 1)

@triton.jit
def normal_stats_read_aot(Q, K, PQ, E, Mask, V, Stats, T, O):
    online_branch_stats(Q, K, PQ, E, Mask, V, O, Stats, T, False, 64, 32, 'ieee', 2)
