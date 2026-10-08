"""Float32 relative-position gather/add/mask/softmax, retaining full attention weights.

Native Float32 QK scores remain an input. Position projection dot width4 is
computed directly at index T-1-query+key; no batched [T,2T-1] intermediate.
All four heads, complete key range, Bool padding, and Float32 output are preserved.
"""
import triton
import triton.language as tl

@triton.jit
def position_softmax_f32(Query, Emb, Scores, Mask, T, Out, B: tl.constexpr, BLOCK: tl.constexpr):
    query = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2).to(tl.int64)
    row = (head * B + batch) * T + query
    cols = tl.arange(0, BLOCK)
    offset = head * 4 * (2 * T - 1) + T - 1 - query + cols
    q0 = tl.load(Query + row * 4)
    q1 = tl.load(Query + row * 4 + 1)
    q2 = tl.load(Query + row * 4 + 2)
    q3 = tl.load(Query + row * 4 + 3)
    e0 = tl.load(Emb + offset, cols < T, 0)
    e1 = tl.load(Emb + offset + (2 * T - 1), cols < T, 0)
    e2 = tl.load(Emb + offset + 2 * (2 * T - 1), cols < T, 0)
    e3 = tl.load(Emb + offset + 3 * (2 * T - 1), cols < T, 0)
    position = tl.fma(q3, e3, tl.fma(q2, e2, tl.fma(q1, e1, q0 * e0)))
    scores = tl.load(Scores + row * T + cols, cols < T, 0, cache_modifier='.cg')
    logits = scores + position
    padding = tl.load(Mask + batch * T + cols, cols < T, 1)
    logits = tl.where(padding, -1000.0, logits)
    logits = tl.where(cols < T, logits, -float('inf'))
    exponent = tl.exp(logits - tl.max(logits, 0))
    weights = exponent / tl.sum(exponent, 0)
    tl.store(Out + row * T + cols, weights, cols < T)
