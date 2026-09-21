"""Offline-selected launch policy for the unchanged slot attention arithmetic.

The KV64 recurrence, TF32x3 dot products, masks and reduction ordering stay in
the original kernel. Stage count is a compiler request, not proof of a real
pipeline: inspect TTGIR/SASS and measure before selecting a policy.
"""
import torch
from acc_infer_clear.kernels.kv_attention import _attention, attention as fallback


def candidate(q, pk, pv, keep, slots, lengths, limit, *, stages=2, warps=4,
              consumer_layout=False):
    b, h, n, d = q.shape
    if d != 64 or n != 8:
        return fallback(q, pk, pv, keep, slots, lengths, limit, consumer_layout)
    out = torch.empty((b, n, h, d) if consumer_layout else (b, h, n, d),
                      device=q.device, dtype=q.dtype)
    _attention[(b, h)](q, pk, pv, keep, slots, lengths, out, *q.stride(),
                       h, n, d, pk.shape[-2], limit, 64, consumer_layout,
                       num_warps=warps, num_stages=stages)
    return out


class AttentionPolicy:
    """Explicit offline policy; no tuning or new specialization during service.

    Invoke before existing graph preparation. The runtime's existing sealed
    graph dispatch, not this function, enforces the no-online-compilation rule.
    Missing batch/KV keys retain the existing implementation.
    """
    def __init__(self, choices):
        self.choices = {(int(b), int(limit)): (int(stages), int(warps))
                        for b, limit, stages, warps in choices}
        for (b, limit), (stages, warps) in self.choices.items():
            if b not in range(1, 9) or limit not in (64, 128, 256, 512, 1024, 2048):
                raise ValueError('Unvalidated slot attention bucket')
            if stages not in range(1, 6) or warps not in (4, 8):
                raise ValueError('Unsupported offline launch configuration')

    def __call__(self, q, pk, pv, keep, slots, lengths, limit,
                 consumer_layout=False):
        choice = self.choices.get((q.shape[0], limit))
        if choice is None:
            return fallback(q, pk, pv, keep, slots, lengths, limit,
                            consumer_layout)
        return candidate(q, pk, pv, keep, slots, lengths, limit,
                         stages=choice[0], warps=choice[1],
                         consumer_layout=consumer_layout)
