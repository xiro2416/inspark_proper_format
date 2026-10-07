"""Opt-in fixed B8 tail of a B64 native-worker head; existing torch ops only.

The mapping is selected once, after round 12. Uniforms and proposal probability
state migrate with each request, rather than being resampled/reprimed. Remaining
ready rows are inert padding in the B8 Graph. This is not in-flight batching.
"""
import torch


def state_pairs(runtime):
    pairs=[(runtime.tokens,0),(runtime.accepted,0),(runtime.past,0),
           (runtime.draft_lengths,0),(runtime.mel_lengths,0),(runtime.done,0),
           (runtime.active,0),(runtime.next_tokens,0),(runtime.draws.values,0),
           (runtime.provider.target_cache,2),(runtime.provider.target_keep,0),
           (runtime.worker._ctx_k_buf,0),(runtime.worker._ctx_v_buf,0)]
    pairs += [(getattr(runtime.policy,name),0) for name in
              ('lengths','rounds','last','ready','last_committed','tokens',
               'probabilities','has_proposal')]
    return pairs


def transfer_state(source, destination, indices, *, restore=False):
    """Copy request-owned state, including rejected proposal/RNG bookkeeping.

    Empty padding becomes ready and consumes no request draw or KV commit.
    Native arenas contain one additional padding slot; only real rows migrate.
    """
    # Production indices come from nonzero and are unique by construction.
    # Keep CPU fixtures checked without adding another GPU metadata read.
    if indices.device.type=='cpu' and len(set(indices.tolist())) != indices.numel():
        raise ValueError('Compacted request slots must be distinct')
    n=indices.numel()
    for (src,axis),(dst,dst_axis) in zip(state_pairs(source),state_pairs(destination)):
        if axis != dst_axis:raise ValueError('State geometry mismatch')
        if restore:
            dst.index_copy_(axis,indices,src.narrow(axis,0,n))
        else:
            dst.zero_()
            dst.narrow(axis,0,n).copy_(src.index_select(axis,indices))
    if not restore:
        destination.policy.ready[n:].fill_(True)
        destination.policy.lengths[n:].fill_(1)
        destination.done[n:].fill_(True)
        destination.failures.zero_();destination.capacity_failures.zero_();destination.status.zero_()


class CompactTail:
    def __init__(self, runtime, after=12):
        self.runtime=runtime
        self.after=after
        self.calls=0
        self.rows=0
        self.last=None

    def run_if_eligible(self, source, *, need_proposal=False):
        indices=torch.nonzero(~source.ready,as_tuple=False).flatten()
        if not 0<indices.numel()<=self.runtime.batch:return None
        transfer_state(source,self.runtime,indices)
        if need_proposal:
            # Verify-only main rounds have committed context but no proposal.
            # Generate that proposal at the smaller batch with the transferred
            # request draws/round counters; RNN initial state remains zero.
            self.runtime.prime_graph.replay()
        result=self.runtime.run()
        transfer_state(self.runtime,source,indices,restore=True)
        source.failures.add_(self.runtime.failures)
        source.capacity_failures.add_(self.runtime.capacity_failures)
        source.status.copy_(self.runtime.status)
        self.calls+=1;self.rows+=indices.numel()
        self.last=dict(result,active_rows=indices.numel(),tail_batch=self.runtime.batch,
                       entry_prime_enqueues=int(need_proposal),
                       prime_enqueues=int(need_proposal)+(result.get('compacted_tail') or {}).get('prime_enqueues',0),
                       draft_enqueues=result['launched_rounds']+int(need_proposal),
                       row_rounds=self.runtime.batch*result['launched_rounds'])
        return self.last
