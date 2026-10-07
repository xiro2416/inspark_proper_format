"""Common GPU round controller for FP8 and INT8 model providers.

Only existing PyTorch/official RNN operations are used. This controller is a
framework adapter and is never advertised as NVIDIA's unmodified executor.
Providers own fixed dense request cache slots for the lifetime of one head.
"""
import hashlib
import time

import torch

from .pcg import FrameworkPCG, categorical


class RequestDraws:
    """Admission-time, request-owned RNG; replay indexes counters on device."""
    width = 217  # proposal7 + group7 + accept7 + 3*64 residual + four scalars

    def __init__(self, seeds, max_rounds, device):
        self.values = torch.empty(len(seeds), max_rounds, self.width, device=device)
        self.reset(seeds)

    def reset(self, seeds):
        if len(seeds) != self.values.shape[0]:
            raise ValueError("RNG batch shape cannot change after capture")
        for row, seed in enumerate(seeds):
            # Separate the head stream from eager prefill's generator, which
            # was already seeded with the public request seed. Reusing that
            # seed here would restart the same random stream and correlate
            # proposal/acceptance draws with prefill sampling.
            public = (int(seed) & ((1 << 64) - 1)).to_bytes(8, "little")
            private = int.from_bytes(hashlib.blake2b(public + b"a0924/framework-dspark/head/v1", digest_size=8).digest(), "little")
            generator = torch.Generator(device=self.values.device).manual_seed(private)
            self.values[row].uniform_(generator=generator)

    def at(self, rounds):
        row = torch.arange(rounds.numel(), device=rounds.device)
        return self.values[row, rounds.long().clamp_max(self.values.shape[1] - 1)]


class FrameworkRoundRuntime:
    """Executable Q7->Q8->PCG->commit loop with one parent CUDA Graph.

    draft_provider(anchor, absolute_positions[B,7], draft_lengths) returns
    hidden/base. target_provider(tokens[B,8], absolute_positions[B,8], past)
    returns logits/selected/final and may tentatively write all eight K/Vs.
    context_writer(selected[B,8,6400], old_draft_lengths, committed_counts)
    stores only committed positions. All provider calls use the current stream
    and must contain no custom math kernels for an official-framework baseline.
    """
    backend = "framework_dspark_adapter"

    def __init__(self, *, proposal, groups, draft_provider, target_provider,
                 context_writer, initial_tokens, token_lengths, past_lengths,
                 draft_lengths, mel_lengths, seeds, eos, kv_capacity,
                 max_tokens=1500, max_rounds=64, ready_token_count=31):
        if initial_tokens.ndim != 2:
            raise ValueError("initial_tokens must be padded [B,T]")
        self.device, self.batch = initial_tokens.device, initial_tokens.shape[0]
        if len(seeds) != self.batch:
            raise ValueError("One independently owned seed is required per request")
        self.proposal = proposal
        self.pcg = FrameworkPCG(groups)
        self.draft_provider, self.target_provider = draft_provider, target_provider
        self.context_writer = context_writer
        self.eos, self.kv_capacity = int(eos), int(kv_capacity)
        self.max_tokens, self.max_rounds = int(max_tokens), int(max_rounds)
        self.ready_token_count = int(ready_token_count)
        self.tokens = initial_tokens.new_zeros(self.batch, self.max_tokens + 8)
        self.tokens[:, :initial_tokens.shape[1]].copy_(initial_tokens)
        self.token_lengths = token_lengths.to(device=self.device, dtype=torch.int32).clone()
        self.past = past_lengths.to(device=self.device, dtype=torch.int32).clone()
        self.draft_lengths = draft_lengths.to(device=self.device, dtype=torch.int32).clone()
        self.mel_lengths = mel_lengths.to(device=self.device, dtype=torch.int32).clone()
        self.last = initial_tokens.gather(1, (self.token_lengths.long() - 1)[:, None]).squeeze(1).clone()
        self.done = (self.last == self.eos) | (self.token_lengths >= self.max_tokens)
        self.ready = self.done | (self.token_lengths >= self.ready_token_count)
        self.rounds = torch.zeros(self.batch, device=self.device, dtype=torch.int32)
        self.accepted = torch.full((self.batch, self.max_rounds), -1, device=self.device, dtype=torch.int32)
        self.committed = torch.zeros_like(self.rounds)
        # Providers can alias this stable tensor to suppress inactive cache
        # writes. It is updated in place before either model call.
        self.active = torch.zeros(self.batch, device=self.device, dtype=torch.bool)
        self.failures = torch.zeros((), device=self.device, dtype=torch.int32)
        self.capacity_failures = torch.zeros_like(self.failures)
        self.status = torch.zeros_like(self.failures)
        self.draws = RequestDraws(seeds, self.max_rounds, self.device)
        self.step7 = torch.arange(7, device=self.device)[None]
        self.step8 = torch.arange(8, device=self.device)[None]
        self.graph = None
        self.graph_burst = 1

    def reset(self, *, initial_tokens, token_lengths, past_lengths, draft_lengths,
              mel_lengths, seeds):
        """Admit another group into captured storage without recapturing.

        The provider imports its valid prefixes first. Shapes and cache bucket
        are deployment-time invariants; selecting another (B,K) runtime happens
        before this call. No tensor storage referenced by the Graph is replaced.
        """
        if initial_tokens.ndim != 2 or initial_tokens.shape[0] != self.batch:
            raise ValueError("Admission batch must match the captured runtime")
        if initial_tokens.shape[1] > self.tokens.shape[1]:
            raise ValueError("Initial code buffer exceeds runtime capacity")
        self.tokens.zero_()
        self.tokens[:, :initial_tokens.shape[1]].copy_(initial_tokens)
        for destination, source in ((self.token_lengths, token_lengths), (self.past, past_lengths),
                                    (self.draft_lengths, draft_lengths), (self.mel_lengths, mel_lengths)):
            if source.shape != (self.batch,):
                raise ValueError("Admission lengths must be [B]")
            destination.copy_(source)
        self.last.copy_(self.tokens.gather(1, (self.token_lengths.long() - 1)[:, None]).squeeze(1))
        self.done.copy_((self.last == self.eos) | (self.token_lengths >= self.max_tokens))
        self.ready.copy_(self.done | (self.token_lengths >= self.ready_token_count))
        self.rounds.zero_(); self.accepted.fill_(-1); self.committed.zero_(); self.active.zero_()
        self.failures.zero_(); self.capacity_failures.zero_(); self.status.zero_()
        self.draws.reset(seeds)

    def _commit(self, proposed, correction, count, end, has_correction, active):
        count = torch.where(active, count, 0)
        has_correction = has_correction & active
        index = self.token_lengths.long()[:, None] + self.step8
        values = torch.cat((proposed, correction[:, None]), 1)
        values = torch.where(self.step8 < count[:, None], values, correction[:, None])
        valid = active[:, None] & ((self.step8 < count[:, None])
                                   | (has_correction[:, None] & (self.step8 == count[:, None])))
        values = torch.where(valid, values, self.tokens.gather(1, index))
        self.tokens.scatter_(1, index, values)
        new_lengths = self.token_lengths + count + has_correction.int()
        accepted_last = proposed.gather(1, (count.long() - 1).clamp(0, 6)[:, None]).squeeze(1)
        new_last = torch.where(has_correction, correction, torch.where(count > 0, accepted_last, self.last))
        history_index = self.rounds.long().clamp_max(self.max_rounds - 1)[:, None]
        history = torch.where(active, count, self.accepted.gather(1, history_index).squeeze(1))
        self.accepted.scatter_(1, history_index, history[:, None])
        self.rounds.add_(active.int())
        self.token_lengths.copy_(new_lengths)
        self.committed.copy_(torch.where(active, count + 1, 0))
        self.past.add_(self.committed)
        self.last.copy_(new_last)
        self.done.logical_or_((end & active) | (active & ((new_last == self.eos) | (new_lengths >= self.max_tokens))))

    def step(self):
        fits = (self.past + 8 <= self.kv_capacity) & (self.draft_lengths + 8 <= self.kv_capacity)
        self.active.copy_((~self.ready) & fits & (self.rounds < self.max_rounds))
        active = self.active
        self.capacity_failures.add_(((~self.ready) & (~fits)).sum().to(torch.int32))
        first = self.past + 1 - self.mel_lengths
        # Invalid rows are masked from all commits, but still need safe addresses
        # when a fixed-shape provider evaluates the entire batch.
        hidden, base = self.draft_provider(self.last, first[:, None] + self.step7,
                                           self.draft_lengths.clamp_max(self.kv_capacity - 8))
        random = self.draws.at(self.rounds)
        proposed, p, _ = self.proposal.sample_uniform(hidden, base, random[:, :7], self.last)
        verify_tokens = torch.cat((self.last[:, None], proposed), 1)
        logits, selected, _ = self.target_provider(verify_tokens, first[:, None] + self.step8,
                                                   self.past.clamp_max(self.kv_capacity - 8))
        self.failures.add_(((~torch.isfinite(logits).flatten(1).all(-1)) & active).sum().to(torch.int32))
        q, _, packed = self.pcg.acceptance(logits[:, :7], p, proposed, random[:, 7:14], random[:, 14:21])
        remaining = (self.max_tokens - self.token_lengths).clamp(0, 7)
        count, end, has_correction, needs_residual = self.pcg.prefix_plan(
            packed, remaining, self.token_lengths, self.eos, self.max_tokens)
        residual_draws = {"candidates": random[:, 21:85], "groups": random[:, 85:149],
                          "thin": random[:, 149:213], "fast": random[:, 213],
                          "coarse": random[:, 214], "member": random[:, 215]}
        residual, _, invalid = self.pcg.residual(q, p, count, needs_residual & active, residual_draws)
        self.failures.add_(invalid.sum().to(torch.int32))
        row = torch.arange(self.batch, device=self.device)
        probability = torch.softmax(logits[row, count.long().clamp_max(7)].float() / .8, -1)
        bonus = categorical(probability, random[:, 216])
        correction = torch.where(needs_residual, residual, bonus)
        self._commit(proposed, correction, count, end, has_correction, active)
        self.context_writer(selected, self.draft_lengths, self.committed)
        self.draft_lengths.add_(self.committed)
        self.ready.copy_(self.done | (self.token_lengths >= self.ready_token_count))
        self.status.copy_(self.ready.all().int() | ((self.failures > 0).int() << 1)
                          | ((self.capacity_failures > 0).int() << 2))

    def capture(self, burst_rounds=2, warmups=2):
        """Capture inactive rounds; valid prefixes and RNG counters stay intact.

        Providers may overwrite speculative tails. Their contract prohibits
        writing any existing committed prefix in such a round.
        """
        if self.device.type != "cuda":
            raise ValueError("CUDA Graph capture requires a CUDA runtime")
        if burst_rounds not in (1, 2, 4):
            raise ValueError("Supported capture burst sizes: 1/2/4")
        original_ready = self.ready.clone()
        self.ready.fill_(True)
        original_done = self.done.clone()
        # step recomputes ready from done, so keep all rows inactive throughout
        # a multi-round capture while preserving their actual terminal state.
        self.done.fill_(True)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                self.step()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for _ in range(burst_rounds):
                self.step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.done.copy_(original_done)
        self.ready.copy_(original_ready)
        self.status.zero_(); self.failures.zero_(); self.capacity_failures.zero_()
        self.graph_burst = burst_rounds

    def run(self):
        launched, reads, wait_ms = 0, 0, 0.0
        compacted=None
        deferred=False;late_rounds=0
        parent_replays=verify_replays=proposal_replays=0
        while launched < self.max_rounds:
            late=(getattr(self,'verify_graph',None) is not None and
                  launched>=getattr(self,'late_verify_after',12))
            if late:
                self.verify_graph.replay();launched+=1;late_rounds+=1;verify_replays+=1;deferred=True
            elif self.graph is not None:
                self.graph.replay(); launched += self.graph_burst;parent_replays+=1
            else:
                self.step(); launched += 1
            started = time.perf_counter(); status = int(self.status.item())
            wait_ms += (time.perf_counter() - started) * 1000; reads += 1
            if status & 6:
                raise RuntimeError(f"Framework DSpark failed: status={status}")
            tail=getattr(self,'compact_tail',None)
            threshold=getattr(tail,'after',12) if tail is not None else 12
            if tail is not None and launched>threshold and getattr(tail,'fallback',None) is not None:
                tail=tail.fallback;threshold=tail.after
            if not status&1 and tail is not None and (launched==threshold or
                    launched>threshold and getattr(self,'verify_graph',None) is not None):
                compacted=(tail.run_if_eligible(self,need_proposal=True) if deferred
                           else tail.run_if_eligible(self))
                if compacted is not None:
                    launched+=compacted['launched_rounds']
                    reads+=compacted['status_reads']
                    wait_ms+=compacted['status_wait_ms']
                    status=1
                    deferred=False
            if status & 1:
                return {"backend": self.backend, "launched_rounds": launched,
                        "late_verify_rounds":late_rounds,"skipped_final_proposals":int(deferred),
                        "graph_replays":parent_replays+verify_replays+(compacted or {}).get('graph_replays',0),
                        "parent_graph_replays":parent_replays,"verify_graph_replays":verify_replays,
                        "proposal_graph_replays":proposal_replays,
                        "compacted_tail": compacted,
                        "status_reads": reads, "status_wait_ms": wait_ms,
                        "graph": self.graph is not None, "native_executor": False,
                        "rng": "request_scoped_domain_separated_preallocated_uniform_v1",
                        "pcg_coarse": "torch_sparse_csr" if self.pcg.coarse_matrix is not None else "torch_gather",
                        "proposal_provenance": getattr(self.proposal, "provenance", "test_provider")}
            if deferred:
                # Acceptance/context commit has finished. Only unfinished
                # heads need a next proposal; reuse the existing prime Graph.
                self.prime_graph.replay();proposal_replays+=1;deferred=False
        raise RuntimeError("Framework DSpark reached the bounded round limit")

    def result(self):
        """One compact host transfer; speech codes remain device resident."""
        metadata = torch.stack((self.token_lengths, self.past, self.draft_lengths,
                                self.rounds, self.done.int(), self.last.int(), self.ready.int()), 1)
        return {"tokens": self.tokens, "accepted": self.accepted,
                "metadata": metadata.cpu().tolist(),
                "metadata_columns": ("token_length", "past_length", "draft_length",
                                     "rounds", "done", "last", "ready"),
                "backend": self.backend, "native_executor": False}
