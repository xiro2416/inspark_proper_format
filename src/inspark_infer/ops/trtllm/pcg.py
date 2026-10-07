"""PCG expressed with existing PyTorch operators, with no host decisions.

This is a framework fallback for a native worker's sampler hook. It preserves
the project's overlapping acoustic-group distribution, rather than replacing
it with token-wise speculative rejection. It is not a new GPU kernel.
"""
import torch


def categorical(probabilities, uniform):
    probabilities = probabilities.float()
    cumulative = probabilities.cumsum(-1).contiguous()
    query = uniform.float()
    squeeze = query.ndim == probabilities.ndim - 1
    if squeeze:
        query = query.unsqueeze(-1)
    query = (query * cumulative[..., -1:]).contiguous()
    # right=True avoids selecting leading zero-mass entries for a zero draw.
    selected = torch.searchsorted(cumulative, query, right=True)
    selected = selected.clamp_max(probabilities.shape[-1] - 1)
    return selected.squeeze(-1) if squeeze else selected


class FrameworkPCG:
    def __init__(self, groups, temperature=0.8, group_chunk=128):
        self.groups, self.temperature = groups, float(temperature)
        self.group_chunk = int(group_chunk)
        if self.group_chunk <= 0:
            raise ValueError("group_chunk must be positive")
        self.coarse_matrix = None
        sparse = getattr(groups, "sparse", None)
        if sparse is not None:
            # Existing cuSPARSE SpMM computes all group masses without the
            # [B,G,max_group_width] intermediate. No new sparse kernel is
            # introduced. The matrix is immutable and constructed at deploy.
            members = sparse.group_members.to(device=groups.group_members.device)
            offsets = sparse.group_offsets.to(device=members.device, dtype=torch.int32)
            weights = sparse.membership_count.to(members.device)[members].float().reciprocal()
            self.coarse_matrix = torch.sparse_csr_tensor(
                offsets, members.to(torch.int32), weights,
                size=(groups.group_members.shape[0], sparse.vocab_size), device=members.device)

    def acceptance(self, logits, draft_probabilities, tokens, group_draws, accept_draws):
        g = self.groups
        b, k, vocab = logits.shape
        q = torch.softmax(logits.float() / self.temperature, -1)
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        p = draft_probabilities.float()
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        counts = g.token_group_counts[tokens]
        choices = (group_draws * counts.float()).long().clamp_min(0)
        choices = torch.minimum(choices, counts.long() - 1)
        ids = g.token_groups[tokens, choices]
        members, weights = g.group_members[ids], g.group_weights[ids]
        qm = (q.gather(2, members) * weights).sum(-1)
        pm = (p.gather(2, members) * weights).sum(-1)
        acceptance = (qm / pm.clamp_min(1e-12)).clamp_max(1)
        exact = (q.gather(2, tokens[..., None]).squeeze(-1)
                 / p.gather(2, tokens[..., None]).squeeze(-1).clamp_min(1e-12)).clamp_max(1)
        flags = accept_draws < acceptance
        packed = torch.stack((flags.float(), tokens.float(), g.group_sizes[ids].float(),
                              acceptance, exact), -1)
        return q, acceptance, packed

    @staticmethod
    def prefix_plan(packed, remaining, current, eos, max_tokens):
        if packed.shape[1:] != (7, 5):
            raise ValueError("Expected [B,7,5] acceptance decisions")
        positions = torch.arange(7, device=packed.device)[None]
        valid = positions < remaining[:, None]
        flags = packed[..., 0].bool()
        tokens = packed[..., 1].long()
        stops = valid & ((~flags) | (tokens == eos))
        first = torch.where(stops, positions, 7).amin(-1)
        at = first.clamp_max(6)
        selected_flag = flags.gather(1, at[:, None]).squeeze(1)
        selected_token = tokens.gather(1, at[:, None]).squeeze(1)
        end = (first < 7) & selected_flag & (selected_token == eos)
        count = torch.where(first < 7, first + end.long(), remaining.long())
        correction = (~end) & ((current + count) < max_tokens)
        residual = correction & (count < remaining)
        return count.int(), end, correction, residual

    def residual(self, q_block, p_block, indices, mask, draws):
        """The existing 64-attempt thinning and exact coarse fallback law.

        Work is statically shaped; no tensor-to-host checks occur here. Coarse
        enumeration is chunked to bound temporary materialization. Profiling
        must decide whether this legal framework path is competitive.
        """
        g = self.groups
        rows = torch.arange(q_block.shape[0], device=q_block.device)
        at = indices.long().clamp(0, q_block.shape[1] - 1)
        q, p = q_block[rows, at].float(), p_block[rows, at].float()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        candidates = categorical(q, draws["candidates"])
        counts = g.token_group_counts[candidates]
        choices = (draws["groups"] * counts.float()).long().clamp_min(0)
        choices = torch.minimum(choices, counts.long() - 1)
        ids = g.token_groups[candidates, choices]
        members, weights = g.group_members[ids], g.group_weights[ids]
        qr = (q[:, None, :].expand(-1, ids.shape[1], -1).gather(2, members) * weights).sum(-1).clamp_min(1e-12)
        pr = (p[:, None, :].expand(-1, ids.shape[1], -1).gather(2, members) * weights).sum(-1)
        success = draws["thin"] < (1 - pr / qr).clamp(0, 1)
        first = success.int().argmax(-1)
        chosen = ids.gather(1, first[:, None]).squeeze(1)
        selected_members, selected_weights = g.group_members[chosen], g.group_weights[chosen]
        conditional = q.gather(1, selected_members) * selected_weights
        valid = torch.arange(selected_members.shape[1], device=q.device)[None] < g.group_sizes[chosen, None]
        conditional = conditional.masked_fill(~valid, 0)
        conditional = torch.where((conditional.sum(-1) <= 1e-12)[:, None], valid.float(), conditional)
        selected = categorical(conditional, draws["fast"])
        fast_token = selected_members.gather(1, selected[:, None]).squeeze(1)
        need_exact = mask & (~success.any(-1))
        if self.coarse_matrix is not None:
            coarse = torch.sparse.mm(self.coarse_matrix, (q - p).T.contiguous()).T.clamp_min(0)
        else:
            coarse = []
            for begin in range(0, g.group_members.shape[0], self.group_chunk):
                gm = g.group_members[begin:begin + self.group_chunk]
                gw = g.group_weights[begin:begin + self.group_chunk]
                coarse.append(((q[:, gm] - p[:, gm]) * gw[None]).sum(-1).clamp_min(0))
            coarse = torch.cat(coarse, -1)
        group_mass = coarse.sum(-1)
        group = categorical(coarse, draws["coarse"])
        exact_members, exact_weights = g.group_members[group], g.group_weights[group]
        exact_conditional = q.gather(1, exact_members) * exact_weights
        member = categorical(exact_conditional, draws["member"])
        exact_token = exact_members.gather(1, member[:, None]).squeeze(1)
        exact_token = torch.where(group_mass > 1e-12, exact_token, categorical(q, draws["member"]))
        decisions = torch.stack((success.any(-1).long(), first, chosen, g.group_sizes[chosen]), -1)
        invalid = need_exact & (~torch.isfinite(exact_conditional.sum(-1)))
        return torch.where(need_exact, exact_token, fast_token), decisions, invalid

    device_batch_draws = residual
