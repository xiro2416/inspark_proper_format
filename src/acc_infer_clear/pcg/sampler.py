from __future__ import annotations
from dataclasses import dataclass
import torch
from .asg import AcousticGroups, DenseAcousticGroups

@dataclass(frozen=True)
class PCGDecision:
    accepted: bool
    group_id: int
    group_size: int
    draft_group_probability: float
    target_group_probability: float
    acceptance_probability: float
    exact_token_acceptance_probability: float

@dataclass(frozen=True)
class PCGBlockDecision:
    accepted: torch.Tensor
    group_ids: torch.Tensor
    group_sizes: torch.Tensor
    draft_group_probabilities: torch.Tensor
    target_group_probabilities: torch.Tensor
    acceptance_probabilities: torch.Tensor
    exact_token_acceptance_probabilities: torch.Tensor

def _normalized(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 1:
        raise ValueError('probabilities must be one-dimensional')
    if bool((probabilities < 0).any().item()):
        raise ValueError('probabilities must be non-negative')
    return probabilities / probabilities.sum().clamp_min(1e-12)

@torch.inference_mode()
def pcg_accept(target_probabilities: torch.Tensor, draft_probabilities: torch.Tensor, draft_token: int | torch.Tensor, groups: AcousticGroups, *, generator=None) -> PCGDecision:
    """Sample an ASG coupled to the draft token and apply PCG acceptance."""
    q = _normalized(target_probabilities.float())
    p = _normalized(draft_probabilities.float())
    token = int(draft_token)
    group = groups.sample_group(token, generator=generator)
    p_group = groups.group_mass(p, group).clamp_min(1e-12)
    q_group = groups.group_mass(q, group)
    acceptance = (q_group / p_group).clamp(max=1.0)
    draw = torch.rand((), device=p.device, generator=generator)
    exact = (q[token] / p[token].clamp_min(1e-12)).clamp(max=1.0)
    members = groups.members(group)
    return PCGDecision(accepted=bool((draw < acceptance).item()), group_id=int(group), group_size=int(members.numel()), draft_group_probability=float(p_group), target_group_probability=float(q_group), acceptance_probability=float(acceptance), exact_token_acceptance_probability=float(exact))

@torch.inference_mode()
def pcg_accept_block(target_probabilities: torch.Tensor, draft_probabilities: torch.Tensor, draft_tokens: torch.Tensor, groups: DenseAcousticGroups, *, generator=None) -> PCGBlockDecision:
    """Vectorized group coupling and acceptance for all K proposal positions."""
    if target_probabilities.ndim != 2 or draft_probabilities.ndim != 2:
        raise ValueError('block probabilities must have shape [K, vocab]')
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError('target and draft block shapes disagree')
    tokens = draft_tokens.reshape(-1).long()
    if tokens.numel() != target_probabilities.shape[0]:
        raise ValueError('one draft token is required per probability row')
    q = target_probabilities.float()
    p = draft_probabilities.float()
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    group_ids = groups.sample_groups(tokens, generator=generator)
    p_group = groups.group_masses(p, group_ids).clamp_min(1e-12)
    q_group = groups.group_masses(q, group_ids)
    acceptance = (q_group / p_group).clamp(max=1.0)
    exact = (q.gather(1, tokens[:, None]).squeeze(1) / p.gather(1, tokens[:, None]).squeeze(1).clamp_min(1e-12)).clamp(max=1.0)
    draws = torch.rand(acceptance.shape, device=acceptance.device, generator=generator)
    return PCGBlockDecision(accepted=draws < acceptance, group_ids=group_ids, group_sizes=groups.group_sizes[group_ids], draft_group_probabilities=p_group, target_group_probabilities=q_group, acceptance_probabilities=acceptance, exact_token_acceptance_probabilities=exact)

def _sample_token_conditioned_on_group(target_probabilities: torch.Tensor, group_id: int | torch.Tensor, groups: AcousticGroups, *, generator=None) -> torch.Tensor:
    members = groups.members(group_id)
    weights = target_probabilities[members] / groups.membership_count[members].to(target_probabilities.dtype)
    if float(weights.sum()) <= 1e-12:
        weights = torch.ones_like(weights)
    index = torch.multinomial(weights, 1, generator=generator)
    return members[index].squeeze(0)

@torch.inference_mode()
def sample_pcg_residual(target_probabilities: torch.Tensor, draft_probabilities: torch.Tensor, groups: AcousticGroups, *, generator=None, max_thinning_attempts: int=64) -> tuple[torch.Tensor, int, bool, int]:
    """Sample the PCG rejection branch exactly.

    Returns ``(token, group_id, used_enumeration_fallback, attempts)``.  The
    normal path samples groups from Qc and thins by ``[1 - Pc/Qc]+``.  If that
    becomes inefficient, exact enumeration of the coarse residual is used;
    this changes cost, not the distribution.
    """
    q = _normalized(target_probabilities.float())
    p = _normalized(draft_probabilities.float())
    for attempt in range(1, max_thinning_attempts + 1):
        target_token = torch.multinomial(q, 1, generator=generator).squeeze(0)
        group = groups.sample_group(target_token, generator=generator)
        q_group = groups.group_mass(q, group).clamp_min(1e-12)
        p_group = groups.group_mass(p, group)
        keep = (1.0 - p_group / q_group).clamp(min=0.0, max=1.0)
        if bool((torch.rand((), device=q.device, generator=generator) < keep).item()):
            token = _sample_token_conditioned_on_group(q, group, groups, generator=generator)
            return (token, int(group), False, attempt)
    q_groups = groups.all_group_masses(q)
    p_groups = groups.all_group_masses(p)
    residual = (q_groups - p_groups).clamp_min(0.0)
    if float(residual.sum()) <= 1e-12:
        token = torch.multinomial(q, 1, generator=generator).squeeze(0)
        group = groups.sample_group(token, generator=generator)
    else:
        group = torch.multinomial(residual, 1, generator=generator).squeeze(0)
        token = _sample_token_conditioned_on_group(q, group, groups, generator=generator)
    return (token, int(group), True, max_thinning_attempts)

@torch.inference_mode()
def sample_pcg_residual_vectorized(target_probabilities: torch.Tensor, draft_probabilities: torch.Tensor, groups: DenseAcousticGroups, *, generator=None, max_thinning_attempts: int=64) -> tuple[torch.Tensor, int, bool, int]:
    """Vectorized thinning: draw all candidates, then take the first success."""
    q = _normalized(target_probabilities.float())
    p = _normalized(draft_probabilities.float())
    candidates = torch.multinomial(q, max_thinning_attempts, replacement=True, generator=generator)
    group_ids = groups.sample_groups(candidates, generator=generator)
    q_rows = q.unsqueeze(0).expand(max_thinning_attempts, -1)
    p_rows = p.unsqueeze(0).expand(max_thinning_attempts, -1)
    q_group = groups.group_masses(q_rows, group_ids).clamp_min(1e-12)
    p_group = groups.group_masses(p_rows, group_ids)
    keep = (1.0 - p_group / q_group).clamp(min=0.0, max=1.0)
    success = torch.rand(keep.shape, device=keep.device, generator=generator) < keep
    indices = torch.nonzero(success, as_tuple=False).flatten()
    if indices.numel():
        first = int(indices[0])
        group = group_ids[first]
        token = _sample_token_conditioned_on_group(q, group, groups.sparse, generator=generator)
        return (token, int(group), False, first + 1)
    q_groups = groups.sparse.all_group_masses(q)
    p_groups = groups.sparse.all_group_masses(p)
    residual = (q_groups - p_groups).clamp_min(0.0)
    if float(residual.sum()) <= 1e-12:
        token = torch.multinomial(q, 1, generator=generator).squeeze(0)
        group = groups.sample_groups(token.reshape(1), generator=generator)[0]
    else:
        group = torch.multinomial(residual, 1, generator=generator).squeeze(0)
        token = _sample_token_conditioned_on_group(q, group, groups.sparse, generator=generator)
    return (token, int(group), True, max_thinning_attempts)

