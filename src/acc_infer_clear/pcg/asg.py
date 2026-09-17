from __future__ import annotations
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

@dataclass(frozen=True)
class AcousticGroups:
    """Sparse overlapping acoustic-similarity groups.

    ``group_offsets/group_members`` is group -> token CSR.  The reverse
    ``token_offsets/token_groups`` is token -> group CSR.  A token's mass is
    split equally over all groups containing it, exactly as defined by PCG.
    """
    group_offsets: torch.Tensor
    group_members: torch.Tensor
    token_offsets: torch.Tensor
    token_groups: torch.Tensor
    membership_count: torch.Tensor
    metadata: dict[str, Any]

    @property
    def vocab_size(self) -> int:
        return int(self.membership_count.numel())

    @property
    def num_groups(self) -> int:
        return int(self.group_offsets.numel() - 1)

    @property
    def device(self) -> torch.device:
        return self.group_members.device

    def to(self, device: torch.device | str) -> 'AcousticGroups':
        return replace(self, group_offsets=self.group_offsets.cpu(), group_members=self.group_members.to(device), token_offsets=self.token_offsets.cpu(), token_groups=self.token_groups.to(device), membership_count=self.membership_count.to(device))

    def validate(self) -> None:
        tensors = (self.group_offsets, self.group_members, self.token_offsets, self.token_groups, self.membership_count)
        if any((tensor.dtype != torch.long for tensor in tensors)):
            raise TypeError('all ASG index tensors must use torch.long')
        if self.group_offsets.ndim != 1 or self.token_offsets.ndim != 1:
            raise ValueError('CSR offsets must be one-dimensional')
        if int(self.group_offsets[0]) != 0 or int(self.token_offsets[0]) != 0:
            raise ValueError('CSR offsets must start at zero')
        if int(self.group_offsets[-1]) != self.group_members.numel():
            raise ValueError('group CSR is inconsistent')
        if int(self.token_offsets[-1]) != self.token_groups.numel():
            raise ValueError('token CSR is inconsistent')
        if self.group_members.numel() != self.token_groups.numel():
            raise ValueError('forward and reverse CSR must contain equal memberships')
        if self.membership_count.numel() + 1 != self.token_offsets.numel():
            raise ValueError('membership_count has the wrong vocabulary size')
        if bool((self.membership_count <= 0).any().item()):
            raise ValueError('every token must belong to at least one group')
        reverse_counts = self.token_offsets[1:] - self.token_offsets[:-1]
        if not torch.equal(reverse_counts.cpu(), self.membership_count.cpu()):
            raise ValueError('membership_count disagrees with reverse CSR')
        if self.group_members.numel():
            if int(self.group_members.min()) < 0 or int(self.group_members.max()) >= self.vocab_size:
                raise ValueError('group member token is out of range')
            if int(self.token_groups.min()) < 0 or int(self.token_groups.max()) >= self.num_groups:
                raise ValueError('reverse group id is out of range')

    def members(self, group_id: int | torch.Tensor) -> torch.Tensor:
        group = int(group_id)
        begin = int(self.group_offsets[group])
        end = int(self.group_offsets[group + 1])
        return self.group_members[begin:end]

    def groups_for_token(self, token_id: int | torch.Tensor) -> torch.Tensor:
        token = int(token_id)
        begin = int(self.token_offsets[token])
        end = int(self.token_offsets[token + 1])
        return self.token_groups[begin:end]

    def group_mass(self, probabilities: torch.Tensor, group_id: int | torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 1 or probabilities.numel() != self.vocab_size:
            raise ValueError('probabilities must have shape [vocab_size]')
        members = self.members(group_id)
        weights = self.membership_count[members].to(probabilities.dtype).reciprocal()
        return (probabilities[members] * weights).sum()

    def all_group_masses(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Compute every coarse probability with one sparse scatter."""
        if probabilities.ndim != 1 or probabilities.numel() != self.vocab_size:
            raise ValueError('probabilities must have shape [vocab_size]')
        counts = self.membership_count
        token_ids = torch.repeat_interleave(torch.arange(self.vocab_size, device=self.device), counts)
        contributions = probabilities[token_ids] / counts[token_ids].to(probabilities.dtype)
        result = torch.zeros(self.num_groups, dtype=probabilities.dtype, device=self.device)
        result.scatter_add_(0, self.token_groups, contributions)
        return result

    def sample_group(self, token_id: int | torch.Tensor, *, generator=None) -> torch.Tensor:
        groups = self.groups_for_token(token_id)
        choice = torch.randint(groups.numel(), (1,), device=groups.device, generator=generator)
        return groups[choice].squeeze(0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {'group_offsets': self.group_offsets.detach().cpu().contiguous(), 'group_members': self.group_members.detach().cpu().contiguous(), 'token_offsets': self.token_offsets.detach().cpu().contiguous(), 'token_groups': self.token_groups.detach().cpu().contiguous(), 'membership_count': self.membership_count.detach().cpu().contiguous()}
        save_file(tensors, str(path), metadata={'pcg_metadata': json.dumps(self.metadata)})

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str='cpu') -> 'AcousticGroups':
        path = Path(path)
        tensors = load_file(str(path), device=str(device))
        with safe_open(str(path), framework='pt', device=str(device)) as handle:
            metadata = json.loads((handle.metadata() or {}).get('pcg_metadata', '{}'))
        result = cls(metadata=metadata, **tensors)
        result.validate()
        return result

    def dense(self, device: torch.device | str) -> 'DenseAcousticGroups':
        """Materialize small padded tables for latency-sensitive K-step inference."""
        group_sizes = (self.group_offsets[1:] - self.group_offsets[:-1]).cpu()
        token_sizes = (self.token_offsets[1:] - self.token_offsets[:-1]).cpu()
        max_group = int(group_sizes.max())
        max_token_groups = int(token_sizes.max())
        group_members = torch.zeros((self.num_groups, max_group), dtype=torch.long)
        group_valid = torch.zeros((self.num_groups, max_group), dtype=torch.bool)
        token_groups = torch.zeros((self.vocab_size, max_token_groups), dtype=torch.long)
        for index in range(self.num_groups):
            row = self.members(index).cpu()
            group_members[index, :row.numel()] = row
            group_valid[index, :row.numel()] = True
        for token in range(self.vocab_size):
            row = self.groups_for_token(token).cpu()
            token_groups[token, :row.numel()] = row
        member_count = self.membership_count.cpu()
        group_weight = member_count[group_members].float().reciprocal() * group_valid
        return DenseAcousticGroups(group_members=group_members.to(device), group_weights=group_weight.to(device), group_sizes=group_sizes.to(device), token_groups=token_groups.to(device), token_group_counts=token_sizes.to(device), sparse=self.to(device))

@dataclass(frozen=True)
class DenseAcousticGroups:
    group_members: torch.Tensor
    group_weights: torch.Tensor
    group_sizes: torch.Tensor
    token_groups: torch.Tensor
    token_group_counts: torch.Tensor
    sparse: AcousticGroups

    def sample_groups(self, token_ids: torch.Tensor, *, generator=None) -> torch.Tensor:
        token_ids = token_ids.long()
        counts = self.token_group_counts[token_ids]
        choices = torch.floor(torch.rand(token_ids.shape, device=token_ids.device, generator=generator) * counts.to(torch.float32)).long()
        return self.token_groups[token_ids, choices]

    def group_masses(self, probabilities: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Pair each probability row with one group id."""
        if probabilities.ndim != 2 or group_ids.ndim != 1:
            raise ValueError('expected probabilities [rows,vocab] and group_ids [rows]')
        if probabilities.shape[0] != group_ids.numel():
            raise ValueError('row and group counts disagree')
        members = self.group_members[group_ids]
        weights = self.group_weights[group_ids].to(probabilities.dtype)
        return probabilities.gather(1, members).mul(weights).sum(-1)

def _csr_from_lists(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([row.numel() for row in rows], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    members = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
    return (offsets, members)

@torch.inference_mode()
def build_acoustic_groups(embeddings: torch.Tensor, *, threshold: float, acoustic_vocab_size: int, chunk_size: int=512) -> AcousticGroups:
    """Build one threshold-neighborhood group per token.

    Acoustic tokens compare only with other acoustic tokens.  Any remaining
    start/EOS tokens become singleton groups, preventing coarse acceptance
    from changing termination semantics.
    """
    if embeddings.ndim != 2:
        raise ValueError('embeddings must have shape [vocab_size, hidden_size]')
    vocab_size = int(embeddings.shape[0])
    if not 0 < acoustic_vocab_size <= vocab_size:
        raise ValueError('invalid acoustic_vocab_size')
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError('cosine threshold must be in [-1, 1]')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    normalized = F.normalize(embeddings[:acoustic_vocab_size].float(), dim=-1)
    groups: list[torch.Tensor] = []
    for start in range(0, acoustic_vocab_size, chunk_size):
        stop = min(start + chunk_size, acoustic_vocab_size)
        similarities = normalized[start:stop] @ normalized.T
        for local_index, row in enumerate(similarities):
            members = torch.nonzero(row > float(threshold), as_tuple=False).flatten().cpu()
            center = start + local_index
            if not bool((members == center).any().item()):
                members = torch.cat((members, torch.tensor([center], dtype=torch.long))).unique(sorted=True)
            groups.append(members)
    for token in range(acoustic_vocab_size, vocab_size):
        groups.append(torch.tensor([token], dtype=torch.long))
    group_offsets, group_members = _csr_from_lists(groups)
    reverse: list[list[int]] = [[] for _ in range(vocab_size)]
    for group_id, members in enumerate(groups):
        for token in members.tolist():
            reverse[token].append(group_id)
    reverse_tensors = [torch.tensor(row, dtype=torch.long) for row in reverse]
    token_offsets, token_groups = _csr_from_lists(reverse_tensors)
    membership_count = token_offsets[1:] - token_offsets[:-1]
    sizes = group_offsets[1:] - group_offsets[:-1]
    metadata = {'format_version': 1, 'method': 'cosine_threshold_equal_split', 'threshold': float(threshold), 'vocab_size': vocab_size, 'acoustic_vocab_size': int(acoustic_vocab_size), 'embedding_size': int(embeddings.shape[1]), 'num_groups': len(groups), 'num_memberships': int(group_members.numel()), 'group_size_min': int(sizes.min()), 'group_size_max': int(sizes.max()), 'group_size_mean': float(sizes.float().mean()), 'membership_min': int(membership_count.min()), 'membership_max': int(membership_count.max()), 'membership_mean': float(membership_count.float().mean())}
    result = AcousticGroups(group_offsets=group_offsets, group_members=group_members, token_offsets=token_offsets, token_groups=token_groups, membership_count=membership_count, metadata=metadata)
    result.validate()
    return result

