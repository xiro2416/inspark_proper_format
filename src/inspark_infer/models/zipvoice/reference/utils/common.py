"""Original model helpers; inference dependency closure only."""
from typing import List, Tuple
import torch

def prepare_avg_tokens_durations(features_lens, tokens_lens):
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations

def pad_labels(y: List[List[int]], pad_id: int, device: torch.device):
    """
    Pad the transcripts to the same length with zeros.

    Args:
      y: the transcripts, which is a list of a list

    Returns:
      Return a Tensor of padded transcripts.
    """
    y = [token_ids + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)

def get_tokens_index(durations: List[List[int]], num_frames: int) -> torch.Tensor:
    """
    Gets position in the transcript for each frame, i.e. the position
    in the symbol-sequence to look up.

    Args:
      durations:
        Duration of each token in transcripts.
      num_frames:
        The maximum frame length of the current batch.

    Returns:
      Return a Tensor of shape (batch_size, num_frames)
    """
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans

def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """
    Apply Time masking.
    Args:
        features_lens:
            input tensor of shape ``(B)``
        mask_size:
            the width size for masking.
        max_len:
            the maximum length of the mask.
    Returns:
        Return a 2-D bool tensor (B, T), where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.
    """
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)
