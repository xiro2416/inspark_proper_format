"""Existing-op ragged staging with validated shared-storage views."""
import torch


def shared_rows(rows,width):
    if not rows:return None
    first=rows[0]
    if first.shape[0]!=1 or width<1:return None
    pitch=(rows[1].data_ptr()-first.data_ptr())//first.element_size() if len(rows)>1 else first.stride(0)
    if pitch<width*first.stride(1):return None
    storage=first.untyped_storage().data_ptr()
    for i,row in enumerate(rows):
        if (row.shape[0]!=1 or row.dtype!=first.dtype or row.device!=first.device
                or row.shape[2:]!=first.shape[2:] or row.stride()[1:]!=first.stride()[1:]
                or row.untyped_storage().data_ptr()!=storage
                or row.data_ptr()!=first.data_ptr()+i*pitch*first.element_size()):return None
    try:return torch.as_strided(first,(len(rows),width,*first.shape[2:]),(pitch,*first.stride()[1:]))
    except RuntimeError:return None


def pack_prefix_suffix(workspace,prefix,suffix,prefix_lengths,token_lengths):
    """One spare column absorbs invalid scatter writes, never a valid position."""
    actual=prefix.shape[0];extent=workspace.shape[1]-1
    length=prefix_lengths+token_lengths+1
    workspace.zero_()
    ppos=torch.arange(prefix.shape[1],device=workspace.device)
    workspace[:actual,:prefix.shape[1]].copy_(prefix.masked_fill(ppos[None,:,None]>=prefix_lengths[:,None,None],0))
    spos=torch.arange(suffix.shape[1],device=workspace.device)
    valid=spos[None]<token_lengths[:,None]+1
    positions=torch.where(valid,prefix_lengths[:,None]+spos[None],extent)
    workspace[:actual].scatter_(1,positions[:,:,None].expand(-1,-1,suffix.shape[-1]),suffix.masked_fill(~valid[:,:,None],0))
    keep=torch.arange(extent,device=workspace.device)[None]<length[:,None]
    return workspace[:actual,:extent],keep
