"""Batch first-chunk PCM conversion without changing each request's crop."""
import torch


def stage_head_pcm(wave,ends,*,pin_memory=True):
    if wave.ndim!=2 or len(ends)!=wave.shape[0] or not ends:
        raise ValueError('Head PCM rows and sample ends differ')
    width=max(ends)
    if min(ends)<=0 or width>wave.shape[1]:
        raise ValueError('Invalid first-chunk sample crop')
    # Same multiplication and cast as the rowwise path. Short rows are cropped
    # on CPU after one copy; no padding enters CFM or Vocoder computation.
    device=(wave[:,:width]*32767).to(torch.int16)
    host=torch.empty(device.shape,device='cpu',dtype=torch.int16,pin_memory=pin_memory)
    return device,host
