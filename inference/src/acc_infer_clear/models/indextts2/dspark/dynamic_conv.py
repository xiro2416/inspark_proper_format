"""DFlash2 grouped dynamic conv: causal two taps or centered three taps.

The legacy class name is retained; kernel_size=3 is explicitly noncausal.
Tap storage order is current, previous, next. Boundaries never cross blocks.
"""
import torch
import torch.nn.functional as F
from torch import nn

class GroupedDynamicCausalConv(nn.Module):

    def __init__(self, hidden_size, kernel_size=2, group_size=16, mode='full'):
        super().__init__()
        if hidden_size % group_size or kernel_size not in (2, 3):
            raise ValueError('requires divisible channel groups and 2 or 3 taps')
        self.kernel_size = kernel_size
        self.group_size = group_size
        if mode not in ('full', 'static_only', 'dynamic_only'):
            raise ValueError('unknown dynamic convolution mode')
        self.mode = mode
        self.register_parameter('base_kernel', None)
        self.kernel_projection = None
        if mode != 'dynamic_only':
            self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
            with torch.no_grad():
                self.base_kernel[:, 0].fill_(1.0)
        if mode != 'static_only':
            groups = hidden_size // group_size
            self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * groups, bias=mode == 'dynamic_only')
            nn.init.zeros_(self.kernel_projection.weight)
            if mode == 'dynamic_only':
                nn.init.zeros_(self.kernel_projection.bias)
                with torch.no_grad():
                    self.kernel_projection.bias.view(2, kernel_size, groups)[:, 0].fill_(1.0)

    def _convolve(self, hidden, dynamic, base):
        batch, anchors, length, width = hidden.shape
        groups = width // self.group_size
        blocks = hidden.reshape(batch, anchors, length, groups, self.group_size)
        if dynamic is not None:
            dynamic = dynamic.reshape(batch, anchors, length, self.kernel_size, groups, 1)
        output = torch.zeros_like(blocks)
        for offset in range(self.kernel_size):
            if offset == 0:
                values = blocks
            elif offset == 1:
                values = F.pad(blocks[:, :, :-1], (0, 0, 0, 0, 1, 0))
            else:
                values = F.pad(blocks[:, :, 1:], (0, 0, 0, 0, 0, 1))
            if base is not None:
                kernel = base[offset].view(1, 1, 1, groups, self.group_size).to(hidden.dtype)
                output = output + kernel * values
            if dynamic is not None:
                output = torch.addcmul(output, dynamic[..., offset, :, :], values)
        return output.reshape_as(hidden)

    def prepare(self, hidden):
        if self.mode == 'static_only':
            return (self._convolve(hidden, None, self.base_kernel[0]), None)
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, groups)
        base = self.base_kernel[0] if self.base_kernel is not None else None
        return (self._convolve(hidden, dynamic[..., 0, :, :], base), dynamic[..., 1, :, :])

    def finish(self, hidden, dynamic):
        base = self.base_kernel[1] if self.base_kernel is not None else None
        return self._convolve(hidden, dynamic, base)

