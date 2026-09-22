"""Same BF16 convolution via height-one 2D descriptors and fused cast/layout copies."""
import torch
from torch import nn

class LayoutConv(nn.Module):
    def __init__(self,old,layout='channels_last'):
        super().__init__();self.old=old;self.layout=layout;self.allowed=None
        fmt=torch.channels_last if layout=='channels_last' else torch.contiguous_format
        self.register_buffer('weight2d',old.weight.detach().unsqueeze(2).contiguous(memory_format=fmt))
    def forward(self,x):
        if self.allowed is not None and (x.shape[0],x.shape[-1]) not in self.allowed:return self.old(x)
        o=self.old;fmt=torch.channels_last if self.layout=='channels_last' else torch.contiguous_format
        xx=x.unsqueeze(2).to(dtype=torch.bfloat16,memory_format=fmt)
        if o.transpose:
            y=torch.nn.functional.conv_transpose2d(xx,self.weight2d,o.bias,(1,o.stride[0]),(0,o.padding[0]),(0,o.output_padding[0]),o.groups,(1,o.dilation[0]))
        else:y=torch.nn.functional.conv2d(xx,self.weight2d,o.bias,(1,o.stride[0]),(0,o.padding[0]),(1,o.dilation[0]),o.groups)
        return y.to(dtype=x.dtype,memory_format=torch.contiguous_format).squeeze(2)
