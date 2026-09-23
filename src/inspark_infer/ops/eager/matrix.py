"""PyTorch low-precision reference math and prepared-module adapters."""
import torch
from torch import nn


def linear(x, weight, bias):
    return torch.nn.functional.linear(x.bfloat16(), weight, bias).to(x.dtype)


def conv1d(x, module):
    if module.transpose:
        y = torch.nn.functional.conv_transpose1d(
            x.bfloat16(), module.weight, module.bias, module.stride,
            module.padding, module.output_padding, module.groups, module.dilation)
    else:
        y = torch.nn.functional.conv1d(
            x.bfloat16(), module.weight, module.bias, module.stride,
            module.padding, module.dilation, module.groups)
    return y.to(x.dtype)


class MatrixLinear(nn.Module):
    def __init__(self, module, precision, caps, plans,
                 extents=(8, 64, 256, 1024, 4096)):
        super().__init__()
        from inspark_infer.quantization.precision import configure_linear
        configure_linear(self, module, precision, caps, plans, extents)

    def forward(self, x):
        from inspark_infer.ops.matrix import linear as dispatch
        return dispatch(x, self)


class MatrixConv(nn.Module):
    def __init__(self, module, precision, caps, plans):
        super().__init__()
        from inspark_infer.quantization.precision import configure_conv
        configure_conv(self, module, precision, caps, plans)

    def forward(self, x):
        from inspark_infer.ops.matrix import conv1d as dispatch
        return dispatch(x, self)
