"""Matrix compute dispatch for already-prepared weights.

BF16 uses the PyTorch reference (also traceable by torch.compile). FP8 requires
native E4M3 hardware and an offline-selected Triton tile; it never falls back to
an unreported precision or tunes during forward.
"""


def linear(x, module):
    if module.precision == "bf16":
        from .eager.matrix import linear as eager
        return eager(x, module.weight, module.bias)
    if module.precision != "fp8":
        raise ValueError(f"Unsupported matrix precision: {module.precision}")
    from .triton.fp8 import linear as kernel
    m = x.numel() // module.in_features
    extent = next((n for n in module.tiles if m <= n), module._last_extent)
    return kernel(x, module.weight, module.scales, module.bias, module.tiles[extent])


def conv1d(x, module):
    if module.precision == "bf16":
        from .eager.matrix import conv1d as eager
        return eager(x, module)
    if module.precision != "fp8":
        raise ValueError(f"Unsupported convolution precision: {module.precision}")
    from .triton.fp8_conv import conv1d as kernel
    extent = next((n for n in module.tiles if x.shape[0] * x.shape[-1] <= n), 4096)
    return kernel(x, module.weight, module.scales, module.bias,
                  module.in_channels, module.out_channels, module.kernel_size[0],
                  module.stride[0], module.padding[0], module.dilation[0],
                  module.transpose, module.output_padding[0], module.tiles[extent])
