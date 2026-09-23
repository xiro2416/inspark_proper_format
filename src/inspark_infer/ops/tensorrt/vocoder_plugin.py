"""TensorRT 11.3 Quick Plugin for BigVGAN's alias-free activation.

The plugin keeps the existing, numerically validated FP32 two-kernel
implementation inside a TensorRT engine.  Its second output is an internal
2x-rate workspace tensor; consumers only use the first output.
"""
from typing import Tuple


def _torch_stream(torch, stream: int):
    """Map TRT's legacy-default-stream sentinel without creating a new stream."""
    return torch.cuda.default_stream() if int(stream) == 0 else torch.cuda.ExternalStream(stream)


def register() -> None:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    # Registration is process-global and importing this module more than once
    # must be harmless (builder and runtime both call register()).
    try:
        getattr(trtp.op.inspark, "alias_free")
        return
    except AttributeError:
        pass

    @trtp.register("inspark::alias_free")
    def alias_free_desc(
        x: trtp.TensorDesc,
        up_filter: trtp.TensorDesc,
        down_filter: trtp.TensorDesc,
        alpha: trtp.TensorDesc,
        inverse_beta: trtp.TensorDesc,
    ) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
        workspace = trtp.from_shape_expr(
            (x.shape_expr[0], x.shape_expr[1], x.shape_expr[2] * 2), x.dtype
        )
        return x.like(), workspace

    @trtp.impl("inspark::alias_free")
    def alias_free_impl(
        x, up_filter, down_filter, alpha, inverse_beta, outputs, stream: int
    ) -> None:
        import torch
        import triton
        from inspark_infer.ops.triton.alias_free import _down, _up_snake

        xx = torch.as_tensor(x, device="cuda", dtype=torch.float32)
        up = torch.as_tensor(up_filter, device="cuda", dtype=torch.float32)
        down = torch.as_tensor(down_filter, device="cuda", dtype=torch.float32)
        aa = torch.as_tensor(alpha, device="cuda", dtype=torch.float32)
        ib = torch.as_tensor(inverse_beta, device="cuda", dtype=torch.float32)
        yy = torch.as_tensor(outputs[0], device="cuda", dtype=torch.float32)
        workspace = torch.as_tensor(outputs[1], device="cuda", dtype=torch.float32)
        if xx.dtype is not torch.float32:
            raise RuntimeError("alias_free plugin requires an FP32 activation boundary")
        b, c, t = xx.shape
        external = _torch_stream(torch, stream)
        with torch.cuda.stream(external):
            _up_snake[(b * c, triton.cdiv(2 * t, 256))](
                xx, up, aa, ib, workspace, t, c, *xx.stride(), 256,
                enable_fp_fusion=False,
            )
            _down[(b * c, triton.cdiv(t, 256))](
                workspace, down, yy, t, 256, enable_fp_fusion=False
            )

    @trtp.register("inspark::deconv1d")
    def deconv1d_desc(
        x: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        bias: trtp.TensorDesc,
        stride: int,
        padding: int,
        output_padding: int,
        dilation: int,
        groups: int,
    ) -> trtp.TensorDesc:
        length = ((x.shape_expr[2] - 1) * stride - 2 * padding
                  + (weight.shape_expr[2] - 1) * dilation + output_padding + 1)
        channels = weight.shape_expr[1] * groups
        return trtp.from_shape_expr((x.shape_expr[0], channels, length), x.dtype)

    @trtp.impl("inspark::deconv1d")
    def deconv1d_impl(
        x, weight, bias, stride: int, padding: int, output_padding: int,
        dilation: int, groups: int, outputs, stream: int,
    ) -> None:
        import torch

        # Quick Plugin's CUDA array interface cannot expose BF16 buffers, so
        # reproduce MatrixConv's BF16 boundary inside the callback.
        xx = torch.as_tensor(x, device="cuda")
        ww = torch.as_tensor(weight, device="cuda")
        bb = torch.as_tensor(bias, device="cuda")
        yy = torch.as_tensor(outputs[0], device="cuda")
        external = _torch_stream(torch, stream)
        with torch.cuda.stream(external):
            value = torch.nn.functional.conv_transpose1d(
                xx.bfloat16(), ww.bfloat16(), bb.bfloat16(), stride, padding,
                output_padding, groups, dilation,
            )
            yy.copy_(value.float())

    @trtp.register("inspark::conv1d")
    def conv1d_desc(
        x: trtp.TensorDesc,
        weight: trtp.TensorDesc,
        bias: trtp.TensorDesc,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ) -> trtp.TensorDesc:
        length = ((x.shape_expr[2] + 2 * padding
                   - (weight.shape_expr[2] - 1) * dilation - 1) // stride + 1)
        return trtp.from_shape_expr(
            (x.shape_expr[0], weight.shape_expr[0], length), x.dtype
        )

    @trtp.impl("inspark::conv1d")
    def conv1d_impl(
        x, weight, bias, stride: int, padding: int, dilation: int,
        groups: int, outputs, stream: int,
    ) -> None:
        import torch

        xx = torch.as_tensor(x, device="cuda")
        ww = torch.as_tensor(weight, device="cuda")
        bb = torch.as_tensor(bias, device="cuda")
        yy = torch.as_tensor(outputs[0], device="cuda")
        external = _torch_stream(torch, stream)
        with torch.cuda.stream(external):
            value = torch.nn.functional.conv1d(
                xx.bfloat16(), ww.bfloat16(), bb.bfloat16(), stride, padding,
                dilation, groups,
            )
            yy.copy_(value.float())


def add_to_network(network, x, up_filter, down_filter, alpha, inverse_beta):
    """Add the registered plugin through the TRT 11.3 Python plugin API."""
    import tensorrt.plugin as trtp

    register()
    return trtp.op.inspark.alias_free(
        x, up_filter, down_filter, alpha, inverse_beta
    )
