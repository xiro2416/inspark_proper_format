"""Export-only standard Q/DQ translation; eager math uses existing torch ops."""
import torch
from inspark_infer.build.unified_acoustic_export import _QDQ


@torch.library.custom_op('inspark_export::qdq',mutates_args=())
def qdq(value:torch.Tensor,scale:torch.Tensor,zero:torch.Tensor,axis:int)->torch.Tensor:
    return _QDQ.forward(None,value,scale,zero,axis)


@qdq.register_fake
def _fake(value,scale,zero,axis):
    return torch.empty_like(value)


def translation(value,scale,zero,axis:int):
    from onnxscript import opset20 as op
    q=op.QuantizeLinear(value,scale,zero,axis=axis)
    return op.DequantizeLinear(q,scale,zero,axis=axis)


def translations():
    return {torch.ops.inspark_export.qdq.default:translation}
