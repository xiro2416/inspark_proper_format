"""Explicit static E4M3 W8A8; floating bias, accumulation and outputs.

The ONNX contract is standard QuantizeLinear/DequantizeLinear, not INT8
reinterpretation or a float model mislabeled as FP8. No SmoothQuant scales
are reused. Quantization is nearest-even with E4M3 finite saturation.
"""
import torch
from torch import nn


@torch.library.custom_op('zipvoice_fp8_export::qdq', mutates_args=())
def export_qdq(value: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return (value / scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale


@export_qdq.register_fake
def _qdq_fake(value, scale, zero):
    return torch.empty_like(value)


@torch.library.custom_op('zipvoice_fp8_export::weight_dq', mutates_args=())
def export_weight_dq(value: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return value.float()*scale


@export_weight_dq.register_fake
def _weight_fake(value, scale, zero):
    return torch.empty_like(value, dtype=torch.float32)


def translations():
    from onnxscript import opset21 as op
    def qdq(value, scale, zero):
        return op.DequantizeLinear(op.QuantizeLinear(value,scale,zero),scale,zero)
    def weight_dq(value, scale, zero):
        return op.DequantizeLinear(value,scale,zero)
    return {torch.ops.zipvoice_fp8_export.qdq.default: qdq,
            torch.ops.zipvoice_fp8_export.weight_dq.default: weight_dq}


class QuantizeDequantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale

    @staticmethod
    def symbolic(g, x, scale):
        zero = g.op('Cast', g.op('Constant', value_t=torch.tensor(0.)), to_i=17)
        quantized = g.op('QuantizeLinear', x, scale, zero)
        return g.op('DequantizeLinear', quantized, scale, zero)


class DequantizeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        return weight.float() * scale

    @staticmethod
    def symbolic(g, weight, scale):
        zero = g.op('Cast', g.op('Constant', value_t=torch.tensor(0.)), to_i=17)
        return g.op('DequantizeLinear', weight, scale, zero)


class FP8Linear(nn.Module):
    def __init__(self, original, input_scale, weight_scale):
        super().__init__()
        device = original.weight.device
        self.register_buffer('input_scale', torch.tensor(input_scale, dtype=torch.float32, device=device))
        self.register_buffer('weight_scale', torch.tensor(weight_scale, dtype=torch.float32, device=device))
        self.register_buffer('zero_point', torch.zeros((),dtype=torch.float8_e4m3fn,device=device))
        weight = (original.weight.detach().float() / self.weight_scale).clamp(-448,448)
        self.register_buffer('weight_fp8', weight.to(torch.float8_e4m3fn))
        self.register_buffer('bias', original.bias.detach().clone() if original.bias is not None else None)

    def forward(self, x):
        if not torch.jit.is_tracing():
            x = export_qdq(x, self.input_scale, self.zero_point)
            weight = export_weight_dq(self.weight_fp8, self.weight_scale, self.zero_point)
            return torch.nn.functional.linear(x,weight,self.bias)
        x = QuantizeDequantize.apply(x, self.input_scale)
        weight = DequantizeWeight.apply(self.weight_fp8, self.weight_scale)
        return torch.nn.functional.linear(x, weight, self.bias)


def original_activation(module, value):
    from inspark_infer.models.zipvoice.reference.models.modules.scaling import SwooshLForward,SwooshRForward
    if module.activation=='SwooshL':return SwooshLForward(value)
    if module.activation=='SwooshR':return SwooshRForward(value)
    raise ValueError('Unknown original activation: '+module.activation)


class FP8ActivationLinear(FP8Linear):
    """Original floating activation followed by standard W8A8 projection."""
    def __init__(self,original,input_scale,weight_scale):
        super().__init__(original,input_scale,weight_scale)
        self.activation=original.activation

    def forward(self,x):
        return super().forward(original_activation(self,x))


def coverage(model):
    from inspark_infer.models.zipvoice.reference.models.modules.zipformer import Zipformer2EncoderLayer
    from inspark_infer.models.zipvoice.reference.models.modules.scaling import ActivationDropoutAndLinear
    layers = [name for name, module in model.fm_decoder.named_modules()
              if isinstance(module, Zipformer2EncoderLayer)]
    if len(layers) != 16:
        raise ValueError(f'Expected 16 ordered FM layers, found {len(layers)}')
    protected = ['fm_decoder.' + name for name in layers[:4]]
    selected = ['fm_decoder.' + name for name in layers[4:]]
    modules, floating_conv = [], []
    for name, module in model.named_modules():
        if not any(name.startswith(prefix + '.') for prefix in selected):
            continue
        if isinstance(module, (nn.Linear,ActivationDropoutAndLinear)):
            if module.weight.shape[1] % 16 or module.weight.shape[0] % 16:
                raise ValueError('Unaligned FP8 projection requires explicit coverage decision: '+name)
            modules.append(name)
        elif isinstance(module, nn.Conv1d):
            # These are depthwise groups with one input channel per group:
            # no compatible TensorRT FP8 convolution tactic is assumed.
            floating_conv.append(dict(name=name, groups=module.groups,
                                      reason='per-group C=1; this recipe retains floating convolution and claims no FP8 Tensor Core coverage'))
    if not modules:
        raise ValueError('No eligible last12 FP8 projections')
    return dict(protected_layers=protected, quantized_layers=selected,
                quantized_modules=modules, floating_convolutions=floating_conv)


def apply_recipe(model, recipe):
    from inspark_infer.models.zipvoice.reference.models.modules.scaling import ActivationDropoutAndLinear
    expected = coverage(model)
    if expected['quantized_modules'] != list(recipe['modules']):
        raise ValueError('Frozen FP8 module mapping differs from model')
    for name, scales in recipe['modules'].items():
        parent, attribute = name.rsplit('.', 1)
        original=model.get_submodule(name)
        wrapper=FP8ActivationLinear if isinstance(original,ActivationDropoutAndLinear) else FP8Linear
        setattr(model.get_submodule(parent), attribute, wrapper(original,
                scales['input_scale'], scales['weight_scale']))
    return model


def load_fp8_model(directory, device='cpu'):
    """Strictly restore materialized FP8 buffers without recalibration."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from inspark_infer.models.zipvoice.reference.models.zipvoice_distill import ZipVoiceDistill
    from inspark_infer.models.zipvoice.tokenizer import EmiliaTokenizer
    directory=Path(directory)
    config=json.loads((directory/'config/model.json').read_text())['model']
    tokenizer=EmiliaTokenizer(directory/'config/tokens.txt')
    model=ZipVoiceDistill(**config,vocab_size=tokenizer.vocab_size,pad_id=tokenizer.pad_id)
    recipe=json.loads((directory/'fp8/quantization.json').read_text())
    model=apply_recipe(model,recipe)
    # Prototype buffers are overwritten completely by strict state restoration.
    model.load_state_dict(load_file(str(directory/'fp8/model.safetensors')),strict=True)
    return model.to(device).eval(),tokenizer
