"""Shared role policy and offline calibration for FP8 and INT8 SmoothQuant.

This module only uses existing torch operators. Calibration is never performed
in a measured inference path, and a missing observation is an error.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


@dataclass
class RoleBinding:
    path: str
    component: str
    module: nn.Module
    parent: nn.Module
    child_name: str
    precision: str
    channel_axis: int
    weight_input_axis: int
    weight_output_axis: int

    def weight(self):
        value = self.module.weight.detach()
        for hook in self.module._forward_pre_hooks.values():
            if getattr(hook, 'name', None) == 'weight' and hasattr(hook, 'compute_weight'):
                value = hook.compute_weight(self.module).detach()
        # HuggingFace GPT Conv1D stores a transposed linear weight.
        if type(self.module).__name__ == 'Conv1D':
            value = value.t()
        return value.float()


def iter_roles(engine, quant_format='fp8', components=('target', 'draft', 'cfm', 'vocoder')):
    if quant_format not in ('fp8', 'int8', 'int8_smoothquant','nvfp4'):
        raise ValueError(f'Unknown quantization format: {quant_format}')
    quant_format = 'int8' if quant_format == 'int8_smoothquant' else quant_format
    # U-ViT constructs this projection on every block but only calls it in
    # receiving layers. Do not invent calibration values for unreachable ops.
    transformer = engine.student.model.transformer
    inactive_skips = {f'cfm.blocks.{i}.skip_in_linear' for i in range(len(transformer.layers))
                      if i not in getattr(transformer, 'layers_receive_skip', ())}

    def walk(parent, name, prefix, component, precision):
        if prefix in inactive_skips:
            return
        module = getattr(parent, name)
        linear = isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D'
        conv = isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
        if linear or conv:
            if '.attention_norm.' in prefix or '.ffn_norm.' in prefix:
                return
            if conv and (module.groups != 1 or module.padding_mode != 'zeros'):
                raise ValueError(f'Unsupported learned convolution: {prefix}')
            transpose = isinstance(module, nn.ConvTranspose1d)
            yield RoleBinding(prefix, component, module, parent, name, precision,
                              1 if conv else -1, 0 if transpose else 1,
                              1 if transpose else 0)
            return
        for child_name, _ in module.named_children():
            yield from walk(module, child_name, f'{prefix}.{child_name}', component, precision)

    roots = {'target': engine.tts.gpt.gpt.h, 'draft': engine.rt.engine.draft.layers,
             'cfm': engine.student.model.transformer.layers}
    for component in components:
        if component == 'vocoder':
            model = engine.tts.bigvgan
            cutoff = math.ceil(model.num_upsamples / 4)
            for i in range(model.num_upsamples):
                precision = 'bf16' if i < cutoff else quant_format
                yield from walk(model.ups, str(i), f'vocoder.stages.{i}.ups', component, precision)
                for j in range(model.num_kernels):
                    yield from walk(model.resblocks, str(i * model.num_kernels + j),
                                    f'vocoder.stages.{i}.resblock{j}', component, precision)
            for name in ('conv_pre', 'conv_post'):
                yield from walk(model, name, f'vocoder.{name}', component, 'bf16')
            continue
        layers = roots[component]
        cutoff = math.ceil(len(layers) / 4)
        for i in range(len(layers)):
            yield from walk(layers, str(i), f'{component}.blocks.{i}', component,
                            'bf16' if i < cutoff else quant_format)
        if component == 'cfm':
            wavenet = engine.student.model.wavenet
            for name in ('in_layers', 'res_skip_layers'):
                for i in range(len(getattr(wavenet, name))):
                    yield from walk(getattr(wavenet, name), str(i), f'cfm.wavenet.{name}.{i}',
                                    component, quant_format)


class ChannelObserver:
    """GPU-resident running channel maxima; one CPU transfer at finalization."""
    def __init__(self, role):
        self.role = role
        self.amax = None
        self.calls = 0
        self.elements = 0

    def __call__(self, module, args):
        value = args[0].detach().float()
        axis = self.role.channel_axis % value.ndim
        maximum = value.abs().amax(tuple(i for i in range(value.ndim) if i != axis))
        self.amax = maximum if self.amax is None else torch.maximum(self.amax, maximum)
        self.calls += 1
        self.elements += value.numel()


def smoothing_scale(activation_amax, weight, input_axis, alpha=1.0):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('SmoothQuant alpha must be in [0,1]')
    if (weight.ndim < 2 or not isinstance(input_axis, int) or isinstance(input_axis, bool)
            or not -weight.ndim <= input_axis < weight.ndim):
        raise ValueError('Invalid weight input channel axis')
    input_axis %= weight.ndim
    a = activation_amax.float()
    w = weight.abs().amax(tuple(i for i in range(weight.ndim) if i != input_axis))
    if (a.ndim != 1 or a.shape != w.shape
            or not bool(torch.isfinite(a).all() & torch.isfinite(w).all() & (a >= 0).all())):
        raise ValueError('Invalid activation/weight channel statistics')
    # Zero channels must not introduce infinities or erase otherwise valid weights.
    scale = (a.clamp_min(1e-5).pow(alpha) / w.clamp_min(1e-5).pow(1-alpha)).clamp_min(1e-5)
    return torch.where((a == 0) | (w == 0), torch.ones_like(scale), scale)


def modelopt_linear_scales(weight, activation_amax, alpha=1.0):
    """Run NVIDIA SmoothQuant on the exact sufficient max-calibration statistics.

    Replaying channel maxima into the official max observers is equivalent to
    replaying the complete observed activation set for this calibration method.
    It avoids a second expensive TTS generation and retains ModelOpt's clamps.
    """
    import copy
    from contextlib import redirect_stdout
    import io
    import modelopt.torch.quantization as mtq
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device)
    linear.weight.data.copy_(weight)
    config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
    config['algorithm'] = {'method': 'smoothquant', 'alpha': alpha}
    def replay_statistics(module):
        module.input_quantizer(activation_amax[None].to(weight.device))
        module.weight_quantizer(module.weight)
    with torch.no_grad(), redirect_stdout(io.StringIO()):
        linear = mtq.quantize(linear, config, replay_statistics)
    return dict(smooth_scale=linear.input_quantizer.pre_quant_scale.detach().float().reciprocal().cpu().tolist(),
                input_scale=max(float(linear.input_quantizer.amax) / 127., 1e-12),
                weight_scale=(linear.weight_quantizer.amax.detach().float().flatten().cpu() / 127.).clamp_min(1e-12).tolist(),
                smoothing_source='modelopt_native', modelopt_version=__import__('modelopt').__version__,
                statistics_replay='exact_input_channel_amax_and_original_weight')


def role_specs(roles, observers, quant_format, alpha=1.0, native_modelopt=True):
    """Build portable explicit-Q/DQ specs, without claiming a native ModelOpt call."""
    result = {}
    for role in roles:
        precision = role.precision if role.precision == 'bf16' else quant_format
        if precision == 'int8_smoothquant':
            precision = 'int8'
        spec = dict(precision=precision, alpha=alpha, smoothing_source='none',
                    input_channel_axis=role.channel_axis,
                    weight_input_axis=role.weight_input_axis,
                    weight_axis=role.weight_output_axis)
        if precision == 'bf16':
            result[role.path] = spec
            continue
        observer = observers[role.path]
        if observer.amax is None or not observer.calls:
            raise ValueError(f'No real calibration observations for {role.path}')
        a = observer.amax.detach().cpu()
        weight = role.weight().cpu()
        if not bool(torch.isfinite(a).all() & torch.isfinite(weight).all()):
            raise ValueError(f'Nonfinite calibration: {role.path}')
        if precision == 'int8' and weight.ndim == 2 and native_modelopt:
            spec.update(modelopt_linear_scales(weight, a, alpha),
                        calibration_calls=observer.calls, calibration_elements=observer.elements)
            result[role.path] = spec
            continue
        if precision == 'int8':
            smooth = smoothing_scale(a, weight, role.weight_input_axis, alpha)
            shape = [1] * weight.ndim
            shape[role.weight_input_axis] = smooth.numel()
            weight = weight * smooth.reshape(shape)
            a = a / smooth
            spec.update(smooth_scale=smooth.tolist(), smoothing_source=(
                'graph_sq' if weight.ndim == 3 else 'smoothquant_formula'))
        maximum = 127.0 if precision == 'int8' else 448.0
        w_amax = weight.abs().amax(tuple(i for i in range(weight.ndim) if i != role.weight_output_axis))
        spec.update(input_scale=max(float(a.max()) / maximum, 1e-12),
                    weight_scale=(w_amax / maximum).clamp_min(1e-12).tolist(),
                    calibration_calls=observer.calls, calibration_elements=observer.elements)
        result[role.path] = spec
    return result


def save_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(path, quant_format=None):
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != 1 or not payload.get('role_specs'):
        raise ValueError('Invalid unified quantization artifact')
    if quant_format and payload['scheme'] != quant_format:
        raise ValueError('Quantization artifact precision mismatch')
    for name, spec in payload['role_specs'].items():
        if spec.get('precision') not in ('bf16', 'fp8', 'int8','nvfp4'):
            raise ValueError(f'Invalid role precision: {name}')
        if spec['precision']=='nvfp4':
            if not math.isfinite(spec['activation_amax']) or spec['activation_amax']<=0 or spec['block_size']!=16:
                raise ValueError(f'Invalid NVFP4 role: {name}')
        if spec['precision'] in ('fp8', 'int8'):
            if not math.isfinite(spec['input_scale']) or spec['input_scale'] <= 0:
                raise ValueError(f'Invalid activation scale: {name}')
            scales = spec['weight_scale']
            if not isinstance(scales, list):
                scales = [scales]
            if not scales or any(not math.isfinite(v) or v <= 0 for v in scales):
                raise ValueError(f'Invalid weight scales: {name}')
            input_axis, output_axis = spec.get('weight_input_axis'), spec.get('weight_axis')
            if input_axis is not None or output_axis is not None:
                if (type(input_axis) is not int or type(output_axis) is not int
                        or {input_axis, output_axis} != {0, 1}):
                    raise ValueError(f'Invalid weight channel axes: {name}')
            if spec['precision'] == 'int8':
                smooth = spec.get('smooth_scale')
                if (not isinstance(smooth, list) or not smooth
                        or any(not math.isfinite(v) or v <= 0 for v in smooth)):
                    raise ValueError(f'Invalid smoothing scales: {name}')
    return payload
