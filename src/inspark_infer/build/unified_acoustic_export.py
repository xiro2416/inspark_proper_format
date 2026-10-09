"""Static acoustic ONNX export using standard operators and explicit precision.

This module contains PyTorch reference/export graphs, not serving kernels. All
quantization scales are supplied by offline calibration. No custom ONNX domain,
TensorRT Python plugin, or project GPU math implementation is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


INTERVALS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
WEIGHTED_TYPES = (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)


class _QDQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale, zero, axis):
        shape = [1] * value.ndim
        if scale.ndim:
            shape[axis] = scale.numel()
        factor = scale.reshape(shape)
        scaled = value.float() / factor
        if zero.dtype == torch.int8:
            rounded = scaled.round().clamp(-128, 127)
        elif zero.dtype == torch.float8_e4m3fn:
            rounded = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        else:
            raise ValueError("Only INT8 and E4M3FN Q/DQ are supported")
        return (rounded * factor).to(value.dtype)

    @staticmethod
    def symbolic(graph, value, scale, zero, axis):
        quantized = graph.op("QuantizeLinear", value, scale, zero, axis_i=axis)
        return graph.op("DequantizeLinear", quantized, scale, zero, axis_i=axis)


def fold_weight_norm_(model: nn.Module) -> list[str]:
    """Materialize loaded immutable g/v before calibration/export.

    Removing a legacy hook recomputes weight from g/v; reading module.weight
    alone can otherwise observe the stale value from before checkpoint load.
    """
    changed = []
    for path, module in model.named_modules():
        hooks = list(module._forward_pre_hooks.values())
        if any(getattr(hook, "name", None) == "weight" and
               hasattr(hook, "compute_weight") for hook in hooks):
            torch.nn.utils.remove_weight_norm(module, name="weight")
            changed.append(path)
        elif hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            chain = module.parametrizations.weight
            if not all(type(item).__name__ == "_WeightNorm" for item in chain):
                raise ValueError(f"Unsupported weight parametrization at {path}")
            torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            changed.append(path)
    return changed


def _positive(value, *, name, device, size=None):
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
    if tensor.ndim > 1 or (size is not None and tensor.numel() != size):
        raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
        raise ValueError(f"{name} must contain finite positive values")
    return tensor


class ExportWeightOp(nn.Module):
    """Linear/Conv reference with immutable SQ compensation and standard Q/DQ."""

    def __init__(self, original: nn.Module, spec: Mapping):
        super().__init__()
        if not isinstance(original, WEIGHTED_TYPES):
            raise TypeError(f"Unsupported weighted operator {type(original).__name__}")
        self.kind = ("linear" if isinstance(original, nn.Linear) else
                     "conv_transpose1d" if isinstance(original, nn.ConvTranspose1d) else "conv1d")
        self.deconv_as_conv = spec.get("conv_transpose_rewrite") == "zero_insert_conv"
        if self.deconv_as_conv and self.kind != "conv_transpose1d":
            raise ValueError("Zero-insertion rewrite only applies to ConvTranspose1d")
        self.precision = spec["precision"]
        if self.precision not in ("fp32", "bf16", "fp8", "int8"):
            raise ValueError(f"Unknown precision {self.precision}")
        self.quantized = self.precision in ("fp8", "int8")
        self.weight_quantization_folded = False
        for name in ("in_features", "out_features", "in_channels", "out_channels", "kernel_size"):
            if hasattr(original, name):
                setattr(self, name, getattr(original, name))
        weight = original.weight.detach().float().clone()
        bias = original.bias.detach().float().clone() if original.bias is not None else None
        if self.kind != "linear":
            for name in ("stride", "padding", "dilation", "groups", "padding_mode"):
                setattr(self, name, getattr(original, name))
            self.output_padding = getattr(original, "output_padding", (0,))
            if self.groups != 1 or self.padding_mode != "zeros":
                raise ValueError("Learned acoustic Conv export requires groups=1 and zero padding")
        channels = original.in_features if self.kind == "linear" else original.in_channels
        smooth = spec.get("smooth_scale")
        if smooth is not None:
            if self.precision != "int8":
                raise ValueError("SmoothQuant scale belongs only to an INT8 recipe")
            smooth = _positive(smooth, name="smooth_scale", device=weight.device, size=channels)
            weight = weight * smooth.reshape(
                (1, channels) if self.kind == "linear" else
                (channels, 1, 1) if self.kind == "conv_transpose1d" else (1, channels, 1))
        self.register_buffer("smooth_scale", smooth)
        self.weight_axis = int(spec.get("weight_axis", 0))
        if not 0 <= self.weight_axis < weight.ndim:
            raise ValueError("weight_axis is outside the weight rank")
        if self.deconv_as_conv:
            weight = weight.transpose(0, 1).flip(-1).contiguous()
            self.weight_axis = {0: 1, 1: 0, 2: 2}[self.weight_axis]
        if self.quantized:
            if "input_scale" not in spec:
                raise ValueError("Quantized export requires a calibrated input_scale")
            act_scale = _positive(spec["input_scale"], name="input_scale", device=weight.device, size=1).reshape(())
            if "weight_scale" in spec:
                weight_scale = _positive(spec["weight_scale"], name="weight_scale", device=weight.device)
                if weight_scale.ndim and weight_scale.numel() != weight.shape[self.weight_axis]:
                    raise ValueError("weight_scale does not match weight_axis")
            else:
                axes = tuple(i for i in range(weight.ndim) if i != self.weight_axis)
                weight_scale = weight.abs().amax(dim=axes).clamp_min(1e-12) / (127.0 if self.precision == "int8" else 448.0)
            zero_dtype = torch.int8 if self.precision == "int8" else torch.float8_e4m3fn
            self.register_buffer("input_scale", act_scale)
            self.register_buffer("weight_scale", weight_scale)
            self.register_buffer("input_zero", torch.zeros((), dtype=zero_dtype, device=weight.device))
            self.register_buffer("weight_zero", torch.zeros(weight_scale.shape, dtype=zero_dtype, device=weight.device))
        elif smooth is not None:
            raise ValueError("Floating point operators cannot have SmoothQuant scales")
        if self.precision == "bf16":
            weight = weight.bfloat16()
            bias = bias.bfloat16() if bias is not None else None
        self.register_buffer("weight", weight.contiguous())
        self.register_buffer("bias", bias)
        self.spec = deepcopy(dict(spec))

    @torch.no_grad()
    def prepare_reference_weights(self):
        """Fold immutable weight Q/DQ once for the Torch reference only.

        The ONNX export path rejects this prepared form so it cannot silently
        lose the low-precision weighted-operator pattern in TensorRT.
        """
        if self.quantized and not self.weight_quantization_folded:
            self.weight.copy_(_QDQ.apply(self.weight, self.weight_scale, self.weight_zero, self.weight_axis))
            self.weight_quantization_folded = True
        return self

    def forward(self, x):
        # All acoustic component interfaces stay FP32. Protected operators cast
        # internally, matching the existing MatrixLinear/MatrixConv policy.
        value = x.float()
        if self.smooth_scale is not None:
            shape = ([1] * (value.ndim - 1) + [-1] if self.kind == "linear" else [1, -1, 1])
            value = value / self.smooth_scale.reshape(shape)
        weight = self.weight
        if self.precision == "bf16":
            value = value.bfloat16()
        if self.deconv_as_conv:
            stride = self.stride[0]
            if stride > 1:
                value = F.pad(value.unsqueeze(-1), (0, stride - 1)).reshape(value.shape[0], value.shape[1], -1)
                value = value[..., :-(stride - 1)]
            pad = self.dilation[0] * (self.kernel_size[0] - 1) - self.padding[0]
            value = F.pad(value, (pad, pad + self.output_padding[0]))
        matrix_conv = getattr(self, 'conv1d_as_gemm', False) and self.kind != 'linear'
        if matrix_conv:
            if self.groups != 1 or not self.quantized or self.weight_axis != 0:
                raise ValueError('Matrix convolution requires group1 and unchanged output-channel Q/DQ')
            if self.kind != 'conv1d' and not self.deconv_as_conv:
                raise ValueError('Matrix convolution requires the existing zero-insertion deconvolution')
            stride = 1 if self.deconv_as_conv else self.stride[0]
            padding = 0 if self.deconv_as_conv else self.padding[0]
            if padding:value=F.pad(value,(padding,padding))
            width=(value.shape[-1]-self.dilation[0]*(self.kernel_size[0]-1)-1)//stride+1
            columns=torch.stack([value[..., k*self.dilation[0]:k*self.dilation[0]+width*stride:stride]
                                 for k in range(self.kernel_size[0])],dim=-1)
            value=columns.permute(0,2,1,3).reshape(value.shape[0],width,-1)
            weight=weight.reshape(weight.shape[0],-1)
        lift_conv = getattr(self, 'conv1d_as_2d', False) and self.kind != 'linear'
        if lift_conv:
            if self.kind != 'conv1d' and not self.deconv_as_conv:
                raise ValueError('Conv2d lifting requires Conv1d or an existing zero-insertion rewrite')
            value, weight = value.unsqueeze(2), weight.unsqueeze(2)
        if self.quantized:
            if getattr(self,'modern_export',False):
                from inspark_infer.build.modern_qdq_export import qdq
                quantize=qdq
            else:quantize=_QDQ.apply
            value = quantize(value, self.input_scale, self.input_zero, 0)
            if not self.weight_quantization_folded:
                axis = self.weight_axis + int(lift_conv and self.weight_axis >= 2)
                weight = quantize(weight, self.weight_scale, self.weight_zero, axis)
        if matrix_conv:
            out = F.linear(value, weight, self.bias).transpose(1,2)
        elif self.kind == "linear":
            out = F.linear(value, weight, self.bias)
        elif lift_conv:
            stride = (1, 1 if self.deconv_as_conv else self.stride[0])
            padding = (0, 0 if self.deconv_as_conv else self.padding[0])
            out = F.conv2d(value, weight, self.bias, stride, padding, (1,self.dilation[0]), self.groups).squeeze(2)
        elif self.kind == "conv1d":
            out = F.conv1d(value, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.deconv_as_conv:
            out = F.conv1d(value, weight, self.bias, 1, 0, self.dilation, self.groups)
        else:
            out = F.conv_transpose1d(value, weight, self.bias, self.stride, self.padding,
                                     self.output_padding, self.groups, self.dilation)
        return out.float()


class StaticAliasFree(nn.Module):
    """BigVGAN's original FIR/Snake graph with constant expanded filters.

    PyTorch's exporter loses filter shape after a dynamic channel ``expand``.
    Channels are model constants, so materialize only these fixed depthwise
    filter weights offline. Every runtime operation remains standard ONNX.
    """
    def __init__(self, original, *, fir_polyphase: bool = False):
        super().__init__()
        activation, up, down = original.act, original.upsample, original.downsample.lowpass
        alpha = activation.alpha.detach().float()
        beta = getattr(activation, "beta", activation.alpha).detach().float()
        if activation.alpha_logscale:
            alpha, beta = alpha.exp(), beta.exp()
        self.channels = alpha.numel()
        self.ratio, self.up_stride, self.up_pad = up.ratio, up.stride, up.pad
        self.crop_left, self.crop_right = up.pad_left, up.pad_right
        self.down_stride, self.down_padding = down.stride, down.padding
        self.down_pads, self.down_pad_mode = (down.pad_left, down.pad_right), down.padding_mode
        self.register_buffer("up_filter", up.filter.detach().float().expand(self.channels, -1, -1).contiguous())
        self.fir_polyphase = bool(fir_polyphase)
        if self.fir_polyphase:
            if (self.ratio != 2 or self.up_stride != 2 or
                    tuple(self.up_filter.shape) != (self.channels, 1, 12)):
                raise ValueError("FIR polyphase export requires ratio=stride=2 and a 12-tap depthwise filter")
            # ConvTranspose correlation at position 2*t+p equals a full
            # convolution with taps w[p::2]. PyTorch Conv is correlation, so
            # reverse those six taps. Each group emits its own even/odd pair.
            phases = torch.stack((self.up_filter[..., ::2].flip(-1),
                                  self.up_filter[..., 1::2].flip(-1)), dim=1)
            phase_filter = phases.reshape(2 * self.channels, 1, 6).contiguous()
        else:
            phase_filter = None
        self.register_buffer("up_phase_filter", phase_filter)
        self.register_buffer("down_filter", down.filter.detach().float().expand(self.channels, -1, -1).contiguous())
        self.register_buffer("alpha", alpha.reshape(1, -1, 1).contiguous())
        self.register_buffer("inverse_beta", (1.0 / (beta + activation.no_div_by_zero)).reshape(1, -1, 1).contiguous())

    def _upsample_full(self, value):
        """Full FIR output before the original crop; all math stays FP32."""
        if self.fir_polyphase:
            phases = F.conv1d(value, self.up_phase_filter, padding=5, groups=self.channels)
            # [channel0 even, channel0 odd, channel1 even, ...] -> time order.
            value = phases.reshape(phases.shape[0], self.channels, 2, phases.shape[-1])
            value = value.permute(0, 1, 3, 2).reshape(phases.shape[0], self.channels, -1)
        else:
            value = F.conv_transpose1d(value, self.up_filter, stride=self.up_stride, groups=self.channels)
        return self.ratio * value

    def forward(self, x):
        value = F.pad(x.float(), (self.up_pad, self.up_pad), mode="replicate")
        value = self._upsample_full(value)
        value = value[..., self.crop_left:-self.crop_right if self.crop_right else None]
        value = value + self.inverse_beta * torch.sin(value * self.alpha).square()
        if self.down_padding:
            value = F.pad(value, self.down_pads, mode=self.down_pad_mode)
        return F.conv1d(value, self.down_filter, stride=self.down_stride, groups=self.channels)


class CFMBF16SDPAAttention(nn.Module):
    """CFM self attention with an explicit post-RoPE BF16 SDPA boundary.

    Retains the existing projection modules and their quantization recipe. Only
    SDPA inputs change dtype; output is restored to FP32 before ``wo``. Every
    operation is a standard Torch/ONNX graph operation, with no custom kernel.
    """
    def __init__(self, original):
        super().__init__()
        if (not hasattr(original, "wqkv") or original.kv_cache is not None or
                getattr(original, "_acc_qkv_rope", None) is not None or
                getattr(original, "_acc_full_mask_elision", False)):
            raise ValueError("CFM BF16 SDPA requires original self attention without KV/fused overrides")
        if original.n_head != original.n_local_heads:
            raise ValueError("CFM BF16 SDPA expects the checkpoint's equal query/KV head counts")
        self.wqkv, self.wo = original.wqkv, original.wo
        self.n_head, self.n_local_heads = original.n_head, original.n_local_heads
        self.head_dim, self.dim = original.head_dim, original.dim
        self.kv_cache = None

    def forward(self, x, freqs_cis, mask, input_pos=None, context=None, context_freqs_cis=None):
        from inspark_infer.models.indextts2.upstream.s2mel.modules.gpt_fast.model import apply_rotary_emb
        if context is not None or self.kv_cache is not None:
            raise ValueError("CFM BF16 SDPA graph does not support cross attention or KV updates")
        batch, length, _ = x.shape
        width = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([width, width, width], dim=-1)
        q = q.view(batch, length, self.n_head, self.head_dim)
        k = k.view(batch, length, self.n_local_heads, self.head_dim)
        v = v.view(batch, length, self.n_local_heads, self.head_dim)
        # Preserve FP32 RoPE and its existing frequencies, including scale.
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        q, k, v = (value.transpose(1, 2).to(torch.bfloat16) for value in (q, k, v))
        attention_mask = mask
        if torch.onnx.is_in_onnx_export() and mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("CFM BF16 SDPA ONNX export expects the checkpoint's Boolean mask")
            # Legacy ONNX SDPA creates FP32 0/-inf for a Boolean mask and then
            # promotes Softmax/PV back to FP32. Canonicalize the SAME mask to
            # exactly representable BF16 0/-inf, retaining all valid positions.
            zero = torch.zeros((), dtype=torch.bfloat16, device=q.device)
            negative_infinity = torch.full((), float("-inf"), dtype=torch.bfloat16, device=q.device)
            attention_mask = torch.where(mask, zero, negative_infinity)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
        y = y.float().transpose(1, 2).reshape(batch, length, self.head_dim * self.n_head)
        return self.wo(y)


def prepare_acoustic_model(model: nn.Module, role_specs: Mapping[str, Mapping], *,
                           fir_polyphase: bool = False, cfm_sdpa_bf16: bool = False) -> dict:
    """Install an explicit, path-keyed recipe into a fresh floating point model.

    Keys are actual ``model.named_modules()`` paths, relative to the CFM
    estimator or BigVGAN root. Unlisted weighted operators remain FP32 and are
    enumerated in the returned manifest. Unknown paths fail rather than silently
    reducing quantization coverage.
    """
    for path, module in model.named_modules():
        if type(module).__module__.startswith("inspark_infer.ops."):
            raise ValueError(f"Expected fresh upstream model, found optimized wrapper at {path}")
    folded = fold_weight_norm_(model)
    modules = dict(model.named_modules())
    attention_paths = []
    if cfm_sdpa_bf16:
        from inspark_infer.models.indextts2.upstream.s2mel.modules.gpt_fast.model import Attention
        attention_paths = [path for path, module in modules.items() if isinstance(module, Attention)]
        if not attention_paths or "" in attention_paths:
            raise ValueError("CFM BF16 SDPA requires a component containing upstream CFM Attention modules")
    missing = set(role_specs) - set(modules)
    if missing:
        raise ValueError(f"Unknown acoustic role paths: {sorted(missing)}")
    alias_paths = []
    for path, module in modules.items():
        if type(module).__name__ == "Activation1d" and all(hasattr(module, key) for key in ("act", "upsample", "downsample")):
            parent_name, _, child = path.rpartition(".")
            modules[parent_name].add_module(child, StaticAliasFree(module, fir_polyphase=fir_polyphase))
            alias_paths.append(path)
    manifest = []
    for path, module in modules.items():
        if path in role_specs and not isinstance(module, WEIGHTED_TYPES):
            raise ValueError(f"Precision role is not Linear/Conv: {path}")
        if not isinstance(module, WEIGHTED_TYPES):
            continue
        spec = dict(role_specs.get(path, {"precision": "fp32"}))
        if not path:
            raise ValueError("Pass a component container rather than a bare weighted operator")
        if spec["precision"] != "fp32":
            parent_name, _, child = path.rpartition(".")
            modules[parent_name].add_module(child, ExportWeightOp(module, spec))
        manifest.append({"path": path, "operator": type(module).__name__,
                         "precision": spec["precision"], "smoothing_source": spec.get("smoothing_source", "none"),
                         "smooth_scale_present": spec.get("smooth_scale") is not None,
                         "weight_axis": spec.get("weight_axis", 0) if spec["precision"] in ("int8", "fp8") else None,
                         "weight_shape": list(module.weight.shape),
                         "conv_transpose_rewrite": spec.get("conv_transpose_rewrite")})
    # Install after weighted-op replacements, so the wrapper retains exactly
    # the original role paths and newly prepared projection module objects.
    for path in attention_paths:
        parent_name, _, child = path.rpartition(".")
        modules[parent_name].add_module(child, CFMBF16SDPAAttention(modules[path]))
    return {"weight_norm_folded": folded, "roles": manifest, "custom_plugins": [],
            "static_alias_free_graphs": alias_paths,
            "fir_polyphase_graphs": alias_paths if fir_polyphase else [],
            "cfm_sdpa_bf16_graphs": attention_paths,
            "arithmetic_overrides": ({"cfm_sdpa_qkv_dtype": "bfloat16", "cfm_sdpa_output_dtype": "float32"}
                                     if cfm_sdpa_bf16 else {}),
            "interface_precision": "fp32", "unlisted_weighted_roles": [r["path"] for r in manifest if r["path"] not in role_specs]}


def acoustic_specs_from_artifact(model, bindings, artifact, component):
    """Resolve canonical shared-policy roles to this component's module paths."""
    physical_paths = {id(module): path for path, module in model.named_modules()}
    selected = [role for role in bindings if role.component == component]
    if not selected:
        raise ValueError(f"No quantization roles for {component}")
    specs, mapping = {}, {}
    for role in selected:
        if role.path not in artifact["role_specs"]:
            raise ValueError(f"Calibration omits required role: {role.path}")
        physical = physical_paths.get(id(role.module))
        if physical is None:
            raise ValueError(f"Calibration role does not belong to the model: {role.path}")
        spec = artifact["role_specs"][role.path]
        if spec["precision"] != role.precision:
            raise ValueError(f"Protected/quantized precision policy mismatch at {role.path}")
        if role.precision == "int8" and spec.get("smooth_scale") is None:
            raise ValueError(f"INT8 SmoothQuant role has no smoothing scale: {role.path}")
        specs[physical] = spec
        mapping[physical] = role.path
    declared = {name for name in artifact["role_specs"] if name.startswith(component + ".")}
    if declared != {role.path for role in selected}:
        raise ValueError(f"Calibration/model role inventory differs for {component}")
    return specs, mapping


class FourStepCFM(nn.Module):
    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.model = estimator
        sample = next(estimator.parameters(), None)
        if sample is None:
            sample = next(estimator.buffers(), torch.empty(0))
        self.register_buffer("times", torch.tensor(INTERVALS, dtype=torch.float32, device=sample.device))

    def forward(self, x, prompt, lengths, style, mu, mask):
        current = x.float().masked_fill(mask, 0)
        for index in range(4):
            velocity = self.model(current, prompt, lengths, self.times[index:index + 1].expand(x.shape[0], -1), style, mu)
            current = (current + 0.25 * velocity.float()).masked_fill(mask, 0)
        return current


class TailWaveNetDiT(nn.Module):
    """Export-only output pruning for a fixed masked-prompt four-step solver.

    Transformer attention still sees every frame. The local WaveNet tail keeps
    the complete convolution halo and the original right boundary. Prompt
    velocities are zero because FourStepCFM discards them at every interval.
    This wrapper is not a general replacement for an unmasked estimator.
    """
    def __init__(self, original: nn.Module, prompt_frames=258):
        super().__init__()
        if (original.final_layer_type != 'wavenet' or original.time_as_token
                or original.style_as_token or original.training):
            raise ValueError('Tail WaveNet requires an eval DiT without time/style tokens')
        radius = 0
        for layer in original.wavenet.in_layers:
            conv = layer.conv.conv
            if layer.causal or conv.stride != (1,) or conv.kernel_size[0] % 2 != 1:
                raise ValueError('Tail WaveNet requires noncausal odd stride-one convolutions')
            radius += (conv.kernel_size[0] - 1) * conv.dilation[0] // 2
        if prompt_frames <= radius:
            raise ValueError('Prompt must contain the full WaveNet receptive-field halo')
        self.original = original
        self.prompt_frames, self.radius = int(prompt_frames), radius

    def forward(self, x, prompt, lengths, times, style, condition):
        from inspark_infer.models.indextts2.upstream.s2mel.modules.commons import sequence_mask
        m = self.original
        batch, _, frames = x.shape
        t1 = m.t_embedder(times)
        cond = m.cond_projection(condition)
        xt = x.transpose(1, 2)
        embedded = torch.cat((xt, prompt.transpose(1, 2), cond), -1)
        if m.transformer_style_condition:
            embedded = torch.cat((embedded, style[:, None, :].repeat(1, frames, 1)), -1)
        embedded = m.cond_x_merge_linear(embedded)
        mask = sequence_mask(lengths, max_length=frames).to(x.device).unsqueeze(1)
        attention_mask = mask[:, None, :].repeat(1, 1, frames, 1) if not m.is_causal else None
        hidden = m.transformer(embedded, t1.unsqueeze(1), m.input_pos[:frames], attention_mask)
        start = self.prompt_frames - self.radius
        hidden = hidden[:, start:]
        if m.long_skip_connection:
            hidden = m.skip_linear(torch.cat((hidden, xt[:, start:]), -1))
        tail_mask = mask[:, :, start:]
        local = m.conv1(hidden).transpose(1, 2)
        t2 = m.t_embedder2(times)
        local = m.wavenet(local, tail_mask, g=t2.unsqueeze(2)).transpose(1, 2) + m.res_projection(hidden)
        velocity = m.conv2(m.final_layer(local, t1).transpose(1, 2))
        return F.pad(velocity[:, :, self.radius:], (self.prompt_frames, 0)).reshape(x.shape)


def inspect_standard_onnx(path: Path) -> dict:
    import onnx
    from collections import Counter

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(str(path))
    illegal = [(node.name, node.domain, node.op_type) for node in model.graph.node if node.domain not in ("", "ai.onnx")]
    if illegal:
        raise ValueError(f"Acoustic ONNX contains nonstandard nodes: {illegal[:8]}")
    return {"nodes": len(model.graph.node), "operators": dict(Counter(node.op_type for node in model.graph.node)),
            "custom_domains": [], "plugins": []}


def export_standard_onnx(model, inputs, output: Path, *, input_names, output_names):
    if any(getattr(module, "weight_quantization_folded", False) for module in model.modules()):
        raise ValueError("Torch reference has folded weight Q/DQ; export a fresh model to preserve Q/DQ nodes")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        result = model(*inputs)
        tensors = result if isinstance(result, tuple) else (result,)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("Non-finite acoustic export output")
        torch.onnx.export(model, inputs, str(output), export_params=True, opset_version=20,
                          do_constant_folding=True, input_names=input_names, output_names=output_names,
                          dynamo=False, external_data=True)
    return inspect_standard_onnx(output)
