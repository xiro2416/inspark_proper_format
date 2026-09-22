#!/usr/bin/env python3
"""Export a static BigVGAN with explicit TensorRT alias-free plugin nodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--frames", type=int, default=52)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--output", default="artifacts/trt113_vocoder/vocoder_b8.onnx")
    parser.add_argument("--regular-conv", choices=("plugin", "native"), default="plugin")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import torch.nn.functional as F
    from torch import nn
    from acc_infer_clear.config import load
    from acc_infer_clear.runtime.device import GPULease
    from acc_infer_clear.streaming.engine import Engine

    class AliasFreePluginFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up_filter, down_filter, alpha, inverse_beta):
            channels = x.shape[1]
            up = F.pad(x, (5, 5), mode="replicate")
            up = 2.0 * F.conv_transpose1d(
                up, up_filter.expand(channels, -1, -1), stride=2, groups=channels
            )
            up = up[..., 15:-15]
            activated = up + inverse_beta[None, :, None] * torch.sin(
                up * alpha[None, :, None]
            ).square()
            padded = F.pad(activated, (5, 6), mode="replicate")
            y = F.conv1d(
                padded, down_filter.expand(channels, -1, -1), stride=2,
                groups=channels,
            )
            return y, activated

        @staticmethod
        def symbolic(g, x, up_filter, down_filter, alpha, inverse_beta):
            return g.op(
                "inspark::alias_free", x, up_filter, down_filter, alpha,
                inverse_beta, outputs=2,
                plugin_version_s="1", plugin_namespace_s="inspark",
            )

    class ExportAliasFree(nn.Module):
        def __init__(self, original):
            super().__init__()
            activation = original.act
            alpha = activation.alpha.detach().float()
            beta = getattr(activation, "beta", activation.alpha).detach().float()
            if activation.alpha_logscale:
                alpha = alpha.exp(); beta = beta.exp()
            self.register_buffer("up_filter", original.upsample.filter.detach().float().contiguous())
            self.register_buffer("down_filter", original.downsample.lowpass.filter.detach().float().contiguous())
            self.register_buffer("alpha", alpha.contiguous())
            self.register_buffer("inverse_beta", (1.0 / (beta + 1e-9)).contiguous())

        def forward(self, x):
            return AliasFreePluginFunction.apply(
                x.float(), self.up_filter, self.down_filter, self.alpha,
                self.inverse_beta,
            )[0]

    class DeconvPluginFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, stride, padding, output_padding, dilation, groups):
            return F.conv_transpose1d(
                x.bfloat16(), weight.bfloat16(), bias.bfloat16(),
                (stride,), (padding,), (output_padding,),
                groups, (dilation,),
            ).float()

        @staticmethod
        def symbolic(g, x, weight, bias, stride, padding, output_padding, dilation, groups):
            return g.op(
                "inspark::deconv1d", x, weight, bias,
                stride_i=stride, padding_i=padding,
                output_padding_i=output_padding, dilation_i=dilation,
                groups_i=groups, plugin_version_s="1",
                plugin_namespace_s="inspark",
            )

    class ExportDeconvPlugin(nn.Module):
        def __init__(self, original):
            super().__init__()
            for name in ("stride", "padding", "output_padding", "groups", "dilation"):
                setattr(self, name, getattr(original, name))
            self.register_buffer("weight", original.weight.detach().float().clone())
            self.register_buffer("bias", (original.bias.detach().float().clone()
                                          if original.bias is not None else None))

        def forward(self, x):
            dtype = x.dtype
            return DeconvPluginFunction.apply(
                x, self.weight, self.bias, self.stride[0],
                self.padding[0], self.output_padding[0], self.dilation[0],
                self.groups,
            ).to(dtype)

    class ConvPluginFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,weight,bias,stride,padding,dilation,groups):
            return F.conv1d(
                x.bfloat16(),weight.bfloat16(),bias.bfloat16(),stride,padding,
                dilation,groups,
            ).float()

        @staticmethod
        def symbolic(g,x,weight,bias,stride,padding,dilation,groups):
            return g.op(
                "inspark::conv1d",x,weight,bias,stride_i=stride,
                padding_i=padding,dilation_i=dilation,groups_i=groups,
                plugin_version_s="1",plugin_namespace_s="inspark",
            )

    class ExportConvPlugin(nn.Module):
        def __init__(self,original):
            super().__init__()
            for name in ("stride","padding","groups","dilation"):
                setattr(self,name,getattr(original,name))
            self.register_buffer("weight",original.weight.detach().float().clone())
            bias=(original.bias.detach().float().clone() if original.bias is not None
                  else torch.zeros(original.out_channels,device=original.weight.device))
            self.register_buffer("bias",bias)

        def forward(self,x):
            return ConvPluginFunction.apply(
                x,self.weight,self.bias,self.stride[0],self.padding[0],
                self.dilation[0],self.groups,
            )

    def replace_alias(module):
        changed = []
        def walk(parent, prefix):
            for name, child in list(parent.named_children()):
                path = f"{prefix}.{name}"
                if type(child).__name__ == "Activation1d":
                    parent.add_module(name, ExportAliasFree(child)); changed.append(path)
                else:
                    walk(child, path)
        walk(module, "vocoder")
        return changed

    def replace_convs(module):
        changed = [];deconvs=[]
        def walk(parent, prefix):
            for name, child in list(parent.named_children()):
                path = f"{prefix}.{name}"
                if type(child).__name__ == "MatrixConv":
                    if getattr(child,"transpose",False):
                        parent.add_module(name,ExportDeconvPlugin(child));deconvs.append(path)
                    elif args.regular_conv == "plugin":
                        parent.add_module(name,ExportConvPlugin(child));changed.append(path)
                else:
                    walk(child, path)
        walk(module, "vocoder")
        return changed,deconvs

    config = load(args.config); config["max_batch"] = args.batch
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    with GPULease(args.gpu):
        engine = Engine(config)
        try:
            engine.prepare_precision("bf16", ["target", "draft", "cfm", "vocoder"], True)
            model = engine.tts.bigvgan.eval().requires_grad_(False)
            device = next(model.buffers()).device
            example = torch.randn(
                args.batch, 80, args.frames, device=device, dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(113),
            )
            with torch.inference_mode():
                baseline = model(example); torch.cuda.synchronize()
            changed = replace_alias(model)
            convs,deconvs = replace_convs(model)
            with torch.inference_mode():
                expected = model(example); torch.cuda.synchronize()
            delta = (baseline - expected).abs()
            torch.onnx.export(
                model, (example,), str(output), export_params=True,
                opset_version=20, do_constant_folding=True,
                input_names=["mel"], output_names=["pcm"], dynamo=False,
                external_data=True, custom_opsets={"inspark": 1},
            )
            report = {
                "batch": args.batch, "frames": args.frames,
                "onnx": str(output), "bytes": output.stat().st_size,
                "input": {"shape": list(example.shape), "dtype": str(example.dtype)},
                "output": {"shape": list(expected.shape), "dtype": str(expected.dtype)},
                "plugin_nodes": len(changed), "plugin_paths": changed,
                "conv_plugins": convs, "deconv_plugins": deconvs,
                "export_rewrite_validation": {
                    "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                    "cosine": float(torch.nn.functional.cosine_similarity(
                        baseline.float().flatten(), expected.float().flatten(), dim=0)),
                },
                "opset": 20,
            }
            output.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        finally:
            engine.close()


if __name__ == "__main__":
    main()
