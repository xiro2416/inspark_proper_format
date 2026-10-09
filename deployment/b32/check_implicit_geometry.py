"""Exact discrete checks for dilation, stride, zero-insertion and bias semantics."""
import json
from pathlib import Path
import torch
from torch import nn
from inspark_infer.build.unified_acoustic_export import ExportWeightOp
from deployment.b32.implicit_int8_probe import ImplicitConv

ROOT=Path(__file__).resolve().parents[2]


def main():
    from inspark_infer.runtime.device import GPULease,select_gpu
    cases=[]
    with GPULease(1):
        select_gpu(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with torch.inference_mode():
            modules=[nn.Conv1d(8,12,3,padding=3,dilation=3),nn.Conv1d(8,12,7,stride=2,padding=3),
                     nn.ConvTranspose1d(8,12,4,stride=2,padding=1),nn.ConvTranspose1d(8,12,4,stride=2,padding=1,output_padding=1)]
            for i,m in enumerate(modules):
                m.weight.copy_((torch.arange(m.weight.numel()).reshape_as(m.weight)%7-3)*.125)
                m.bias.copy_(torch.arange(12)*.03125)
                deconv=isinstance(m,nn.ConvTranspose1d)
                spec=dict(precision='int8',smooth_scale=[1.125]*8,input_scale=.25,weight_scale=[.125]*12,weight_axis=1 if deconv else 0)
                if deconv:spec['conv_transpose_rewrite']='zero_insert_conv'
                reference=ExportWeightOp(m.cuda(),spec)
                x=((torch.arange(32*8*19,device='cuda').reshape(32,8,19)%9-4)*.25).float()
                expected=reference(x)
                w=(reference.weight/reference.weight_scale[:,None,None]).round().clamp(-128,127).to(torch.int8)
                op=ImplicitConv(w,reference.input_scale,reference.weight_scale,reference.bias,reference.smooth_scale,
                    stride=1 if deconv else m.stride[0],padding=m.dilation[0]*(m.kernel_size[0]-1)-m.padding[0] if deconv else m.padding[0],
                    dilation=m.dilation[0],expand=m.stride[0] if deconv else 1)
                # Output padding changes the legal right extent, not coordinate mapping.
                if deconv and m.output_padding[0]:
                    op.output_padding=m.output_padding[0]
                observed=op(x)
                if not torch.equal(expected,observed):raise RuntimeError('Discrete operation logic mismatch: '+str(i))
                cases.append(dict(case=i,type=type(m).__name__,output_shape=list(observed.shape),bit_identical=True))
    (ROOT/'deployment/b32/history/implicit-geometry-checks.json').write_text(json.dumps(dict(passed=True,cases=cases),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(cases))))


if __name__=='__main__':main()
