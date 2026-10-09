"""Exact INT8 convolution window-storage probe, prior to a full-engine rebuild."""
import argparse,json,hashlib,inspect
from pathlib import Path
import torch
from torch import nn
from inspark_infer.build.unified_acoustic_export import ExportWeightOp,inspect_standard_onnx
from inspark_infer.build.modern_qdq_export import translations

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');a=p.parse_args()
    if 'conv1d_quantized_columns' not in inspect.getsource(ExportWeightOp.forward):
        raise RuntimeError('Probe archived: first apply deployment/b32/history/integer-columns-candidate.patch; this candidate was not retained')
    torch.manual_seed(4108)
    spec=dict(precision='int8',smooth_scale=[1.]*24,input_scale=.03125,
        weight_scale=[.0078125]*24,weight_axis=0)
    module=nn.Conv1d(24,24,11,padding=5)
    op=ExportWeightOp(module,spec);op.conv1d_as_gemm=True
    values=torch.randn(32,24,137)
    expected=op(values);op.conv1d_quantized_columns=True;observed=op(values)
    assert torch.equal(expected,observed)
    # Include zero-insertion/deconvolution and nontrivial smoothing.
    deconv=ExportWeightOp(nn.ConvTranspose1d(8,12,4,stride=2,padding=1),
        dict(precision='int8',smooth_scale=[1.125]*8,input_scale=.0625,
             weight_scale=[.015625]*12,weight_axis=1,conv_transpose_rewrite='zero_insert_conv'))
    deconv.conv1d_as_gemm=True;x=torch.randn(32,8,19)
    y=deconv(x);deconv.conv1d_quantized_columns=True;assert torch.equal(y,deconv(x))
    h=ROOT/'deployment/b32/history';h.joinpath('integer-columns-cpu-equivalence.json').write_text(json.dumps(dict(passed=True,batch=32,conv=True,deconv_zero_insert=True,bit_identical=True),indent=2)+'\n')
    if not a.export:return
    from inspark_infer.runtime.device import GPULease,select_gpu
    with GPULease(1):
        select_gpu(1);op.cuda().eval().requires_grad_(False);op.modern_export=True
        inputs=(torch.randn(32,24,13312,device='cuda'),)
        for integer in (False,True):
            op.conv1d_quantized_columns=integer
            path=ROOT/'.cache/b32-columns-probe'/('integer' if integer else 'floating')/'model.onnx';path.parent.mkdir(parents=True,exist_ok=True)
            torch.onnx.export(op,inputs,str(path),opset_version=20,dynamo=True,external_data=True,
                input_names=['x'],output_names=['y'],custom_translation_table=translations(),optimize=True)
            print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),graph=inspect_standard_onnx(path))),flush=True)


if __name__=='__main__':main()
