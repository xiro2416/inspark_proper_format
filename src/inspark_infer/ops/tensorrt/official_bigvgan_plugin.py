"""TRT adapter for NVIDIA's unchanged BigVGAN activation extension.

No GPU arithmetic is defined here. TensorRT owns its output; the vendor op
allocates a temporary, then the existing Torch copy operation fills that output.
The extra copy is deliberately included in subsequent performance tests.
"""
from inspark_infer.runtime.bundle_paths import read_json
import hashlib
import os
import importlib.util
import json
from pathlib import Path

NAME='nvidia_bigvgan::alias_free'
_MODULE=None


def vendor_module():
    global _MODULE
    if _MODULE is None:
        root=Path(os.environ.get('INSPARK_ASSET_ROOT',Path(__file__).resolve().parents[4]))
        record=read_json(root/'artifacts/sm120_0924/unified/gpu_inference_20261001/official_bigvgan_kernel.json')
        if record['math_source_modified'] or record['arch']!='sm120':
            raise ValueError('Expected unchanged NVIDIA source compiled for SM120')
        for name,digest in record['source_sha256'].items():
            if Path(name).exists() and hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError('NVIDIA activation source identity changed')
        binary=Path(record['module'])
        if not binary.exists():binary=root/'.cache/nvidia_bigvgan_sm120/inspark_nvidia_bigvgan_sm120.so'
        if hashlib.sha256(binary.read_bytes()).hexdigest()!=record.get('module_sha256'):
            raise ValueError('NVIDIA activation binary identity changed')
        spec=importlib.util.spec_from_file_location('inspark_nvidia_bigvgan_sm120',str(binary))
        _MODULE=importlib.util.module_from_spec(spec);spec.loader.exec_module(_MODULE)
    return _MODULE


def register():
    import tensorrt.plugin as trtp
    try:
        getattr(trtp.op.nvidia_bigvgan,'alias_free')
        return
    except AttributeError:pass
    vendor_module()

    @trtp.register(NAME)
    def desc(x:trtp.TensorDesc,up_filter:trtp.TensorDesc,down_filter:trtp.TensorDesc,
             alpha:trtp.TensorDesc,beta:trtp.TensorDesc)->trtp.TensorDesc:
        return x.like()

    @trtp.impl(NAME)
    def impl(x,up_filter,down_filter,alpha,beta,outputs,stream:int)->None:
        import torch
        tensors=[torch.as_tensor(t,device='cuda') for t in (x,up_filter,down_filter,alpha,beta)]
        output=torch.as_tensor(outputs[0],device='cuda')
        if any(t.dtype!=torch.float32 or not t.is_contiguous() for t in [*tensors,output]):
            raise ValueError('NVIDIA activation adapter requires contiguous FP32 tensors')
        external=torch.cuda.default_stream() if int(stream)==0 else torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(external):
            result=vendor_module().forward(*tensors)
            output.copy_(result)
