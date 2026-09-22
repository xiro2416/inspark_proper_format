"""Isolated FP32 N1280 LN+row-FP8 experiment, not approved deployment.

Offline calibration must establish finite ordinary inputs and bitwise Y/S/Q.
No online tensor-value checks, autotuning or implicit compilation are performed.
Unsupported shapes must be routed to the previous deployment by the caller.
"""
import hashlib,os
from pathlib import Path
_extension=None

def prepare():
    global _extension
    if _extension is not None:return _extension
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_suffix('.cu')
    name='target_norm_quant_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    prior=os.environ.get('TORCH_CUDA_ARCH_LIST');os.environ['TORCH_CUDA_ARCH_LIST']='12.0'
    try:
        _extension=load(name=name,sources=[str(source)],extra_cflags=['-O3'],extra_cuda_cflags=[
            '-O3','-lineinfo','--fmad=true','--ftz=false','--prec-div=true','--prec-sqrt=true','-Xptxas=-v'],verbose=True)
    finally:
        if prior is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:os.environ['TORCH_CUDA_ARCH_LIST']=prior
    return _extension

def fused(x,w,b,eps=1e-5,emit_y=False,division=0,sync_strategy=0):
    """Return (Q: float8_e4m3fn [M,1280], S: FP32[M], Y or None).

    division=0: explicit PTX div.full.f32, matching upstream Triton PTX.
    division=1: rcp.approx.ftz + mul.rn counterfactual, NOT pre-approved.
    sync_strategy=0: three-barrier control, reused statistics shared storage.
    sync_strategy=1: separate amax storage, one barrier and per-thread scale.
    """
    if _extension is None:raise RuntimeError('Run prepare() explicitly OFFLINE first')
    import torch
    q,s,y=_extension.forward(x,w,b,eps,emit_y,division,sync_strategy)
    return q.view(torch.float8_e4m3fn),s,y if emit_y else None
