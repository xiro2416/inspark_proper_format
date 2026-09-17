"""Explicit offline build only; no automatic compilation in forward."""
from pathlib import Path
import hashlib,os
_extension=None

def prepare():
    global _extension
    if _extension is not None:return _extension
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_suffix('.cu')
    name='target_exact_ln_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    # Toolkit supports SM120; no fast-math, fmad remains NVCC default true.
    # Device-specific offline experiment, not a portable deployment claim.
    old=os.environ.get('TORCH_CUDA_ARCH_LIST');os.environ['TORCH_CUDA_ARCH_LIST']='12.0'
    try:
        _extension=load(name=name,sources=[str(source)],extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3','-lineinfo','--fmad=true','--ftz=false','--prec-div=true','--prec-sqrt=true','-Xptxas=-v'],verbose=True)
    finally:
        if old is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:os.environ['TORCH_CUDA_ARCH_LIST']=old
    return _extension

def layernorm(x,w,b,eps=1e-5,mode=2):
    if _extension is None:raise RuntimeError('Call exact_norm.prepare() OFFLINE before inference/capture')
    return _extension.forward(x,w,b,eps,mode)
