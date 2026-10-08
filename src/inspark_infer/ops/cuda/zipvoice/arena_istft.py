"""Full-batch CENTER ISTFT using storage released by serial Vocos execution."""
import math
import ctypes
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def spectrum_kernel(C, S, B: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    valid = i < B*F*513
    b = i//(F*513)
    frame = (i//513)%F
    freq = i%513
    mag = libdevice.exp(tl.load(C+b*1026*F+freq*F+frame, valid, 0))
    mag = tl.minimum(mag, 100.)
    phase = tl.load(C+b*1026*F+(freq+513)*F+frame, valid, 0)
    tl.store(S+2*i, mag*libdevice.cos(phase), valid)
    tl.store(S+2*i+1, mag*libdevice.sin(phase), valid)


@triton.jit
def overlap_add_kernel(X, W, O, B: tl.constexpr, F: tl.constexpr, N: tl.constexpr,
                       SCALE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    valid = i < B*N
    batch = i//N
    sample = i%N+512
    latest = sample//256
    numerator = tl.full((BLOCK,), 0., tl.float32)
    denominator = tl.full((BLOCK,), 0., tl.float32)
    for back in tl.static_range(3, -1, -1):
        frame = latest-back
        offset = sample-frame*256
        active = valid & (frame >= 0) & (frame < F) & (offset >= 0) & (offset < 1024)
        window = tl.load(W+offset, active, 0)
        value = tl.load(X+(batch*F+frame)*1024+offset, active, 0)*SCALE
        numerator = numerator+value*window
        denominator = denominator+window*window
    result = numerator/denominator
    tl.store(O+i, tl.minimum(tl.maximum(result, -1.), 1.), valid)


@triton.jit
def scale_waves_kernel(W, R, B: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    valid=i<B*N
    rms=tl.load(R+i//N,valid,1.)
    wave=tl.load(W+i,valid,0.)
    scaled=wave*rms/.1
    tl.store(W+i,tl.where(rms<.1,scaled,wave),valid)


class CufftC2R:
    """Single-GPU batched C2R with explicit caller-owned workspace."""
    def __init__(self, transforms):
        path=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13/lib/libcufft.so.12'
        if not path.resolve().is_relative_to(Path('/workspace')):
            raise ValueError('cuFFT must come from the project environment')
        self.lib=ctypes.CDLL(str(path))
        self.handle=ctypes.c_int(0)
        integer=ctypes.c_int;pointer=ctypes.c_void_p
        signatures={
            'cufftCreate':[ctypes.POINTER(integer)],
            'cufftSetAutoAllocation':[integer,integer],
            'cufftMakePlanMany':[integer,integer,ctypes.POINTER(integer),ctypes.POINTER(integer),
                                 integer,integer,ctypes.POINTER(integer),integer,integer,integer,
                                 integer,ctypes.POINTER(ctypes.c_size_t)],
            'cufftSetWorkArea':[integer,pointer], 'cufftSetStream':[integer,pointer],
            'cufftExecC2R':[integer,pointer,pointer], 'cufftDestroy':[integer],
            'cufftGetVersion':[ctypes.POINTER(integer)]}
        for name,args in signatures.items():
            function=getattr(self.lib,name);function.argtypes=args;function.restype=integer
        self.check('cufftCreate',ctypes.byref(self.handle))
        self.check('cufftSetAutoAllocation',self.handle,0)
        n=(integer*1)(1024);inp=(integer*1)(513);out=(integer*1)(1024)
        size=ctypes.c_size_t()
        # CUFFT_C2R=0x2c from the installed CUDA13 cufft.h.
        self.check('cufftMakePlanMany',self.handle,1,n,inp,1,513,out,1,1024,0x2c,
                   transforms,ctypes.byref(size))
        self.workspace_bytes=size.value
        version=integer();self.check('cufftGetVersion',ctypes.byref(version));self.version=version.value

    def check(self,name,*arguments):
        code=getattr(self.lib,name)(*arguments)
        if code:raise RuntimeError(f'{name} returned cuFFT error {code}')

    def execute(self,spectrum,output,workspace):
        self.check('cufftSetStream',self.handle,torch.cuda.current_stream().cuda_stream)
        self.check('cufftSetWorkArea',self.handle,workspace.data_ptr())
        self.check('cufftExecC2R',self.handle,spectrum.data_ptr(),output.data_ptr())

    def close(self):
        if self.handle.value:
            self.check('cufftDestroy',self.handle);self.handle.value=0

    def __del__(self):
        if getattr(self,'handle',None) is not None and self.handle.value:
            self.lib.cufftDestroy(self.handle)


class ArenaISTFT:
    def __init__(self, arena, dead_scratch_bytes, batch, frames, window, direct_cufft=False):
        self.batch, self.frames = batch, frames
        self.samples = (frames-1)*256
        if (arena.dtype != torch.uint8 or not arena.is_contiguous() or
                window.shape != (1024,) or window.dtype != torch.float32 or not window.is_cuda):
            raise ValueError('Expected contiguous CUDA byte arena and original F32 Hann window')
        self.window = window
        offset = 0
        def view(shape, dtype):
            nonlocal offset
            size = math.prod(shape)*torch.empty((), dtype=dtype).element_size()
            tensor = arena.narrow(0, offset, size).view(dtype).reshape(shape)
            offset = (offset+size+255)//256*256
            return tensor
        self.spectrum = view((batch, frames, 513), torch.complex64)
        self.time_frames = view((batch, frames, 1024), torch.float32)
        self.waves = view((batch, self.samples), torch.float32)
        self.fft = CufftC2R(batch*frames) if direct_cufft else None
        self.fft_workspace = view((max(256,self.fft.workspace_bytes),),torch.uint8) if self.fft else None
        if offset > dead_scratch_bytes or offset > arena.numel():
            raise ValueError('ISTFT views overlap live Vocos coefficients or exceed the arena')
        self.storage_bytes = offset

    def scale_rms_(self,rms):
        if rms.shape!=(self.batch,1) or rms.dtype!=torch.float32 or not rms.is_contiguous():
            raise ValueError('Expected original F32 per-row prompt RMS')
        scale_waves_kernel[(triton.cdiv(self.batch*self.samples,1024),)](
            self.waves,rms,self.batch,self.samples,BLOCK=1024,enable_fp_fusion=False)
        return self.waves

    @torch.inference_mode()
    def __call__(self, coefficients):
        if (coefficients.shape != (self.batch, 1026, self.frames) or
                coefficients.dtype != torch.float32 or not coefficients.is_cuda or
                not coefficients.is_contiguous()):
            raise ValueError('Expected original complete F32 Vocos coefficients')
        if coefficients.data_ptr() < self.waves.data_ptr()+self.waves.numel()*4 and coefficients.data_ptr()+coefficients.numel()*4 > self.spectrum.data_ptr():
            raise ValueError('Live coefficients overlap ISTFT intermediate storage')
        spectrum_kernel[(triton.cdiv(self.batch*self.frames*513, 1024),)](
            coefficients, self.spectrum.view(torch.float32), self.batch, self.frames,
            BLOCK=1024, enable_fp_fusion=False)
        if self.fft:
            self.fft.execute(self.spectrum,self.time_frames,self.fft_workspace)
        else:
            torch.fft.irfft(self.spectrum, n=1024, dim=-1, norm='backward', out=self.time_frames)
        overlap_add_kernel[(triton.cdiv(self.batch*self.samples, 1024),)](
            self.time_frames, self.window, self.waves, self.batch, self.frames, self.samples,
            SCALE=1./1024 if self.fft else 1., BLOCK=1024, enable_fp_fusion=False)
        return self.waves
