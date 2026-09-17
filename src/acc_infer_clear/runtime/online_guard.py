"""Validation-only assertions against online compilation/capture, not a backend."""
import sys

class OnlineGuard:
    def __init__(self):
        import torch
        from triton.runtime.jit import JITFunction
        self.originals=[];self.calls={'torch_compile':0,'capture':0,'triton_compile':0}
        def reject(kind):
            def fail(*args,**kwargs):
                self.calls[kind]+=1
                raise RuntimeError('Unexpected online '+kind)
            return fail
        for owner,name,kind in [(torch,'compile','torch_compile'),(torch.cuda,'CUDAGraph','capture'),(torch.cuda,'graph','capture')]:
            self.originals.append((owner,name,getattr(owner,name)));setattr(owner,name,reject(kind))
        seen=set()
        for name,module in list(sys.modules.items()):
            if not name.startswith('acc_infer_clear.kernels'):continue
            for value in vars(module).values():
                if isinstance(value,JITFunction) and id(value) not in seen and hasattr(value,'compile'):
                    seen.add(id(value));self.originals.append((value,'compile',value.compile));value.compile=reject('triton_compile')
    def close(self):
        for owner,name,value in reversed(self.originals):setattr(owner,name,value)
        self.originals=[]
