"""Single physical GPU selection and cooperative ownership across benchmark processes."""
from pathlib import Path
import os,sys,fcntl,subprocess

def select_gpu(index):
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():raise RuntimeError('Select GPU before CUDA initialization')
    os.environ['CUDA_VISIBLE_DEVICES']=str(index)

class GPULease:
    def __init__(self,index):self.index=str(index);self.handle=None
    def __enter__(self):
        root=Path('/tmp/acc-infer-gpu-locks');root.mkdir(exist_ok=True)
        self.handle=(root/(self.index+'.lock')).open('a+')
        try:
            fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            used,util=map(int,subprocess.check_output(['nvidia-smi','-i',self.index,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            # Diagnostic escape hatch for a known orphaned CUDA context.  The
            # default remains strict; callers must explicitly bound how much
            # pre-existing memory they have already identified as their own.
            memory_limit=int(os.environ.get('ACC_GPU_EXISTING_MEMORY_LIMIT_MIB','1024'))
            if used>memory_limit or util>10:raise RuntimeError(f'GPU{self.index} is busy ({used}MiB,{util}%)')
        except BaseException:self.__exit__();raise
        return self
    def __exit__(self,*args):
        if self.handle:self.handle.close();self.handle=None
