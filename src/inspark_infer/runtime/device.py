"""Single physical GPU selection and cooperative ownership across benchmark processes."""
from pathlib import Path
import os,sys,fcntl,subprocess,time

def select_gpu(index):
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():raise RuntimeError('Select GPU before CUDA initialization')
    os.environ['CUDA_VISIBLE_DEVICES']=str(index)

class GPULease:
    def __init__(self,index):self.index=str(index);self.handle=None
    def __enter__(self):
        root=Path(os.environ.get('ACC_GPU_LOCK_DIR','/workspace/.cache/inspark/gpu-locks'))
        if not root.resolve().is_relative_to(Path('/workspace')):
            raise ValueError('GPU lock directory must stay within /workspace')
        root.mkdir(parents=True,exist_ok=True)
        self.handle=(root/(self.index+'.lock')).open('a+')
        try:
            fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def sample():
                output=subprocess.check_output(['nvidia-smi','-i',self.index,
                    '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                return tuple(map(int,output.strip().split(',')))
            used,util=sample()
            # Diagnostic escape hatch for a known orphaned CUDA context.  The
            # default remains strict; callers must explicitly bound how much
            # pre-existing memory they have already identified as their own.
            memory_limit=int(os.environ.get('ACC_GPU_EXISTING_MEMORY_LIMIT_MIB','1024'))
            self.shared=os.environ.get('ACC_GPU_ALLOW_SHARED')=='1'
            # Utilization is sampled over a recent window and can briefly
            # reflect the preceding build child after its CUDA context exits.
            # Never retry a memory-over-limit device; only recheck a transient
            # high-utilization reading with little resident memory.
            if used<=memory_limit and util>10:
                for _ in range(4):
                    time.sleep(.25)
                    used,util=sample()
                    if used>memory_limit or util<=10:break
            self.initial_memory_mib=used;self.initial_utilization=util
            if (used>memory_limit and not self.shared) or util>10:
                raise RuntimeError(f'GPU{self.index} is busy ({used}MiB,{util}%)')
            if self.shared:
                print(f'GPU{self.index}: explicit shared-device run; pre-existing memory={used} MiB, utilization={util}%; external processes are preserved',file=sys.stderr)
        except BaseException:self.__exit__();raise
        return self
    def __exit__(self,*args):
        if self.handle:self.handle.close();self.handle=None
