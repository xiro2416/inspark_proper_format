"""Existing FP32 acoustic-condition graph with exact time and profiled batch."""
from inspark_infer.runtime.bundle_paths import read_json
import hashlib,json
from pathlib import Path
import torch
from inspark_infer.ops.tensorrt.native113 import _import_trt113


class TRTConditionBank:
    def __init__(self,manifest):
        path=Path(manifest).resolve();m=read_json(path)
        if m['precision']!='fp32' or m['tf32'] or m['quantization_changed']:
            raise ValueError('Condition engine must preserve protected FP32')
        trt=self.trt=_import_trt113()
        if m['trt']!=trt.__version__:raise ValueError('Condition TRT version mismatch')
        engine_path=path.parent/m['engine'];blob=engine_path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=m['sha256']:raise ValueError('Condition engine hash mismatch')
        for source in m.get('runtime_model_sources',m['provenance']['model_sources']):
            if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:
                raise ValueError('Condition checkpoint changed')
        self.runtime=trt.Runtime(trt.Logger(trt.Logger.ERROR));self.engine=self.runtime.deserialize_cuda_engine(blob)
        self.context=self.engine.create_execution_context();self.count=m['count'];self.frames=m['frames'];self.max_batch=m['max_batch']
        self.context.set_input_shape('codes',(self.max_batch,self.count));self.context.set_input_shape('latents',(self.max_batch,self.count,1280))
        self.output=torch.empty(tuple(self.context.get_tensor_shape('condition')),device='cuda',dtype=torch.float32)
        self.hits=0;self.misses=0;self.graphs={self.count:self.engine}
        self.backend='TensorRT FP32 exact-time conditions, dynamic batch'

    def run(self,codes,latents,count):
        if count!=self.count or not 1<=len(codes)<=self.max_batch:
            self.misses+=1;return None
        c=torch.cat(codes,0);x=torch.cat(latents,0)
        if c.dtype!=torch.int64 or x.dtype!=torch.float32:raise ValueError('Condition dtype changed')
        self.context.set_input_shape('codes',tuple(c.shape));self.context.set_input_shape('latents',tuple(x.shape))
        for name,value in (('codes',c),('latents',x),('condition',self.output)):
            if not self.context.set_tensor_address(name,value.data_ptr()):raise RuntimeError('Condition binding failed')
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError('Condition enqueue failed')
        # Bindings live until the current stream completes, including async
        # consumers on the parent stream. Allocator observes their producer.
        c.record_stream(torch.cuda.current_stream());x.record_stream(torch.cuda.current_stream())
        self.hits+=1;return self.output[:len(codes)]


class TRTConditionCoverage:
    def __init__(self,manifest):
        path=Path(manifest).resolve();m=read_json(path)
        if m['kind']!='protected_fp32_condition_bank':raise ValueError('Invalid condition coverage manifest')
        self.entries={int(n):TRTConditionBank(path.parent/p if not Path(p).is_absolute() else p)
                      for n,p in m['plans'].items()}
        self.graphs={n:entry.engine for n,entry in self.entries.items()}
        self.hits=0;self.misses=0;self.backend='TensorRT protected FP32 exact-time condition coverage'

    def run(self,codes,latents,count):
        entry=self.entries.get(count)
        if entry is None:self.misses+=1;return None
        result=entry.run(codes,latents,count)
        if result is None:self.misses+=1
        else:self.hits+=1
        return result
