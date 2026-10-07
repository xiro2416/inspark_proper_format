"""Standard TRT segments and unchanged NVIDIA activations in one parent Graph."""
from inspark_infer.runtime.bundle_paths import read_json
import hashlib
import json
from pathlib import Path
import torch
from inspark_infer.ops.tensorrt.unified_ar import StaticEngine
from inspark_infer.ops.tensorrt.official_bigvgan_plugin import vendor_module


class OfficialVocoderSegments:
    def __init__(self,partition,plan_suffix=None,fallback=None):
        path=Path(partition).resolve();self.partition_path=path
        self.manifest=read_json(path);m=self.manifest
        plan_suffix=plan_suffix or m.get('plan_suffix','l3m')
        custom=m['kind']=='custom_activation_trt_segments'
        if (m['kind'] not in ('official_vendor_trt_segments','custom_activation_trt_segments') or m['batch'] not in (64,128)
                or m['frames']!=52 or bool(m['new_gpu_math'])!=custom or custom and not m.get('custom_small_fir')):
            raise ValueError('Expected explicit fixed B64/B128 F52 vendor or custom-activation partition')
        for key in ('checkpoint',):
            r=m[key]
            if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:
                raise ValueError('Vocoder partition source identity changed')
        self.batch=m['batch'];self.frames=52;self.component='vocoder';self.flow=m['flow'];self.engines={};self.parameters={};self.custom_activations={}
        self.vendor=vendor_module();self.calls=0;self.fallbacks=0;self.fallback=fallback
        self.plan=str(path);self.precision='existing FP8/BF16 Conv roles; FP32 FIR activation/interfaces'
        self.plugins=['inspark_custom::small_fir_activation'] if custom else []
        self.new_gpu_math=custom;self.partition_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.backend=('TensorRT segments + custom AOT/Triton FIR + NVIDIA late activation' if custom
                      else 'TensorRT segments + unchanged NVIDIA BigVGAN CUDA activation')
        self.activation_stream_count=int(m.get('activation_streams',0))
        if self.activation_stream_count not in (0,2,3):
            raise ValueError('Vocoder activation_streams must be 0, 2 or 3')
        self.activation_streams=[torch.cuda.Stream() for _ in range(self.activation_stream_count)]
        for row in self.flow:
            if row['kind']=='trt_segment':
                engine=StaticEngine(row.get('plan') or path.parent/f"segment{row['depth']}_{plan_suffix}.plan.json",self.batch)
                if engine.plan.get('quantization_recipe')!=m['quantization_recipe'] or engine.plan.get('tf32') is not False:
                    raise ValueError('Segment changes precision/calibration/role policy')
                for i in range(engine.engine.num_io_tensors):
                    name=engine.engine.get_tensor_name(i)
                    if engine.engine.get_tensor_location(name)!=engine.trt.TensorLocation.DEVICE:
                        raise ValueError('Partition introduced a host shape tensor')
                self.engines[row['depth']]=engine
            elif row['kind']=='vendor_activation':
                self.parameters[row['name']]=tuple(torch.tensor(v,device='cuda',dtype=torch.float32) for v in row['parameters'])
                stage=(int(row['name'].split('/resblocks.')[1].split('/')[0])//3 if '/resblocks.' in row['name'] else 6)
                if stage in m.get('custom_tiled_stages',[]):
                    from inspark_infer.ops.triton.vocoder_tiled_fir import TiledFIR
                    self.custom_activations[row['name']]=TiledFIR(*self.parameters[row['name']],
                        block=int(m.get('tiled_fir_block',256)),warps=int(m.get('tiled_fir_warps',4)))
                elif stage in m.get('custom_vendor_stages',[]):
                    from inspark_infer.ops.triton.vocoder_small_fir import SmallFIR
                    self.custom_activations[row['name']]=SmallFIR(*self.parameters[row['name']])
            else:raise ValueError('Unknown partition operation')
        # Report embedded TRT plugins separately from direct Triton activation
        # calls, including mixed whole-row/tiled prefix engines.
        self.plugins=sorted({name for engine in self.engines.values() for name in engine.plan.get('plugins',[])})
        if custom and len(self.custom_activations)==len(self.parameters):
            self.backend='TensorRT segments + custom AOT/Triton FIR'

    def __call__(self,mel):
        if tuple(mel.shape)!=(self.batch,80,52) or mel.dtype!=torch.float32 or not mel.is_cuda or not mel.is_contiguous():
            if self.fallback is not None:
                self.fallbacks+=1;return self.fallback(mel)
            raise ValueError(f'Vocoder partition requires contiguous B{self.batch}/F52 FP32 mel')
        values={self.manifest['inputs'][0]:mel}
        index=0
        while index<len(self.flow):
            row=self.flow[index]
            if row['kind']=='vendor_activation':
                group=[]
                while index<len(self.flow) and self.flow[index]['kind']=='vendor_activation':
                    group.append(self.flow[index]);index+=1
                # Only fork mutually independent branches whose complete
                # inputs already exist. Dependent activations remain serial.
                outputs={r['output'] for r in group}
                independent=all(r['input'] in values and r['input'] not in outputs for r in group)
                parallel=bool(self.activation_streams) and len(group)>1 and independent
                main=torch.cuda.current_stream()
                if parallel:
                    for stream in self.activation_streams:stream.wait_stream(main)
                for offset,r in enumerate(group):
                    stream=self.activation_streams[offset%len(self.activation_streams)] if parallel else main
                    with torch.cuda.stream(stream):
                        if r['name'] in self.custom_activations:
                            values[r['output']]=self.custom_activations[r['name']](values[r['input']])
                        else:
                            values[r['output']]=self.vendor.forward(values[r['input']],*self.parameters[r['name']])
                if parallel:
                    for stream in self.activation_streams:main.wait_stream(stream)
            else:
                values.update(self.engines[row['depth']]({name:values[name] for name in row['inputs']}))
                index+=1
        self.calls+=1
        return values[self.manifest['outputs'][0]]

    def route_for_signature(self,mel):
        if tuple(mel.shape)==(self.batch,80,52) and mel.dtype==torch.float32 and mel.is_cuda and mel.is_contiguous():
            return dict(kind='hybrid',backend=self.backend,
                        component='vocoder',batch=self.batch,frames=52,plan=self.plan,sha256=self.partition_sha256,
                        plugins=list(self.plugins),fallback=False,trt_enqueues=len(self.engines),vendor_enqueues=len(self.parameters)-len(self.custom_activations),
                        custom_activation_enqueues=len(self.custom_activations),new_gpu_math=self.new_gpu_math)
        if self.fallback is None:raise ValueError('No fallback for unsupported partition shape')
        return self.fallback.route_for_signature(mel)

    def record_replay(self,count=1):self.calls+=count

    def stats(self):
        return dict(backend=self.backend,
                    trt_enqueues_per_call=len(self.engines),vendor_enqueues_per_call=len(self.parameters)-len(self.custom_activations),
                    custom_activation_enqueues=len(self.custom_activations),
                    activation_streams=self.activation_stream_count,
                    new_gpu_math=self.new_gpu_math,plugins=list(self.plugins),calls=self.calls,partition=str(self.partition_path))
