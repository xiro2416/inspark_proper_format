"""Reuse one complete native B64 vocoder twice without sharing request state."""
from pathlib import Path
import torch
from inspark_infer.runtime.bundle_paths import read_json
from inspark_infer.ops.tensorrt.native113 import NativeVocoder113

class NativeVocoderSerialMicrobatch:
    def __init__(self,plan_path,fallback):
        original=read_json(Path(fallback.plan));candidate=read_json(Path(plan_path))
        if original['batch']!=128 or candidate['batch']!=64:
            raise ValueError('Native vocoder microbatch requires B128 and B64 plans')
        for key in ('kind','frames','trt','precision','plugins'):
            if candidate.get(key)!=original.get(key):raise ValueError('Microbatch changes '+key)
        a,b=original['quantization_recipe'],candidate['quantization_recipe']
        for key in ('scheme','role_manifest','role_specs_sha256'):
            if a.get(key)!=b.get(key):raise ValueError('Microbatch changes quantization '+key)
        if a['calibration']['sha256']!=b['calibration']['sha256']:
            raise ValueError('Microbatch changes calibration content')
        sources=lambda p:sorted((x['role'],x['sha256']) for x in p['provenance']['model_sources'])
        if sources(original)!=sources(candidate):raise ValueError('Microbatch changes checkpoint sources')
        self.fallback=fallback;self.chunk=NativeVocoder113(plan_path,fallback.eager)
        self.output=torch.empty_like(fallback.output);self.calls=self.fallbacks=0
        self.new_gpu_math=getattr(self.chunk,'new_gpu_math',False)
        self.plan=str(Path(plan_path).resolve())
    def describe_route(self,mel):
        route=dict(self.fallback.describe_route(mel))
        if route['kind']=='tensorrt':
            route.update(plan=self.chunk.plan,sha256=self.chunk.engine_sha256,engine_batch=64,
                         microbatches=2,execution='serial B64 complete native vocoder reused twice',
                         plan_sha256=self.chunk.provenance['plan_sha256'])
        return route
    route_for_signature=describe_route
    def __call__(self,mel):
        self.calls+=1
        if self.describe_route(mel)['kind']!='tensorrt':
            self.fallbacks+=1;return self.fallback(mel)
        for begin in (0,64):
            # Snapshot the writable context output before the next enqueue.
            self.output[begin:begin+64].copy_(self.chunk(mel[begin:begin+64]))
        return self.output
    def stats(self):
        return dict(backend='TensorRT native vocoder serial B64x2',batch=128,engine_batch=64,
                    microbatches=2,calls=self.calls,fallbacks=self.fallbacks,plan=self.plan,chunk=self.chunk.stats())
