"""Reuse one exact B64 vocoder for B128, snapshotting its writable output."""
from inspark_infer.runtime.bundle_paths import read_json
import json
from pathlib import Path
import torch
from inspark_infer.ops.tensorrt.official_vocoder_segments import OfficialVocoderSegments


class VocoderSerialMicrobatch:
    def __init__(self,partition,fallback):
        candidate=read_json(Path(partition));original=fallback.manifest
        if original['batch']!=128 or candidate['batch']!=64 or original['frames']!=52 or candidate['frames']!=52:
            raise ValueError('Vocoder microbatch supports exact B128/F52 to B64/F52 only')
        if original['quantization_recipe']!=candidate['quantization_recipe']:
            raise ValueError('Vocoder microbatch changes quantization recipe')
        if original['checkpoint']['sha256']!=candidate.get('checkpoint',{}).get('sha256'):
            raise ValueError('Vocoder microbatch changes checkpoint')
        self.chunk=OfficialVocoderSegments(partition);self.fallback=fallback
        self.batch=128;self.frames=52
        self.output=torch.empty(128,1,52*256,device='cuda',dtype=torch.float32)
        self.calls=self.fallbacks=0;self.plan=str(Path(partition).resolve())

    def __call__(self,mel):
        self.calls+=1
        if tuple(mel.shape)!=(128,80,52) or mel.dtype!=torch.float32 or not mel.is_cuda or not mel.is_contiguous():
            self.fallbacks+=1;return self.fallback(mel)
        for begin in (0,64):
            value=self.chunk(mel[begin:begin+64])
            if tuple(value.shape)!=(64,1,52*256) or value.dtype!=torch.float32:
                raise ValueError('B64 vocoder output contract changed')
            # Snapshot before the next enqueue reuses value's backing store.
            self.output[begin:begin+64].copy_(value)
        return self.output

    def route_for_signature(self,mel):
        if tuple(mel.shape)==(128,80,52) and mel.dtype==torch.float32 and mel.is_cuda and mel.is_contiguous():
            route=self.chunk.route_for_signature(mel[:64])
            route.update(batch=128,engine_batch=64,microbatches=2,execution='serial B64 vocoder context reused twice')
            for field in ('trt_enqueues','vendor_enqueues','custom_activation_enqueues'):
                if field in route:route[field]*=2
            return route
        return self.fallback.route_for_signature(mel)

    def record_replay(self,count=1):self.calls+=count

    def stats(self):
        return dict(backend='TensorRT B64 vocoder serial context reuse for B128',batch=128,engine_batch=64,
                    microbatches=2,calls=self.calls,fallbacks=self.fallbacks,partition=self.plan,chunk=self.chunk.stats())
