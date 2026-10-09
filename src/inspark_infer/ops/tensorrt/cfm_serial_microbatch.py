"""Replay one exact B64 four-step CFM twice for a B128 head.

Each request keeps its complete time axis. Output is copied before reusing the
same context's writable buffer; no cross-request state or RNG is introduced.
"""
from inspark_infer.runtime.bundle_paths import read_json
import json
from pathlib import Path

import torch

from inspark_infer.ops.tensorrt.native113 import NativeCFMSolver113


class CFMSerialMicrobatch:
    def __init__(self, plan_path, fallback):
        original = read_json(Path(fallback.plan))
        candidate = read_json(Path(plan_path))
        if original['batch'] != 128 or candidate['batch'] != 64:
            raise ValueError('CFM serial microbatch supports exact B128 to B64 only')
        for key in ('kind', 'frames', 'prompt_frames', 'trt', 'precision'):
            if candidate.get(key) != original.get(key):
                raise ValueError('Microbatch CFM changes ' + key)
        if candidate['kind'] != 'full_solver':
            raise ValueError('Microbatch requires complete four-step solver')
        a, b = original['quantization_recipe'], candidate['quantization_recipe']
        for key in ('scheme', 'role_manifest','role_specs_sha256'):
            if a[key] != b[key]:
                raise ValueError('Microbatch CFM changes quantization ' + key)
        if a['calibration']['sha256']!=b['calibration']['sha256']:
            raise ValueError('Microbatch CFM changes calibration content')
        sources = lambda plan: sorted((v['role'], v['sha256']) for v in plan['provenance']['model_sources'])
        if sources(original) != sources(candidate):
            raise ValueError('Microbatch CFM changes source checkpoints')
        self.fallback = fallback
        self.chunk = NativeCFMSolver113(plan_path, fallback.eager)
        self.eager = fallback.eager
        self.model, self.times = fallback.model, fallback.times
        self.identity = dict(fallback.identity, execution='serial B64 context reused twice',
                             microbatch_plan=self.chunk.plan, microbatch_sha256=self.chunk.engine_sha256)
        self.output = torch.empty_like(fallback.output)
        self.calls = self.fallbacks = 0

    def describe_route(self, *inputs):
        route = dict(self.fallback.describe_route(*inputs))
        if route['kind'] == 'tensorrt':
            route.update(plan=self.chunk.plan, sha256=self.chunk.engine_sha256,
                         engine_batch=64, batch=128, microbatches=2,
                         solver_kind='two_serial_full_solver_enqueues',
                         plan_sha256=self.chunk.provenance['plan_sha256'],
                         optimization_level=self.chunk.optimization_level,
                         tiling_optimization_level=self.chunk.tiling_optimization_level,
                         plugins=list(self.chunk.plugins),custom_math_kernels=bool(self.chunk.plugins))
        return route

    route_for_signature = describe_route

    def __call__(self, *inputs):
        self.calls += 1
        if self.describe_route(*inputs)['kind'] != 'tensorrt':
            self.fallbacks += 1
            return self.fallback(*inputs)
        for begin in (0, 64):
            values = tuple(v[begin:begin + 64] for v in inputs)
            self.output[begin:begin + 64].copy_(self.chunk(*values))
        return self.output

    def stats(self):
        return dict(backend='TensorRT 11.3 native serial microbatch four-step solver',
                    batch=128, engine_batch=64, microbatches=2, calls=self.calls,
                    fallbacks=self.fallbacks, identity=dict(self.identity),
                    chunk=self.chunk.stats())
