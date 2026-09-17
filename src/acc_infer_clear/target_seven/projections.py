"""Offline-selected FP8 projection pipelines; original modules remain fallbacks.

No autotuning, allocation of transformed weights, or graph capture on forward.
Selection is per projection role and M/N/K, never a global precision change.
"""
import hashlib
import json
from pathlib import Path
import torch
from acc_infer_clear.ar_pipeline.gemm import run
from acc_infer_clear.ar_pipeline.deploy import identity as parent_identity
from acc_infer_clear.kernels.stage2_gemm import col_linear
from acc_infer_clear.kernels.planner import Tile

ROLES = {'attn.c_attn': 'qkv', 'attn.c_proj': 'out',
         'mlp.c_fc': 'up', 'mlp.c_proj': 'down'}

def identity():
    value = parent_identity()
    value['seven_projection_source'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return value

def base_linear(module):
    while hasattr(module, 'old'):
        module = module.old
    return module

def signature(role, m, n, k, precision='fp8'):
    return f'{role}:{precision}:{m}:{n}:{k}'

def candidates(precision='fp8'):
    if precision == 'bf16':
        # Same values, different physical layout may select a different cuBLAS path.
        yield dict(kind='bf16_column_storage')
        # Counterfactual only: separate bias rounds the GEMM result first and may
        # differ from fused addmm. Bitwise gate must reject any such difference.
        yield dict(kind='bf16_mm_bias')
        return
    # Shared bytes ~= stages * BK * (BM + BN). Keep K order and split_k=1.
    for bn in (32, 64):
        for stages in (3, 4, 6):
            yield dict(kind='explicit', plan=dict(bm=16, bn=bn, bk=128,
                       stages=stages, swizzle=True, inner=False, double=False))
    yield dict(kind='explicit', plan=dict(bm=16, bn=32, bk=128,
               stages=4, swizzle=True, inner=True, double=False))
    for bn in (32, 64):
        for stages in (3, 5):
            yield dict(kind='compiler', tile=dict(bm=16, bn=bn, bk=128,
                       warps=4, stages=stages, split_k=1))
    yield dict(kind='cublas', tile=dict(bm=16, bn=32, bk=128,
               warps=4, stages=3, split_k=1))

class Projection(torch.nn.Module):
    def __init__(self, old, choices):
        super().__init__()
        self.old = old
        self.choices = {int(k): v for k, v in choices.items()}
        raw = base_linear(old)
        if raw.precision not in ('fp8', 'bf16'):
            raise ValueError('Unsupported projection precision')
        self.precision = raw.precision
        self.in_features, self.out_features = raw.in_features, raw.out_features
        weight_col = (raw.weight[:, :raw.scales.numel()].t().contiguous()
                      if self.precision == 'fp8' else raw.weight.t().contiguous().t())
        self.register_buffer('weight_col', weight_col)
        self.register_buffer('ones', torch.ones(1, device=raw.weight.device))

    @property
    def weight(self): return self.old.weight

    @property
    def bias(self): return self.old.bias

    def forward(self, x):
        choice = self.choices.get(x.numel() // self.in_features)
        if choice is None or x.dtype != torch.float32:
            return self.old(x)
        raw = base_linear(self.old)
        if self.precision == 'bf16':
            xb = x.bfloat16()
            if choice['kind'] == 'bf16_column_storage':
                return torch.nn.functional.linear(xb, self.weight_col, raw.bias).to(x.dtype)
            if choice['kind'] == 'bf16_mm_bias':
                y = torch.mm(xb.reshape(-1, self.in_features), raw.weight.t())
                if raw.bias is not None: y = y + raw.bias
                return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)
            raise ValueError('Invalid BF16 choice')
        if choice['kind'] == 'explicit':
            return run(x, self.weight_col, raw.scales, raw.bias, choice['plan'])
        return col_linear(x, self.weight_col, raw.scales, raw.bias,
                          Tile(**choice['tile']),
                          'scaled_mm' if choice['kind'] == 'cublas' else 'triton', self.ones)

def prepare(engine, path):
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target, 'graph_sealed', False):
        raise RuntimeError('Prepare projections before admission/capture')
    if getattr(engine, '_target_seven_projections', False):
        raise RuntimeError('Projection plan already applied')
    data = json.loads(Path(path).read_text())
    if data['identity'] != identity():
        raise ValueError('Projection plan source/device mismatch')
    if not data.get('full_validation_passed', False):
        raise ValueError('Microbenchmark plan requires full validation before deployment')
    return install(engine, data)

def install(engine, data):
    """Internal experiment hook; caller must perform before any graph capture."""
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target, 'graph_sealed', False):
        raise RuntimeError('Install before admission/capture')
    changed = []
    for index, block in enumerate(engine.tts.gpt.gpt.h):
        for path, role in ROLES.items():
            parent_name, name = path.split('.')
            parent = getattr(block, parent_name)
            old = getattr(parent, name)
            raw = base_linear(old)
            if getattr(raw, 'precision', None) not in ('fp8', 'bf16'):
                continue
            choices = {m: data['choices'][signature(role, m, raw.out_features, raw.in_features, raw.precision)]
                       for m in range(8, 65, 8)
                       if signature(role, m, raw.out_features, raw.in_features, raw.precision) in data['choices']}
            if choices:
                setattr(parent, name, Projection(old, choices))
                changed.append(f'target.{index}.{path}')
    engine._target_seven_projections = True
    return dict(changed=changed, unknown_shapes='previous_deployment', online_tuning=False)
