"""Offline-selected full-M projections for fixed B8 Draft/Target graphs."""
import ast
import hashlib
import json
import os
from pathlib import Path

import torch

from inspark_infer.ops.triton.target_full_m.tiled import run as full_m_linear
from inspark_infer.ops.triton.target_seven.projections import base_linear
from inspark_infer.ops.triton.target_norm_quant.prequantized import binding


def identity():
    root = Path(__file__).resolve().parents[3]
    files = [Path(__file__), root / 'ops/triton/target_full_m/tiled.py']
    from inspark_infer.ops.triton.target_seven.projections import identity as parent_identity
    value = parent_identity()
    value['full_m_sources'] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    return value


class FullMLinear(torch.nn.Module):
    def __init__(self, old, choices):
        super().__init__()
        self.old = old
        self.choices = {int(k): dict(v) for k, v in choices.items()}
        raw = base_linear(old)
        self.in_features = raw.in_features
        self.out_features = raw.out_features
        self.precision = raw.precision
        col, _ = binding(old, next(iter(self.choices)))
        if col is None:
            col = raw.weight[:, :raw.scales.numel()].t().contiguous()
        self.register_buffer('weight_col', col)

    @property
    def weight(self):
        return self.old.weight

    @property
    def bias(self):
        return self.old.bias

    def forward(self, x):
        m = x.numel() // self.in_features
        plan = self.choices.get(m)
        if plan is None or x.dtype != torch.float32:
            return self.old(x)
        raw = base_linear(self.old)
        return full_m_linear(x, self.weight_col, raw.scales, raw.bias, plan)


def install(engine, data, install_combined=True):
    """Install before Target/Draft CUDA Graph capture; B8 shapes only."""
    if engine.sessions or engine.head_graphs is not None or getattr(engine.rt.target, 'graph_sealed', False):
        raise RuntimeError('Full-M projections must be installed before graph capture/admission')
    changed = []

    def walk(root, component, choices, prefix):
        for name, module in list(root.named_children()):
            label = f'{prefix}.{name}'
            raw = base_linear(module)
            if getattr(raw, 'precision', None) == 'fp8':
                selected = {}
                for key, candidate in choices.items():
                    m, n, k, has_bias = ast.literal_eval(key)
                    if (n, k, has_bias) == (raw.out_features, raw.in_features,
                                            raw.bias is not None):
                        selected[m] = candidate
                if selected:
                    root.add_module(name, FullMLinear(module, selected))
                    changed.append(label)
                    continue
            walk(module, component, choices, label)

    walk(engine.tts.gpt.gpt.h, 'target', data.get('target', {}), 'target')
    walk(engine.rt.engine.draft.layers, 'draft', data.get('draft', {}), 'draft')
    if install_combined and data.get('draft_combined_qkv'):
        if not hasattr(engine.rt.backbone, 'shared_qkv'):
            raise RuntimeError('Combined Draft QKV requires draft_qkv_fusion preparation')
        for index, fused in engine.rt.backbone.shared_qkv.items():
            fused.full_m_b8_plan = dict(data['draft_combined_qkv'])
            changed.append(f'draft.{index}.combined_qkv')
    engine._full_m_prepared = True
    return dict(changed=changed, b8_only=True, weight_multiplicity=1,
                unknown_shapes='previous_deployment', online_tuning=False)


def _load(path):
    data = json.loads(Path(path).read_text())
    if data.get('identity') != identity():
        raise ValueError('Full-M plan source/device mismatch')
    if not data.get('full_validation_passed', False) and os.environ.get('ACC_FULL_M_EXPERIMENT') != '1':
        raise ValueError('Full-M plan has not passed complete graph/first-packet validation')
    return data


def prepare(engine, path):
    return install(engine, _load(path), install_combined=False)


def prepare_combined(engine, path):
    data = _load(path)
    if not data.get('draft_combined_qkv'):
        return dict(changed=[])
    if not hasattr(engine.rt.backbone, 'shared_qkv'):
        raise RuntimeError('Draft shared QKV has not been prepared')
    changed=[]
    for index, fused in engine.rt.backbone.shared_qkv.items():
        fused.full_m_b8_plan=dict(data['draft_combined_qkv'])
        changed.append(f'draft.{index}.combined_qkv')
    return dict(changed=changed,b8_only=True,weight_multiplicity=1,
                online_tuning=False)
