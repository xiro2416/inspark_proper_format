"""Shared identity, workload and single-GPU policy for the FP8 release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BATCHES = (1, 2, 4, 8, 16, 32, 64)
SOURCE_REVISION = 'f6da21d25400d1b3e0b9a70de333503b3374df76'


def root():
    return Path(os.getenv('INSPARK_REPO_ROOT', Path(__file__).resolve().parents[4])).resolve()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def private_report(batch, name):
    """Raw evaluation text/audio identities remain outside Git and publication."""
    return root()/f'outputs/fp8/b{batch}/reports/{name}.json'


def environment(gpu=3):
    r = root()
    env = dict(os.environ)
    env.update(INSPARK_REPO_ROOT=str(r), PYTHONPATH=str(r / 'src'),
               CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_DISABLE_XET='1')
    for name, directory in {'HF_HOME': '.cache/huggingface', 'UV_CACHE_DIR': '.cache/uv',
                            'XDG_CACHE_HOME': '.cache/fp8', 'TRITON_CACHE_DIR': '.cache/fp8/triton',
                            'CUDA_CACHE_PATH': '.cache/fp8/cuda', 'TORCH_HOME': '.cache/fp8/torch',
                            'TMPDIR': '.cache/tmp'}.items():
        p = r / directory
        p.mkdir(parents=True, exist_ok=True)
        env[name] = str(p)
    return env


def profiles(batch):
    fm, text, vocos = {}, {}, {}
    for label, frames, tokens in zip(('min', 'opt', 'max'), (600, 760, 920), (52, 78, 141)):
        for name in ('x', 'text_condition', 'speech_condition'):
            fm.setdefault(name, {})[label] = [batch, frames, 100]
        for name in ('t', 'guidance_scale'):
            fm.setdefault(name, {})[label] = [batch, 1, 1]
        fm.setdefault('padding_mask', {})[label] = [batch, frames]
        for name, shape in {'token_ids': [batch, tokens], 'token_lens': [batch],
                            'features_lens': [batch], 'frame_positions': [1, frames]}.items():
            text.setdefault(name, {})[label] = shape
        vocos.setdefault('mel', {})[label] = [batch, 100, frames - 375]
    return {'fm': fm, 'text': text, 'unique': profiles(1)['text'] if batch != 1 else text,
            'vocos': vocos}


def workload(batch, frames, tokens):
    return dict(batch=batch, prompt_frames=375, target_frames=frames-375,
                total_frames=frames, joint_tokens=tokens-1, padded_tokens=tokens,
                steps=8, t_shift=.5, guidance=1., feat_scale=.1)


def profile_manifest(batch):
    return {'batch': batch, 'engines': {k: {'shape_profile': {
        n: [v[x] for x in ('min', 'opt', 'max')] for n, v in p.items()}}
        for k, p in profiles(batch).items()}}
