"""Resolve portable asset paths against an explicitly marked release bundle."""
import json
import os
from pathlib import Path

def read_json(path):
    path=Path(path).resolve()
    value=json.loads(path.read_text())
    root=next((p for p in [path.parent,*path.parents] if (p/'.bundle_root').is_file()),None)
    if root is None and os.environ.get('INSPARK_ASSET_ROOT'):
        candidate=Path(os.environ['INSPARK_ASSET_ROOT']).resolve()
        if (candidate/'.bundle_root').is_file():root=candidate
    def resolve(item):
        if isinstance(item,str) and item.startswith('bundle://'):
            if root is None:raise ValueError('Portable asset path requires a marked bundle root')
            target=(root/item[len('bundle://'):]).resolve()
            if not target.is_relative_to(root):raise ValueError('Asset path escapes the bundle')
            return str(target)
        if isinstance(item,str) and item.startswith('model://'):
            model_root=os.environ.get('INSPARK_MODEL_ROOT')
            if not model_root:raise ValueError('Model source paths require INSPARK_MODEL_ROOT')
            base=Path(model_root).resolve()
            # Model files may deliberately link to an independent immutable archive.
            # Check lexical traversal before following those user-owned data links.
            target=Path(os.path.abspath(base/item[len('model://'):]))
            if not target.is_relative_to(base):raise ValueError('Model path escapes model root')
            return str(target)
        if isinstance(item,list):return [resolve(v) for v in item]
        if isinstance(item,dict):return {k:resolve(v) for k,v in item.items()}
        return item
    return resolve(value)
