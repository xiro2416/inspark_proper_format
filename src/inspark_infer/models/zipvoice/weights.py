"""Strict loaders for retained original Eager and SmoothQuant INT8 weights."""
import json
from pathlib import Path

def load_model(directory, route='int8', device='cpu'):
    if route not in ('eager','int8'):
        raise ValueError('Supported weight routes are eager and int8')
    from safetensors.torch import load_file
    from .reference.models.zipvoice_distill import ZipVoiceDistill
    from .tokenizer import EmiliaTokenizer
    from .packed import PackedWeighted
    directory=Path(directory)
    config=json.loads((directory/'config/model.json').read_text())['model']
    tokenizer=EmiliaTokenizer(directory/'config/tokens.txt')
    model=ZipVoiceDistill(**config,vocab_size=tokenizer.vocab_size,pad_id=tokenizer.pad_id)
    if route=='int8':
        recipe=json.loads((directory/'int8/quantization.json').read_text())
        assert len(recipe['modules'])==180
        for name in recipe['modules']:
            parent,attr=name.rsplit('.',1)
            setattr(model.get_submodule(parent),attr,PackedWeighted(model.get_submodule(name),prototype=True))
    # Prototype allocation is overwritten completely; this performs no calibration.
    model.load_state_dict(load_file(str(directory/route/'model.safetensors')),strict=True)
    return model.to(device).eval(),tokenizer
