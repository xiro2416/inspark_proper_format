"""Component-specific calibration and semantic checkpoint identities."""
import hashlib,json
from pathlib import Path

def calibration_path(plan,component):
    return plan.get('component_calibrations',{}).get(component,plan['calibration'])

def calibration_digest(plan,component):
    return hashlib.sha256(Path(calibration_path(plan,component)).read_bytes()).hexdigest()

def tensor_digest(state):
    import torch
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise ValueError('Checkpoint identity expects tensors only')
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([key,list(value.shape),str(value.dtype)],separators=(',',':')).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def checkpoint_states(config):
    import torch
    from safetensors.torch import load_file
    root=Path(config['weights'])
    return {
        'target':lambda:torch.load(root/'index_tts2/gpt.pth',map_location='cpu',weights_only=True),
        'draft':lambda:load_file(str(root/'draft_onpolicy100/model.safetensors')),
        'cfm':lambda:torch.load(config['student'],map_location='cpu',weights_only=True)['student'],
        'vocoder':lambda:torch.load(root/'index_tts2/hf_cache/bigvgan/bigvgan_generator.pt',map_location='cpu',weights_only=True)['generator'],
    }

def validate_model_identities(engine,plan):
    if getattr(engine,'torch',None) is not None and engine.torch.cuda.get_device_name()!='NVIDIA RTX 6000D':
        raise ValueError('Published optimized engines require NVIDIA RTX 6000D')
    expected=plan.get('model_tensor_hashes')
    if expected is None:return # Historical inputs remain usable for local migration diagnostics.
    if set(expected)!={'target','draft','cfm','vocoder'}:raise ValueError('Incomplete release model identities')
    for component,load in checkpoint_states(engine.config).items():
        if tensor_digest(load())!=expected[component]:
            raise ValueError(f'{component} checkpoint does not match this engine release')

def validate_component_roles(plan):
    composite=json.loads(Path(plan['calibration']).read_text())
    for component,path in plan.get('component_calibrations',{}).items():
        source=json.loads(Path(path).read_text())
        old={k:v for k,v in source['role_specs'].items() if k.startswith(component+'.')}
        new={k:v for k,v in composite['role_specs'].items() if k.startswith(component+'.')}
        if source['scheme']!=composite['scheme'] or not old or old!=new:
            raise ValueError(f'{component} reused calibration differs from the composite recipe')
