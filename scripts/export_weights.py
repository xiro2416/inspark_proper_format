import json, re, hashlib, shutil
from pathlib import Path
from collections import Counter
import torch
from safetensors.torch import save_file, load_file

OUT=None
torch.set_num_threads(8)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def tensor_state(state):
    assert all(isinstance(v,torch.Tensor) for v in state.values())
    return {k:v.detach().cpu().contiguous().clone() for k,v in state.items()}

def fold(state):
    result=dict(state)
    for key in list(state):
        if key.endswith('.weight_g'):
            prefix=key[:-len('weight_g')]
            g,v=state[key].float(),state[prefix+'weight_v'].float()
            result[prefix+'weight']=torch._weight_norm(v,g,0).contiguous()
            del result[key];del result[prefix+'weight_v']
    return result

def key_for(role,component):
    if component=='target':return role.replace('target.blocks.','gpt.h.',1)+'.weight'
    if component=='draft':return role.replace('draft.blocks.','layers.',1)+'.weight'
    if component=='cfm':return role.replace('cfm.blocks.','transformer.layers.',1).replace('cfm.wavenet.','wavenet.',1)+'.weight'
    m=re.match(r'vocoder.stages\.(\d+)\.resblock(\d+)\.(.*)',role)
    if m:return f'resblocks.{int(m[1])*3+int(m[2])}.{m[3]}.weight'
    m=re.match(r'vocoder.stages\.(\d+)\.ups\.(\d+)$',role)
    if m:return f'ups.{m[1]}.{m[2]}.weight'
    return re.sub(r'vocoder.stages\.\d+\.','',role).replace('vocoder.','',1)+'.weight'

def emit(component,raw,recipes):
    raw=tensor_state(raw)
    dst=OUT/'unquantized'/f'{component}.safetensors';dst.parent.mkdir(parents=True,exist_ok=True)
    save_file(raw,str(dst))
    base=fold(raw)
    for variant,recipe in recipes.items():
        state=dict(base); roles={}; count=Counter()
        for role,spec in recipe['role_specs'].items():
            if not role.startswith(component+'.'):continue
            key=key_for(role,component)
            if key not in state:raise KeyError((role,key))
            w=base[key].float()
            transpose=component=='target'
            if transpose:w=w.t().contiguous()
            p=spec['precision'];count[p]+=1
            meta=dict(spec,state_key=key,stored_layout='canonical_weight',transpose_from_checkpoint=transpose)
            if p=='bf16':q=w.bfloat16()
            else:
                if p=='int8':
                    smooth=torch.tensor(spec['smooth_scale'],dtype=torch.float32)
                    shape=[1]*w.ndim;shape[spec['weight_input_axis']]=smooth.numel()
                    w=w*smooth.reshape(shape)
                    state[key+'.smooth_scale']=smooth
                scale=torch.tensor(spec['weight_scale'],dtype=torch.float32)
                shape=[1]*w.ndim;shape[spec['weight_axis']]=scale.numel()
                normalized=w/scale.reshape(shape)
                q=normalized.round().clamp(-128,127).to(torch.int8) if p=='int8' else normalized.clamp(-448,448).to(torch.float8_e4m3fn)
                state[key+'.weight_scale']=scale
                state[key+'.input_scale']=torch.tensor(spec['input_scale'],dtype=torch.float32)
            state[key]=q.contiguous();roles[role]=meta
        path=OUT/variant/f'{component}.safetensors';path.parent.mkdir(parents=True,exist_ok=True)
        save_file({k:v.contiguous().clone() for k,v in state.items()},str(path))
        restored=load_file(str(path))
        assert set(restored)==set(state)
        for k,v in state.items():
            r=restored[k];assert r.shape==v.shape and r.dtype==v.dtype
            assert torch.equal(r.reshape(-1).view(torch.uint8),v.contiguous().reshape(-1).view(torch.uint8)),k
        write(path.with_suffix('.json'),dict(component=component,scheme=recipe['scheme'],roles=roles,
             weight_norm_folded=True,non_role_parameters='original dtype and values',role_counts=dict(count)))
        print(variant,component,dict(count),path.stat().st_size,flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser(description='Export actual mixed FP8/INT8 and original-dtype component checkpoints')
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--student',type=Path,required=True,help='Inference-only format1 student checkpoint')
    p.add_argument('--calibration-dir',type=Path,required=True,help='Full fp8.json and int8_smoothquant.json role manifests')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    global OUT
    OUT=args.output_dir.resolve();models=args.models.resolve()
    recipes={v:json.loads((args.calibration_dir/f'{s}.json').read_text()) for v,s in [('fp8','fp8'),('int8','int8_smoothquant')]}
    emit('target',torch.load(models/'index_tts2/gpt.pth',map_location='cpu',weights_only=True),recipes)
    emit('draft',load_file(str(models/'draft_onpolicy100/model.safetensors')),recipes)
    emit('cfm',torch.load(args.student,map_location='cpu',weights_only=True)['student'],recipes)
    emit('vocoder',torch.load(models/'index_tts2/hf_cache/bigvgan/bigvgan_generator.pt',map_location='cpu',weights_only=True)['generator'],recipes)
    for variant,recipe in recipes.items():write(OUT/variant/'quantization_config.json',recipe)

if __name__=='__main__':main()
