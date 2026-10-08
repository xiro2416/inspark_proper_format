"""Calibrate 128 real bilingual requests once, freeze, and export standard FP8 Q/DQ."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspark_infer.runtime.zipvoice_fp8.common import SOURCE_REVISION, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise RuntimeError('Select exactly --gpu before importing Torch')
    from inspark_infer.runtime.device import GPULease
    with GPULease(args.gpu):
        run(args)


def run(args):
    import torch
    import onnx
    from safetensors.torch import load_file, save_file
    from inspark_infer.models.zipvoice.weights import load_model
    from inspark_infer.models.zipvoice.packed import export_positions
    from inspark_infer.runtime.zipvoice_fp8.quantization import coverage, apply_recipe, translations,original_activation
    from inspark_infer.models.zipvoice.reference.models.modules.scaling import ActivationDropoutAndLinear
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT / 'models/zipvoice/fp8'
    folder.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(ROOT/'models/zipvoice', 'eager', 'cuda')
    cov = coverage(model)
    recipe_path = folder/'quantization.json'
    if not recipe_path.is_file():
        if args.export_only:
            raise FileNotFoundError('Frozen calibration required')
        manifest_path = ROOT/'outputs/fp8/data/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        cases = sorted(manifest['calibration'], key=lambda x:x['total_frames'])
        assert len(cases)==128 and {x['language'] for x in cases}=={'en','zh'}
        maxima = {name:torch.zeros((),device='cuda') for name in cov['quantized_modules']}
        hooks = []
        for name in cov['quantized_modules']:
            def observe(module, inputs, name=name):
                value=original_activation(module,inputs[0]) if isinstance(module,ActivationDropoutAndLinear) else inputs[0]
                maxima[name].copy_(torch.maximum(maxima[name], value.detach().abs().amax()))
            hooks.append(model.get_submodule(name).register_forward_pre_hook(observe))
        grid = torch.linspace(0,1,9,device='cuda');grid=.5*grid/(1-.5*grid)
        with torch.inference_mode():
            for start in range(0,len(cases),16):
                wave = cases[start:start+16]
                data = [load_file(c['condition']) for c in wave]
                tokens = [x['token_ids'][0,:-1].tolist() for x in data]
                lens = torch.tensor([c['total_frames'] for c in wave],dtype=torch.int64,device='cuda')
                frames = int(lens.max());batch=len(wave)
                text, mask = model.forward_text_train(tokens,lens)
                speech = torch.stack([torch.nn.functional.pad(x['prompt_mel'][0]*.1,
                                     (0,0,0,frames-375)) for x in data]).cuda()
                noise = torch.randn(batch,frames,100,device='cuda',
                                    generator=torch.Generator(device='cuda').manual_seed(9100+start))
                guide = torch.ones(batch,1,1,device='cuda')
                for k in range(8):
                    velocity=model.forward_fm_decoder(grid[k].expand(batch).reshape(batch,1,1),
                                                    noise,text,speech,mask,guide)
                    noise.add_(velocity.float()*(grid[k+1]-grid[k]))
                if not torch.isfinite(noise).all():
                    raise RuntimeError('Nonfinite semantic calibration trajectory')
                print(json.dumps({'event':'calibration_wave','start':start,'batch':batch,
                                  'frames':frames,'steps':8}),flush=True)
        for hook in hooks:hook.remove()
        modules={}
        for name in cov['quantized_modules']:
            input_max=float(maxima[name].cpu());weight_max=float(model.get_submodule(name).weight.detach().abs().amax().cpu())
            if not input_max>0 or not weight_max>0:
                raise ValueError('Degenerate calibration: '+name)
            modules[name]=dict(input_scale=input_max/448,weight_scale=weight_max/448,
                               input_amax=input_max,weight_amax=weight_max)
        recipe=dict(schema=1,precision='fp8',format='E4M3FN',method='static_max',
                    weight_axis=None,activation_axis=None,calibration_requests=128,
                    calibration_manifest_sha256=sha(manifest_path),source_revision=SOURCE_REVISION,
                    eager_weights_sha256=sha(ROOT/'models/zipvoice/eager/model.safetensors'),
                    first_floating_layers=4,last_fp8_layers=12,steps=8,modules=modules,
                    coverage=cov,calibration_arithmetic='FP32 original model trajectories; TF32 disabled')
        write(recipe_path,recipe)
    recipe=json.loads(recipe_path.read_text())
    assert recipe['eager_weights_sha256']==sha(ROOT/'models/zipvoice/eager/model.safetensors')
    model=apply_recipe(export_positions(model),recipe).cpu().eval()
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(folder/'model.safetensors'))
    class FM(torch.nn.Module):
        def __init__(self,m):super().__init__();self.model=m
        def forward(self,t,x,text_condition,speech_condition,padding_mask,guidance_scale):
            return self.model.forward_fm_decoder(t,x,text_condition,speech_condition,padding_mask,guidance_scale)
    wrapped=FM(model).eval()
    names=['t','x','text_condition','speech_condition','padding_mask','guidance_scale']
    b,t=2,760
    inputs=(torch.full((b,1,1),.25),torch.randn(b,t,100),torch.randn(b,t,100),
            torch.randn(b,t,100),torch.zeros(b,t,dtype=torch.bool),torch.ones(b,1,1))
    dynamic={name:{0:'batch'} for name in [*names,'velocity']}
    for name in ('x','text_condition','speech_condition','padding_mask','velocity'):dynamic[name][1]='frames'
    onnx_path=folder/'fm.onnx'
    report=dict(status='export_started',recipe_sha256=sha(recipe_path),coverage=cov,
                exporter='pending',torch=torch.__version__)
    # Probe modern export. Retain only explicit FP8 Q/DQ and dynamic B/T IO;
    # custom autograd Q/DQ uses the established symbolic compatibility path
    # when Dynamo cannot preserve that contract or dynamic model constraints.
    try:
        modern=folder/'fm-modern-probe.onnx'
        with torch.inference_mode():
            torch.onnx.export(wrapped,inputs,str(modern),dynamo=True,opset_version=21,
                              input_names=names,output_names=['velocity'],dynamic_axes=dynamic,
                              custom_translation_table=translations(),optimize=False)
        graph=onnx.load(modern)
        if not any(n.op_type=='QuantizeLinear' for n in graph.graph.node):
            raise RuntimeError('Modern autograd export did not retain explicit FP8 Q/DQ')
        if not all(graph.graph.input[i].type.tensor_type.shape.dim[0].dim_param for i in range(6)):
            raise RuntimeError('Modern export specialized batch shape')
        onnx.save_model(graph,str(onnx_path),save_as_external_data=True,
                       all_tensors_to_one_file=True,location='fm.onnx.data',size_threshold=1024)
        report['exporter']='modern Dynamo explicit QDQ'
    except Exception as e:
        report['modern_export_error']=str(e)[:6000]
        (folder/'modern-export-error.txt').write_text(traceback.format_exc())
        with torch.inference_mode():
            torch.onnx.export(wrapped,inputs,str(onnx_path),dynamo=False,opset_version=21,
                input_names=names,output_names=['velocity'],dynamic_axes=dynamic,
                external_data=True,do_constant_folding=False)
        report['exporter']='legacy custom-autograd explicit E4M3 QDQ compatibility export'
    graph=onnx.load(onnx_path)
    from inspark_infer.runtime.zipvoice_fp8.graph import lower_sequences,fold_shape_constants
    report['trt_sequence_lowering']=lower_sequences(graph)
    graph=fold_shape_constants(graph)
    # Consolidate external tensors for relocation and reproducible identity.
    external=folder/'fm-final.data'
    if external.exists():external.unlink()
    onnx.save_model(graph,str(onnx_path),save_as_external_data=True,
                   all_tensors_to_one_file=True,location=external.name,size_threshold=1024)
    onnx.checker.check_model(str(onnx_path))
    fp8=[v for v in graph.graph.initializer if v.data_type==onnx.TensorProto.FLOAT8E4M3FN
         and v.name.endswith('weight_fp8')]
    q=sum(n.op_type=='QuantizeLinear' for n in graph.graph.node)
    if len(fp8)!=len(recipe['modules']) or q!=len(recipe['modules']):
        raise RuntimeError(f'FP8 graph coverage mismatch: weights={len(fp8)}, activations={q}')
    report.update(status='exported_unvalidated',graph_sha256=sha(onnx_path),
                  weights_sha256=sha(folder/'model.safetensors'),fp8_weights=len(fp8),
                  activation_quantizers=q,external_sha256=sha(external))
    write(ROOT/'reports/sm120/zipvoice/fp8/history/002-calibration-export.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
