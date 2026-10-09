"""Exact-batch exports of published Index components, with unchanged calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'local_assets/download/engines/rtx6000d_sm120/draft900_cfm800'


class TargetVerification(nn.Module):
    """Q8 verification, BF16 attention/cache, FP32 norms/residual/head outputs."""
    def __init__(self, target):
        super().__init__()
        self.transformer = target.model.transformer
        self.lm_head = target.model.lm_head
        self.selected_ids = tuple(target.target_layer_ids)

    def forward(self, x, mask, *cache):
        hidden, selected, appended = x, [], []
        b, qlen, _ = x.shape
        additive = torch.where(mask, torch.zeros((), dtype=torch.bfloat16, device=x.device),
                               torch.full((), float('-inf'), dtype=torch.bfloat16, device=x.device))
        for i, block in enumerate(self.transformer.h):
            attention = block.attn
            q, k, v = attention.c_attn(block.ln_1(hidden)).split(attention.split_size, -1)
            def heads(value):
                return value.bfloat16().reshape(b, qlen, attention.num_heads, attention.head_dim).transpose(1, 2)
            q, k, v = heads(q), heads(k), heads(v)
            appended.extend((k, v))
            keys, values = torch.cat((cache[2*i], k), 2), torch.cat((cache[2*i+1], v), 2)
            context = F.scaled_dot_product_attention(q * (attention.head_dim ** -0.5), keys, values,
                                                     attn_mask=additive, scale=1.0).float()
            hidden = hidden + attention.c_proj(context.transpose(1, 2).contiguous().reshape_as(x))
            hidden = hidden + block.mlp(block.ln_2(hidden))
            if i in self.selected_ids:
                selected.append(hidden)
        final = self.transformer.ln_f(hidden)
        return (self.lm_head(final), torch.cat(selected, -1), final, *appended)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', choices=['target', 'draft', 'cfm', 'vocoder', 'prefill', 'latent'], required=True)
    p.add_argument('--batch', type=int, choices=[1, 2, 4, 8, 16, 32, 48, 64, 128], required=True)
    p.add_argument('--legacy-export', action='store_true', help='Explicit inherited exporter fallback after a recorded modern failure')
    p.add_argument('--cfm-kind', choices=['full_solver','estimator'], default='full_solver')
    p.add_argument('--vocoder-conv2d', action='store_true')
    p.add_argument('--vocoder-gemm', action='store_true')
    a = p.parse_args()
    if a.batch == 48 and a.component not in ('target', 'draft'):
        p.error('B48 is an internal ready-pipeline AR profile; only Target/Draft are supported')
    if a.vocoder_conv2d and a.vocoder_gemm:p.error('Choose a single equivalent convolution representation')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.quantization.unified import load_artifact, iter_roles
    from inspark_infer.build.unified_acoustic_export import (prepare_acoustic_model, acoustic_specs_from_artifact,
                                                           FourStepCFM, ExportWeightOp, inspect_standard_onnx)
    from inspark_infer.build.unified_draft_export import prepare_draft_roles, DraftBackbone, input_names
    from inspark_infer.build.unified_prefix_export import prepare_target_roles, from_engine, EXTENTS, PREFILL_OUTPUTS
    from trt113_provenance import capture_provenance, capture_onnx_artifact, file_record
    role_component = 'target' if a.component in ('target', 'prefill', 'latent') else a.component
    calibration = BUNDLE / 'artifacts/current_release/calibration' / (
        'int8_smoothquant_target_vocoder.json' if role_component in ('target', 'vocoder') else 'int8_smoothquant.json')
    recipe = load_artifact(calibration, 'int8_smoothquant')
    config_path = ROOT / 'local_assets/runtime/runtime_fp32_b1.yaml'
    cfg = load(config_path); cfg['max_batch'] = a.batch
    component_directory = 'cfm-estimator' if a.component=='cfm' and a.cfm_kind=='estimator' else ('vocoder-gemm' if a.component=='vocoder' and a.vocoder_gemm else 'vocoder-conv2d' if a.component=='vocoder' and a.vocoder_conv2d else a.component)
    output = ROOT / 'artifacts/sm89/int8_smoothquant' / f'b{a.batch}' / component_directory / 'model.onnx'
    output.parent.mkdir(parents=True, exist_ok=True)
    with GPULease(1):
        engine = Engine(cfg)
        try:
            with torch.cuda.stream(engine.model.stream), torch.inference_mode():
                provenance = capture_provenance(role_component, cfg, config_path, model=engine)
                b = a.batch
                zero = lambda *shape: torch.zeros(*shape, device='cuda')
                if a.component in ('target', 'prefill', 'latent'):
                    manifest = prepare_target_roles(engine, recipe)
                    if a.component == 'target':
                        model = TargetVerification(engine.rt.engine.target)
                        inputs = (zero(b,8,1280), torch.ones(b,1,8,88,device='cuda',dtype=torch.bool),
                                  *(torch.zeros(b,20,80,64,device='cuda',dtype=torch.bfloat16) for _ in range(48)))
                        names = ['x','mask',*(f'{plane}_cache_in_{i}' for i in range(24) for plane in ('k','v'))]
                        outputs = ['logits','selected','final',*(f'{plane}_append_{i}' for i in range(24) for plane in ('k','v'))]
                        kind, frames = 'verification', 8
                    else:
                        # Source prefix/latent attention is FP32; preserve its policy.
                        model = from_engine(engine, a.component)
                        frames = EXTENTS[a.component]
                        inputs = (zero(b,frames,1280),torch.ones(b,frames,device='cuda',dtype=torch.int64))
                        names = ['x','keep']; outputs = list(PREFILL_OUTPUTS) if a.component=='prefill' else ['latent']
                        kind = a.component
                elif a.component == 'draft':
                    manifest = prepare_draft_roles(engine, recipe)
                    model = DraftBackbone(engine.rt.engine.draft)
                    inputs = (zero(b,7,1280),torch.ones(b,1,7,87,device='cuda',dtype=torch.bool),
                              *(zero(b,20,80,64) for _ in range(6)))
                    names, outputs, kind, frames = input_names(), ['hidden','base'], 'backbone', 7
                else:
                    model = engine.student.model if a.component=='cfm' else engine.tts.bigvgan
                    bindings = list(iter_roles(engine, recipe['scheme'], (a.component,)))
                    specs, mapping = acoustic_specs_from_artifact(model, bindings, recipe, a.component)
                    if a.component=='vocoder':
                        # The published exporter supports this exact algebraic rewrite;
                        # SM89 TRT cannot build protected BF16 ConvTranspose directly.
                        for name,spec in specs.items():
                            if isinstance(model.get_submodule(name),nn.ConvTranspose1d):
                                spec=dict(spec,conv_transpose_rewrite='zero_insert_conv')
                                specs[name]=spec
                    manifest = prepare_acoustic_model(model, specs, fir_polyphase=a.component=='vocoder', cfm_sdpa_bf16=a.component=='cfm')
                    for role in manifest['roles']:
                        role['canonical_path'] = mapping.get(role['path'])
                    if a.component=='vocoder' and a.vocoder_conv2d:
                        lifted=[]
                        for path,module in model.named_modules():
                            if isinstance(module,ExportWeightOp) and module.precision=='int8':
                                module.conv1d_as_2d=True
                                lifted.append(path)
                        manifest['conv1d_as_2d']=lifted
                    if a.component=='vocoder' and a.vocoder_gemm:
                        matrix=[]
                        for path,module in model.named_modules():
                            if isinstance(module,ExportWeightOp) and module.precision=='int8':
                                module.conv1d_as_gemm=True
                                matrix.append(path)
                        manifest['conv1d_as_gemm']=matrix
                    if a.component == 'cfm':
                        if a.cfm_kind=='full_solver':
                            model = FourStepCFM(model)
                            inputs = (zero(b,80,310),zero(b,80,310),torch.full((b,),310,device='cuda',dtype=torch.int64),
                                      zero(b,192),zero(b,310,512),(torch.arange(310,device='cuda')[None,None]<258).expand(b,1,310).contiguous())
                            names, outputs, kind, frames = ['x','prompt','lengths','style','mu','mask'], ['output'], 'full_solver', 310
                        else:
                            times=torch.tensor([[0.,.25]],device='cuda').expand(b,2).contiguous()
                            inputs=(zero(b,80,310),zero(b,80,310),torch.full((b,),310,device='cuda',dtype=torch.int64),times,zero(b,192),zero(b,310,512))
                            names,outputs,kind,frames=['x','prompt','lengths','times','style','mu'],['velocity'],'estimator',310
                    else:
                        inputs, names, outputs, kind, frames = (zero(b,80,52),), ['mel'], ['pcm'], 'vocoder', 52
                model.eval().requires_grad_(False)
                values = model(*inputs)
                values = values if isinstance(values,tuple) else (values,)
                if not all(bool(torch.isfinite(v).all()) for v in values):
                    raise ValueError('Nonfinite export reference')
                if a.legacy_export:
                    torch.onnx.export(model, inputs, str(output), opset_version=20, dynamo=False,
                                      external_data=True, input_names=names, output_names=outputs, do_constant_folding=True)
                else:
                    from inspark_infer.build.modern_qdq_export import translations
                    for module in model.modules():
                        if isinstance(module,ExportWeightOp): module.modern_export = True
                    torch.onnx.export(model, inputs, str(output), opset_version=20, dynamo=True,
                                      external_data=True, input_names=names, output_names=outputs,
                                      custom_translation_table=translations(), optimize=True, report=True,
                                      artifacts_dir=str(output.parent))
                graph = inspect_standard_onnx(output)
                onnx_record = capture_onnx_artifact(output)
                specs = {k:v for k,v in recipe['role_specs'].items() if k.startswith(role_component+'.')}
                record = dict(component=role_component, kind=kind, batch=b, frames=frames,
                              onnx=str(output), onnx_sha256=onnx_record['sha256'], onnx_artifact=onnx_record,
                              provenance=provenance, provenance_status=provenance['status'], plugins=[], graph=graph,
                              export_settings=dict(opset=20,dynamo=not a.legacy_export,external_data=True,
                                                   precision='int8_smoothquant_static_qdq_fp32_interfaces',vocoder_conv2d=a.vocoder_conv2d,vocoder_gemm=a.vocoder_gemm),
                              quantization_recipe=dict(scheme=recipe['scheme'],alpha=recipe.get('alpha'),
                                  calibration=file_record(calibration,'calibration_artifact'),role_manifest=manifest,
                                  role_specs_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()),
                              inputs=[dict(name=n,shape=list(v.shape),dtype=str(v.dtype)) for n,v in zip(names,inputs)],
                              output_names=outputs, validation=dict(output_finite=True,numerical_audit='pending_real_inputs'))
                if a.component in ('target','draft'): record['kv_limit']=80
                if a.component=='cfm':record['prompt_frames']=258
                output.with_suffix('.export.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(dict(status='exported',component=a.component,batch=b,onnx=str(output),nodes=graph['nodes'])),flush=True)
        finally:
            engine.close()


if __name__ == '__main__':
    main()
