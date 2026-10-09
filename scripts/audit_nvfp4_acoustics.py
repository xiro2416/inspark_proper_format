"""One actual native/hybrid head vs the same latest unquantized and NVFP4 models."""
import os
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=7);p.add_argument('--deployment',required=True);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from inspark_infer.runtime.config import load
    from inspark_infer.runtime.deployment import load as load_plan
    from inspark_infer.runtime.engine import Engine
    from inspark_infer.runtime.device import GPULease
    from inspark_infer.quantization.nvfp4 import install
    from benchmarks.unified_first_chunk import load_manifest,wave_cases,run_wave
    from benchmarks.benchmark_unified_first_chunk import prepare_references
    from scripts.audit_unified_acoustics import freeze_head_graphs,metrics
    manifest=load_manifest(args.manifest);plan=load_plan(args.deployment);cfg=load(args.config);batch=plan['batch'];cfg['max_batch']=batch
    with GPULease(args.gpu):
        engine=Engine(cfg)
        try:
            prepare_references(engine,manifest);engine.prepare_deployment(plan)
            before=dict(engine.head_graphs.hits);wave=run_wave(engine,wave_cases(manifest['splits']['evaluation'],batch,0),'nvfp4-acoustic-audit',admission_mode='batch')
            frozen=freeze_head_graphs(engine.head_graphs,batch,before)
        finally:engine.close()
        torch.cuda.empty_cache()
        reference=Engine(cfg)
        try:
            with torch.inference_mode(),torch.cuda.stream(reference.model.stream):
                inputs=tuple(v.cuda() for v in frozen['cfm_inputs']);mel_input=frozen['vocoder_inputs'][0].cuda()
                high_mel=reference.student(*inputs).cpu();high_pcm=reference.tts.bigvgan(mel_input).cpu()
                recipe=json.loads(Path(plan['calibration']).read_text());install(reference,recipe)
                same_mel=reference.student(*inputs).cpu();same_pcm=reference.tts.bigvgan(mel_input).cpu()
                actual_mel=frozen['cfm_output'];actual_pcm=frozen['vocoder_output']
                report={'waveform_diagnostics':{'unique_pcm_rows':int(torch.unique(actual_pcm.reshape(batch,-1),dim=0).shape[0]),'saturated_fraction':float((actual_pcm.abs()>=.999).float().mean()),'unique_reference_rows':int(torch.unique(same_pcm.reshape(batch,-1),dim=0).shape[0])},'scope':f'one real B{batch} head; reporting only, no numerical thresholds','routes':frozen['routes'],
                    'CFM_same_recipe':metrics(same_mel,actual_mel),'CFM_unquantized':metrics(high_mel,actual_mel),
                    'Vocoder_same_recipe':metrics(same_pcm,actual_pcm),'Vocoder_unquantized':metrics(high_pcm,actual_pcm),
                    'native_rounds':[row['rounds'] for row in wave['rows']]}
                args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
                assert all(report[k]['finite'] for k in ['CFM_same_recipe','CFM_unquantized','Vocoder_same_recipe','Vocoder_unquantized'])
        finally:reference.close()
if __name__=='__main__':main()
