"""Evaluate original FP32 at each full target batch with identical caller noise."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import BATCHES,sha,write,private_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=int,nargs='+',choices=BATCHES,default=list(BATCHES))
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--limit-quality',type=int)
    p.add_argument('--variant',choices=['native','attention','attentiongeo','attentionres','attentionff','attentionprotected'],default='native')
    args=p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!=str(args.gpu):raise ValueError('Single selected GPU required')
    from inspark_infer.runtime.device import GPULease
    with GPULease(args.gpu):run(args)


def run(args):
    import torch
    import soundfile as sf
    from safetensors.torch import load_file
    from vocos import Vocos
    from inspark_infer.models.zipvoice.weights import load_model
    from inspark_infer.models.zipvoice.pcm import pcm_quantized_exact
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_model(ROOT/'models/zipvoice','eager','cuda')
    vocos=Vocos.from_hparams(str(ROOT/'models/zipvoice/vocos/config.yaml'))
    vocos.load_state_dict(torch.load(ROOT/'models/zipvoice/vocos/pytorch_model.bin',map_location='cpu',weights_only=True),strict=True)
    vocos=vocos.eval().cuda()
    manifest=json.loads((ROOT/'outputs/fp8/data/manifest.json').read_text())
    cases=manifest['quality'][:args.limit_quality] if args.limit_quality else manifest['quality']
    for batch in args.batches:
        report=dict(status='running',batch=batch,eager_weights_sha256=sha(ROOT/'models/zipvoice/eager/model.safetensors'),
                    semantics='Original full-batch FP32 model, same original token-duration/masks, exact full-batch caller noise',
                    fixed_numeric_gate=False,tf32=False,cases=[])
        for i,case in enumerate(cases):
            quality_dir='quality' if args.variant=='native' else 'quality-'+args.variant
            fp8_folder=ROOT/f'outputs/fp8/b{batch}/{quality_dir}/{i:03d}'
            evidence=json.loads((fp8_folder/'report.json').read_text())
            selected=load_file(str(fp8_folder/'selected-state.safetensors'))
            rows=evidence['diagnostic_selected_state_rows']
            seed=evidence['diagnostic_noise_seed'];frames=case['total_frames']
            data=load_file(case['condition'])
            tokens=data['token_ids'].expand(batch,-1)[:,:-1].tolist()
            lengths=torch.full((batch,),frames,device='cuda',dtype=torch.int64)
            speech=torch.nn.functional.pad(data['prompt_mel'].expand(batch,-1,-1)*.1,(0,0,0,frames-375)).cuda()
            with torch.inference_mode():
                text,mask=model.forward_text_train(tokens,lengths)
                initial=torch.empty(batch,frames,100,device='cuda').normal_(generator=torch.Generator(device='cuda').manual_seed(seed))
                assert torch.equal(initial[rows].cpu(),selected['initial_state'])
                assert torch.equal(speech[rows].cpu(),selected['speech_condition'])
                assert torch.equal(mask[rows].cpu(),selected['padding_mask'])
                state=initial.clone();grid=torch.linspace(0,1,9,device='cuda');grid=.5*grid/(1-.5*grid)
                assert torch.equal(grid.cpu(),selected['time_grid'])
                for k in range(8):
                    velocity=model.forward_fm_decoder(grid[k].expand(batch).reshape(batch,1,1),state,text,speech,mask,
                                                       torch.ones(batch,1,1,device='cuda'))
                    state.add_(velocity*(grid[k+1]-grid[k]))
                wave=vocos.decode((state[:,375:,:].permute(0,2,1)/.1).contiguous()).clamp(-1,1)
                rms=data['prompt_rms'].expand(batch).cuda()[:,None]
                wave=torch.where(rms<.1,wave*rms/.1,wave)
                if not torch.isfinite(state).all() or not torch.isfinite(wave).all():
                    raise RuntimeError('Original reference produced nonfinite output')
                actual=selected['final_state'].float();expected=state[rows].cpu()
                error=actual-expected
                text_error=selected['text_condition']-text[rows].cpu()
                row_report=[]
                reference_dir='fp32-reference' if args.variant=='native' else 'fp32-reference-'+args.variant
                out=ROOT/f'outputs/fp8/b{batch}/{reference_dir}/{i:03d}';out.mkdir(parents=True,exist_ok=True)
                for row in rows:
                    pcm=pcm_quantized_exact(wave[row].float().cpu().numpy(),cache_fades=True)
                    if not len(pcm):raise RuntimeError('Empty reference PCM')
                    path=out/f'{row:04d}.wav';sf.write(path,pcm,24000,subtype='PCM_16')
                    row_report.append(dict(row=row,fp32_wav=str(path),fp32_wav_sha256=sha(path),
                                           fp8_wav=str(fp8_folder/f'{row:04d}.wav'),
                                           fp8_wav_sha256=sha(fp8_folder/f'{row:04d}.wav')))
                report['cases'].append(dict(case=i,language=case['language'],frames=frames,
                    text=case['text'],reference_wav=case['reference_wav'],reference_sha256=case['reference_sha256'],
                    condition_sha256=case['condition_sha256'],seed=seed,noise_and_speech_mask_time_exact=True,
                    state_relative_l2=float(error.norm()/expected.norm().clamp_min(1e-20)),
                    state_max_abs=float(error.abs().max()),text_condition_relative_l2=float(text_error.norm()/text[rows].cpu().norm().clamp_min(1e-20)),rows=row_report))
                del text,mask,speech,initial,state,velocity,wave,expected
            write(private_report(batch,'004-fp32-reference'+('' if args.variant=='native' else '-'+args.variant)),report)
            print(json.dumps({'event':'reference_case_done','batch':batch,'case':i}),flush=True)
        report['status']='fp32_target_reference_complete_quality_metrics_pending'
        write(private_report(batch,'004-fp32-reference'+('' if args.variant=='native' else '-'+args.variant)),report)


if __name__=='__main__':main()
