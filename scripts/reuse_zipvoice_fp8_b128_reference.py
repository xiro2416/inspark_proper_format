"""Reuse original full-B128 FP32 audio only under exact reference-input identity."""
import argparse,json,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from inspark_infer.runtime.zipvoice_fp8_b128.common import sha,write,private_report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',required=True);args=p.parse_args()
 import torch
 from safetensors.torch import load_file
 source=private_report(128,'004-fp32-reference-attention');baseline=json.loads(source.read_text())
 if sha(ROOT/'models/zipvoice/eager/model.safetensors')!=baseline['eager_weights_sha256']:raise ValueError('Original weights changed')
 report=dict(status='original_full_b128_fp32_audio_reused_exact_inputs_quality_pending',batch=128,eager_weights_sha256=baseline['eager_weights_sha256'],source_reference_audit_sha256=sha(source),tf32=False,cases=[],fixed_numeric_gate=False,semantics='Original full B128 FP32 audio reused only after exact condition, selected initial noise/text/speech/mask/time, row identity and original audio hash checks; candidate delta reported against inherited FP8, not mislabeled FP32')
 for c in baseline['cases']:
  i=c['case'];old=ROOT/f'outputs/fp8/b128/quality-attention/{i:03d}';new=ROOT/f'outputs/fp8/b128/quality-{args.variant}/{i:03d}'
  a=json.loads((old/'report.json').read_text());b=json.loads((new/'report.json').read_text())
  if b['status']!='complete_functional_only':raise ValueError('Candidate invocation incomplete')
  for key in ['input_sha256','diagnostic_noise_seed','diagnostic_selected_state_rows','shape','workload']:
   if a[key]!=b[key]:raise ValueError('Reference ownership changed: '+key)
  x=load_file(old/'selected-state.safetensors');y=load_file(new/'selected-state.safetensors')
  for key in ['initial_state','speech_condition','text_condition','time_grid','padding_mask']:
   if not torch.equal(x[key],y[key]):raise ValueError('Original reference inputs changed: '+key)
  delta=y['final_state'].float()-x['final_state'].float();case=copy.deepcopy(c)
  case.pop('state_relative_l2',None);case.pop('state_max_abs',None)
  case.update(reference_reused_exact_inputs=True,candidate_vs_inherited_state_relative_l2=float(delta.norm()/x['final_state'].float().norm()),candidate_vs_inherited_state_max_abs=float(delta.abs().max()))
  for row in case['rows']:
   if sha(row['fp32_wav'])!=row['fp32_wav_sha256']:raise ValueError('Original reference audio changed')
   path=new/f"{row['row']:04d}.wav";row.update(fp8_wav=str(path),fp8_wav_sha256=sha(path))
  report['cases'].append(case)
 write(private_report(128,'004-fp32-reference-'+args.variant),report)
 print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
if __name__=='__main__':main()
