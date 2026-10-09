"""Inspect actual mixed precision tactics and the inherited FIR IO ABI."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.deployment.read_text());rows=[]
 keys=[c+'_plan' for c in ['target','draft','context','prefill','latent','latent_suffix','cfm','vocoder']]+[k for k in ['cfm_microbatch_plan','vocoder_serial_plan','middle_target_plan','middle_draft_plan','middle_context_plan','tail_target_plan','tail_draft_plan','tail_context_plan'] if k in d]
 for key in keys:
  c=key.removesuffix('_plan')
  path=Path(d[key]);meta=json.loads(path.read_text());layers=json.loads(path.with_name(path.name.replace('.plan.json','.inspector.json')).read_text())['Layers']
  family=meta['component']
  low=[n for n in layers if n.get('LayerType')=='gemm' and 'nvfp4_linear' in n['Name']]
  bad=[n['Name'] for n in low if 'xe2m1' not in n.get('TacticName','')]
  if bad or (family!='vocoder' and not low):raise ValueError(f'{c} native NVFP4 coverage failed: {bad}')
  fp8=[n for n in layers if n.get('LayerType')=='correlation' and 'e4m3' in n.get('TacticName','')]
  fir=[n for n in layers if 'ExistingTiledFIR' in n['Name']]
  if family=='vocoder':
   if not fp8 or len(fir)!=109:raise ValueError('Vocoder native Conv/FIR coverage changed')
   if not all(t['Datatype']=='Float' for n in fir for t in [*n['Inputs'],*n['Outputs']]):raise ValueError('FIR FP32 ABI changed')
  if family=='cfm' and not fp8:raise ValueError('CFM direct FP8 convolution coverage missing')
  rows.append({'component':c,'batch':meta['batch'],'engine_sha256':meta['sha256'],'native_fp4_gemms':len(low),'native_fp8_conv_implementation_layers':len(fp8),'fp32_fir_plugins':len(fir),'tiling':meta['tiling_optimization_level'],'actual_aux_streams':meta['actual_aux_streams']})
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
