"""Package selected static engines with their immutable runtime dependencies."""
import argparse,json,hashlib,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--raw-sources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.deployment.read_text());out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise ValueError('Use an empty package destination')
 (out/'.bundle_root').write_text('Selected InSpark static mixed deployment\n');mapping={}
 def copy(source,relative):
  src=Path(source).resolve();dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  try:dest.hardlink_to(src)
  except OSError:shutil.copyfile(src,dest)
  mapping[str(src)]='bundle://'+str(relative)
 calibs={'calibration':d['calibration'],**d.get('component_calibrations',{})}
 for name,source in calibs.items():
  if str(Path(source).resolve()) not in mapping:copy(source,Path('artifacts/current_release/calibration')/(name+'.json'))
 plans={}
 keys=[c+'_plan' for c in ['target','draft','context','prefill','latent','latent_suffix','cfm','vocoder']]+[k for k in ['cfm_microbatch_plan','vocoder_serial_plan','tail_target_plan','tail_draft_plan','tail_context_plan','middle_target_plan','middle_draft_plan','middle_context_plan'] if k in d]
 for key in keys:
  source=Path(d[key]).resolve();meta=json.loads(source.read_text());component=key.removesuffix('_plan');relative=Path('artifacts/current_release')/d['precision']/('b'+str(d['batch']))/component
  record=meta.get('quantization_recipe',{}).get('calibration',{})
  extra=record.get('path')
  if extra and str(Path(extra).resolve()) not in mapping:
   copy(extra,Path('artifacts/current_release/calibration')/(record['sha256']+'.json'))
  engine=Path(meta['engine']);engine=engine if engine.is_absolute() else source.parent/engine
  copy(engine,relative/'model.engine');mapping[str(source)]='bundle://'+str(relative/'model.plan.json');plans[relative/'model.plan.json']=meta
  inspector=source.with_name(source.name.replace('.plan.json','.inspector.json'))
  if inspector.exists():copy(inspector,relative/'model.inspector.json')
 official=Path(d['official_sources']).resolve()
 for file in official.rglob('*'):
  if file.is_file() and '__pycache__' not in file.parts:copy(file,Path('artifacts/official_trtllm_dspark')/file.relative_to(official))
 mapping[str(official)]='bundle://artifacts/official_trtllm_dspark'
 for name in ['s2mel.pth','bigvgan_generator.pt']:copy(a.raw_sources/name,Path('raw_sources')/name)
 def portable(v):
  if isinstance(v,str):return mapping.get(v,v)
  if isinstance(v,list):return [portable(x) for x in v]
  if isinstance(v,dict):return {k:portable(x) for k,x in v.items()}
  return v
 for path,meta in plans.items():
  meta=portable(meta);meta['engine']='model.engine';(out/path).write_text(json.dumps(meta,indent=2)+'\n')
 deploy=out/'artifacts/current_release/deployments'/f"{d['precision']}_b{d['batch']}.json";deploy.parent.mkdir(parents=True,exist_ok=True);deploy.write_text(json.dumps(portable(d),indent=2)+'\n')
 manifest=[]
 for file in sorted(out.rglob('*')):
  if file.is_file():
   with file.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
   manifest.append({'path':str(file.relative_to(out)),'bytes':file.stat().st_size,'sha256':h})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('selected bundle',len(manifest),'assets',sum(x['bytes'] for x in manifest),'bytes')
if __name__=='__main__':main()
