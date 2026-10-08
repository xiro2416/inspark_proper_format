"""Hash-bound ZipVoice INT8 engine resolution, independent of IndexTTS2 bundles."""
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,uuid
from pathlib import Path,PurePosixPath

BATCHES=(1,2,4,8,16,32,64)
COMPONENTS={'fm','text','unique','vocos'}

def root():
 p=Path(os.environ.get('INSPARK_REPO_ROOT',Path(__file__).resolve().parents[3])).resolve()
 if not p.is_relative_to(Path('/workspace')):raise ValueError('Checkout must be under /workspace')
 return p

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_path(base,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe bundle path')
 out=(base/str(p)).resolve()
 if not out.is_relative_to(base.resolve()):raise ValueError('Bundle path escapes root')
 return out

def registry():return json.loads((root()/'configs/hardware/sm89/zipvoice_int8_registry.json').read_text())

def plugin_package(batch,value=None):
 import re
 value=f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{batch}' if value is None else value
 if not isinstance(value,str) or not re.fullmatch(rf'inspark_infer\.ops\.tensorrt\.zipvoice\.a1007\.b{batch}(?:_[a-z0-9_]+)?',value):
  raise ValueError('Plugin package does not match the model/target batch')
 return value

def validate_bundle(bundle,batch=None):
 bundle=Path(bundle).resolve();m=json.loads((bundle/'manifest.json').read_text())
 if m.get('schema')!=1 or m.get('model')!='zipvoice' or m.get('precision')!='int8' or m.get('batch') not in BATCHES:raise ValueError('Unsupported ZipVoice bundle')
 if batch is not None and m['batch']!=batch:raise ValueError('Wrong bundle batch')
 if set(m['engines'])!=COMPONENTS:raise ValueError('Incomplete ZipVoice components')
 if m.get('certified_for_production') is not False:raise ValueError('No production certification established')
 for name,x in m['files'].items():
  p=safe_path(bundle,name)
  if not p.is_file() or p.stat().st_size!=x['bytes'] or digest(p)!=x['sha256']:raise ValueError(f'Bundle integrity failure: {name}')
 for key,x in m['engines'].items():
  if x['path'] not in m['files'] or x['sha256']!=m['files'][x['path']]['sha256']:raise ValueError('Engine is not bound to file inventory')
  if not x.get('shape_profile'):raise ValueError('Missing engine shape profile')
 package=plugin_package(m['batch'],m.get('plugin_package'))
 prefix='src/'+package.replace('.','/')+'/'
 if any(not name.startswith(prefix) for name in m['plugin_sources']):raise ValueError('Plugin source is outside the selected target package')
 for name,expected in {**m['plugin_sources'],**m.get('support_sources',{})}.items():
  if digest(safe_path(root(),name))!=expected:raise ValueError(f'Plugin source mismatch: {name}')
 if digest(safe_path(root(),m['runner']))!=m['runner_sha256']:raise ValueError('Runner source mismatch')
 if 'application' in m and m['application']!=Path(m['runner']).stem:raise ValueError('Application route is not bound to the selected runner')
 return m

def check_workload(m,w):
 b=m['batch'];fixed={'batch':b,'prompt_frames':375,'steps':8,'t_shift':.5,'guidance':1.,'feat_scale':.1}
 if any(w.get(k)!=v for k,v in fixed.items()):raise ValueError('Workload changes the original model contract')
 if any(type(w.get(k)) is not int or w[k]<=0 for k in ['target_frames','total_frames','joint_tokens','padded_tokens']):raise ValueError('Invalid workload dimensions')
 if w['total_frames']!=375+w['target_frames'] or w['padded_tokens']!=w['joint_tokens']+1:raise ValueError('Inconsistent workload dimensions')
 for kind,x in m['engines'].items():
  if kind=='fm':shapes={k:[b,w['total_frames'],100] for k in ['x','text_condition','speech_condition']};shapes.update(t=[b,1,1],guidance_scale=[b,1,1],padding_mask=[b,w['total_frames']])
  elif kind=='vocos':shapes={'mel':[b,100,w['target_frames']]}
  else:
   rows=1 if kind=='unique' else b
   shapes={'token_ids':[rows,w['padded_tokens']],'token_lens':[rows],'features_lens':[rows],'frame_positions':[1,w['total_frames']]}
  for key,bounds in x['shape_profile'].items():
   dims=shapes.get(key)
   if dims is None:raise ValueError(f'Unknown binding: {kind}/{key}')
   lo,_,hi=bounds
   if len(dims)!=len(lo) or any(a<v or a>z for a,v,z in zip(dims,lo,hi)):raise ValueError(f'Shape outside {kind}/{key} profile: {dims}')
 return w

def gpu_info(index):
 if index<0:raise ValueError('Select one physical GPU')
 row=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=name,compute_cap,memory.total,uuid','--format=csv,noheader,nounits'],text=True).strip().split(',')
 return {'name':row[0].strip(),'sm':int(float(row[1])*10),'memory_total_mib':int(row[2]),'uuid':row[3].strip(),'physical_gpu':index}

def compatible(m,g):
 h=m['hardware']
 if any(g[k]!=h[k] for k in ('name','sm','memory_total_mib')):raise ValueError('GPU model/SM/memory class mismatch; rebuild and validate locally')

def ensure(batch,gpu=None,output_root=None):
 if batch not in BATCHES:raise ValueError(f'Supported INT8 batches: {BATCHES}')
 reg=registry()
 if str(batch) not in reg['bundles']:raise FileNotFoundError(f'No prepared A_1007 bundle for B{batch}; complete offline build and validation first')
 e=reg['bundles'][str(batch)]
 bundle=(root()/e['local_path']) if output_root is None else Path(output_root).resolve()/f'b{batch}'/e['bundle_id']
 if not bundle.is_relative_to(Path('/workspace')):raise ValueError('Bundle storage must stay within /workspace')
 if not bundle.exists():
  rev=reg.get('revision')
  if not isinstance(rev,str) or len(rev)!=40:raise FileNotFoundError('No local bundle or published pinned revision; no automatic rebuild')
  from huggingface_hub import HfApi,hf_hub_download
  token=os.getenv('HF_TOKEN')
  if not token:
   p=Path(os.getenv('HF_HOME',str(root()/'.cache/huggingface')))/'token'
   if p.is_file():token=p.read_text().strip()
  if not token:raise ValueError('Private HF credentials required')
  api=HfApi(endpoint='https://huggingface.co',token=token)
  if api.model_info(reg['repo_id'],revision=rev).private is not True:raise ValueError('Engine repository must be private')
  prefix=e['bundle_path'];used=[];active_endpoint=[os.getenv('HF_ENDPOINT','https://hf-mirror.com')]
  def download(name):
   endpoint=active_endpoint[0]
   try:p=hf_hub_download(repo_id=reg['repo_id'],revision=rev,filename=prefix+'/'+name,token=token,endpoint=endpoint)
   except Exception:
    if endpoint=='https://huggingface.co':raise
    endpoint='https://huggingface.co';active_endpoint[0]=endpoint;p=hf_hub_download(repo_id=reg['repo_id'],revision=rev,filename=prefix+'/'+name,token=token,endpoint=endpoint)
   used.append(endpoint);return Path(p)
  stage=bundle.parent/('.staging-'+uuid.uuid4().hex);stage.mkdir(parents=True)
  shutil.copy2(download('manifest.json'),stage/'manifest.json');m=json.loads((stage/'manifest.json').read_text())
  if m['bundle_id']!=e['bundle_id'] or m['batch']!=batch:raise ValueError('Pinned bundle identity mismatch')
  for name in m['files']:
   dst=safe_path(stage,name);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(download(name),dst)
  validate_bundle(stage,batch);(stage/'fetch-report.json').write_text(json.dumps({'revision':rev,'endpoints':sorted(set(used))})+'\n')
  stage.rename(bundle)
 m=validate_bundle(bundle,batch)
 if m['bundle_id']!=e['bundle_id']:raise ValueError('Registry bundle identity mismatch')
 if gpu is not None:compatible(m,gpu_info(gpu))
 return bundle,m
