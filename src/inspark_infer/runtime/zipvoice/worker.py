"""Isolated ZipVoice setup/download/prepare/inference worker."""
from __future__ import annotations
import argparse,importlib,json,os,sys
from pathlib import Path
from inspark_infer.build.zipvoice import BATCHES,check_workload,compatible,ensure,gpu_info,root,validate_bundle

def parser():
 p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
 for name in ['ensure','prepare','infer']:
  q=subs.add_parser(name);q.add_argument('--model',choices=['zipvoice'],default='zipvoice');q.add_argument('--precision',choices=['int8'],default='int8')
  if name=='ensure':q.add_argument('--batches',default='1,2,4,8,16,32,64');q.add_argument('--output-root',type=Path)
  else:q.add_argument('--batch',type=int,choices=BATCHES,required=True);q.add_argument('--bundle',type=Path);q.add_argument('--output',type=Path,required=True);q.add_argument('--prompt-wav',type=Path);q.add_argument('--prompt-text');q.add_argument('--text')
  q.add_argument('--gpu',type=int,default=1)
  if name=='infer':
   q.add_argument('--inputs',type=Path);q.add_argument('--workload',type=Path);q.add_argument('--seed',type=int,default=9100);q.add_argument('--warmup',type=int,default=1);q.add_argument('--repetitions',type=int,default=1);q.add_argument('--save-indices',type=int,nargs='+');q.add_argument('--functional-checks',action='store_true');q.add_argument('--dump-selected-state',action='store_true');q.add_argument('--disable-text-reuse',action='store_true')
 return p

def main(argv=None):
 args=parser().parse_args(argv)
 if args.gpu<0:raise ValueError('Physical GPU must be nonnegative')
 if args.command=='ensure':
  values=tuple(int(s) for s in args.batches.split(','))
  if len(set(values))!=len(values) or not values:raise ValueError('Batches must be unique')
  resolved=[]
  for b in values:
   bundle,m=ensure(b,args.gpu,args.output_root);resolved.append({'batch':b,'bundle':str(bundle),'bundle_id':m['bundle_id']})
  print(json.dumps({'status':'bundles_verified','model':'zipvoice','precision':'int8','bundles':resolved}));return 0
 if args.bundle:bundle=args.bundle.resolve();m=validate_bundle(bundle,args.batch)
 else:bundle,m=ensure(args.batch,args.gpu if args.command=='infer' else None)
 if args.command=='prepare' or args.inputs is None:
  if not all([args.prompt_wav,args.prompt_text,args.text]):raise ValueError('Supply --prompt-wav, --prompt-text and --text')
  from inspark_infer.models.zipvoice.frontend import prepare
  prepared=args.output if args.command=='prepare' else args.output/'condition.safetensors'
  w=prepare(bundle,m,args.prompt_wav,args.prompt_text,args.text,prepared)
  if args.command=='prepare':print(json.dumps({'status':'condition_prepared','condition':str(prepared),'workload':w}));return 0
  args.inputs=prepared
 else:
  if any([args.prompt_wav,args.prompt_text,args.text]):raise ValueError('Choose prepared inputs or audio/text, not both')
  if not args.workload:raise ValueError('--inputs requires --workload')
  w=json.loads(args.workload.read_text())
 check_workload(m,w)
 g=gpu_info(args.gpu);compatible(m,g)
 # Set the one-GPU mask before importing Torch or the route's plugin closure.
 os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
 import torch,tensorrt
 if torch.__version__!='2.11.0+cu130' or tensorrt.__version__!=m['hardware']['tensorrt']:raise ValueError('Pinned Torch/TensorRT runtime mismatch')
 from inspark_infer.runtime.device import GPULease
 with GPULease(args.gpu) as lease:
  args.output.mkdir(parents=True,exist_ok=True)
  inventory={'status':'integration_runtime_inventory','physical_gpu':args.gpu,'gpu_uuid':g['uuid'],'compute_capability':[8,9],'tensorrt':m['hardware']['tensorrt'],'workload':w,'origin_mapping_source_sha256':m['origin_mapping_source_sha256'],'engines':{}}
  inventory['plugin_package']=m.get('plugin_package',f'inspark_infer.ops.tensorrt.zipvoice.a1007.b{args.batch}')
  for key,x in m['engines'].items():inventory['engines'][key]={**x,'path':str(bundle/x['path']),'plugin_sources':{str(root()/p):v for p,v in m['plugin_sources'].items()}}
  inv=args.output/'engine-inventory.json';inv.write_text(json.dumps(inventory,indent=2)+'\n')
  route_module=m.get('application','a1007_delivery' if args.batch==32 else 'a1007')
  if route_module not in ('a1007','a1007_delivery','a1007_graph','a1007_delivery_graph'):raise ValueError('Unsupported ZipVoice application route')
  route=importlib.import_module(f'inspark_infer.runtime.zipvoice.routes.{route_module}')
  route_args=['--batch',str(args.batch),'--engine-manifest',str(inv),'--inputs',str(args.inputs),'--gpu',str(args.gpu),'--output',str(args.output),'--shared-context-workspace','--arena-istft','--include-input-transfer','--seed',str(args.seed),'--warmup',str(args.warmup),'--repetitions',str(args.repetitions)]
  policy=m.get('pcm_policy',{})
  if 'chunk' in policy:route_args+=['--pcm-chunk',str(policy['chunk'])]
  if 'workers' in policy:route_args+=['--pcm-workers',str(policy['workers'])]
  if args.save_indices is not None:route_args+=['--save-indices',*map(str,args.save_indices)]
  if args.functional_checks:route_args+=['--functional-only']
  if args.dump_selected_state:route_args+=['--dump-selected-state']
  if args.disable_text_reuse:route_args+=['--disable-text-reuse']
  sys.argv=[route.__file__,*route_args];route.main()
  report=args.output/'report.json';d=json.loads(report.read_text());d['preflight']={'initial_memory_mib':lease.initial_memory_mib,'initial_utilization':lease.initial_utilization,'shared_device':lease.shared};d['bundle_id']=m['bundle_id'];d['integration_route']='zipvoice_int8';report.write_text(json.dumps(d,indent=2)+'\n')
 return 0

if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as e:
  print(json.dumps({'status':'error','error_type':type(e).__name__,'details':str(e)}),file=sys.stderr);raise SystemExit(1)
