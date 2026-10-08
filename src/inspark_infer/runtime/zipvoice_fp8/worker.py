"""FP8 prepare/ensure/infer entrypoint with per-call physical GPU binding."""
import argparse
import json
import os
from pathlib import Path
import sys

from inspark_infer.build.zipvoice_fp8 import ensure,validate_bundle,compatible,gpu_info,check_workload
from .common import BATCHES,root,write


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('ensure','prepare','infer'):
        q=sub.add_parser(name);q.add_argument('--model',choices=['zipvoice'],default='zipvoice')
        q.add_argument('--precision',choices=['fp8'],default='fp8');q.add_argument('--gpu',type=int,default=3)
        if name=='ensure':
            q.add_argument('--batches',default='1,2,4,8,16,32,64');q.add_argument('--output-root',type=Path)
        else:
            q.add_argument('--batch',type=int,choices=BATCHES,required=True);q.add_argument('--bundle',type=Path)
            q.add_argument('--output',type=Path,required=True);q.add_argument('--prompt-wav',type=Path)
            q.add_argument('--prompt-text');q.add_argument('--text')
        if name=='infer':
            q.add_argument('--inputs',type=Path);q.add_argument('--workload',type=Path)
            q.add_argument('--seed',type=int,default=9100);q.add_argument('--warmup',type=int,default=1)
            q.add_argument('--repetitions',type=int,default=1);q.add_argument('--save-indices',type=int,nargs='+')
            q.add_argument('--functional-checks',action='store_true');q.add_argument('--dump-selected-state',action='store_true')
            q.add_argument('--disable-text-reuse',action='store_true');q.add_argument('--minimum-seconds',type=float,default=0)
    args=p.parse_args(argv)
    if args.gpu<0:raise ValueError('Select one physical GPU')
    if args.command=='ensure':
        batches=tuple(int(x) for x in args.batches.split(','))
        if not batches or len(set(batches))!=len(batches):raise ValueError('Batches must be nonempty and unique')
        result=[]
        for batch in batches:
            bundle,m=ensure(batch,args.gpu,args.output_root)
            result.append(dict(batch=batch,bundle=str(bundle),bundle_id=m['bundle_id']))
        print(json.dumps(dict(status='fp8_bundles_verified',bundles=result)));return 0
    if args.bundle:bundle=args.bundle.resolve();m=validate_bundle(bundle,args.batch)
    else:bundle,m=ensure(args.batch,args.gpu if args.command=='infer' else None)
    if args.command=='prepare' or not args.inputs:
        if not all((args.prompt_wav,args.prompt_text,args.text)):raise ValueError('Supply reference WAV, accurate transcription and complete target text')
        from inspark_infer.models.zipvoice.frontend import prepare
        prepared=args.output if args.command=='prepare' else args.output/'condition.safetensors'
        w=prepare(bundle,m,args.prompt_wav,args.prompt_text,args.text,prepared)
        if args.command=='prepare':print(json.dumps(dict(status='fp8_condition_prepared',workload=w)));return 0
        args.inputs=prepared
    else:
        if any((args.prompt_wav,args.prompt_text,args.text)) or not args.workload:
            raise ValueError('Choose prepared inputs with --workload, or the reference/text trio')
        w=json.loads(args.workload.read_text())
    check_workload(m,w)
    g=gpu_info(args.gpu);compatible(m,g)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    import torch,tensorrt
    if torch.__version__!=m['runtime']['torch'] or tensorrt.__version__!=m['hardware']['tensorrt']:
        raise ValueError('Pinned FP8 runtime mismatch')
    from inspark_infer.runtime.device import GPULease
    with GPULease(args.gpu) as lease:
        inventory=dict(physical_gpu=args.gpu,gpu_uuid=g['uuid'],compute_capability=[12,0],
                       tensorrt=tensorrt.__version__,power_limit_w=m['hardware']['power_limit_w'],
                       workload=w,engines={k:{**v,'path':str(bundle/v['path'])} for k,v in m['engines'].items()},
                       component_precisions=m['component_precisions'],origin_mapping_source_sha256=m['mapping_source_sha256'])
        inventory['plugin_packages']=m.get('plugin_packages',[])
        inv=args.output/'engine-inventory.json';write(inv,inventory)
        command=['--batch',str(args.batch),'--engine-manifest',str(inv),'--gpu',str(args.gpu),
                 '--inputs',str(args.inputs),'--output',str(args.output),'--shared-context-workspace',
                 '--arena-istft','--include-input-transfer','--seed',str(args.seed),
                 '--warmup',str(args.warmup),'--repetitions',str(args.repetitions),
                 '--pcm-chunk',str(m['pcm_policy']['chunk']),'--pcm-workers',str(m['pcm_policy']['workers']),
                 '--minimum-seconds',str(args.minimum_seconds)]
        if args.save_indices is not None:command+=['--save-indices',*map(str,args.save_indices)]
        for key,flag in [('functional_checks','--functional-only'),('dump_selected_state','--dump-selected-state'),('disable_text_reuse','--disable-text-reuse')]:
            if getattr(args,key):command.append(flag)
        application=m.get('application','route')
        if application not in ('route','attention_route'):raise ValueError('Unsupported FP8 application')
        import importlib
        route=importlib.import_module('.'+application,__package__)
        sys.argv=[route.__file__,*command];route.main()
        path=args.output/'report.json';report=json.loads(path.read_text())
        report.update(bundle_id=m['bundle_id'],preflight=dict(initial_memory_mib=lease.initial_memory_mib,
                      initial_utilization=lease.initial_utilization,shared_device=lease.shared))
        write(path,report)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        print(json.dumps(dict(status='error',error_type=type(e).__name__,details=str(e))),file=sys.stderr)
        raise SystemExit(1)
